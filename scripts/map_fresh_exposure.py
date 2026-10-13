"""Map provisional trained water-state changes and one hypothetical access set.

Only saved, checksum-bound water products and the plan's historical OSM extract
are read. No model, network, reference labels, observed closures or debris map
is consumed or produced by this stage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from affine import Affine
import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import GeometryCollection, Polygon, box, mapping, shape
from shapely.ops import transform, unary_union

from akash_flood.access import build_road_graph
from akash_flood.buildings import building_footprints
from akash_flood.catalog import utc, validate_bbox
from akash_flood.fresh_case import event_boundary, normalized_utc, reject_source_path
from akash_flood.nearest_access import analyze_nearest_access
from akash_flood.osm import named_points
from akash_flood.snap_products import validate_grid
from map_radar_exposure import Coverage, EPSILON, feature_collection, physical_segments, pixel_polygons
from prove_access_bounds import verify_certificates

ROOT = Path(__file__).resolve().parents[1]
MASK_FILES = {"aoi": "aoi-mask.tif", "assessed": "valid-mask.tif",
              "before": "before-water.tif", "after": "after-water.tif",
              "new": "new-water-candidates.tif", "lost": "lost-water-candidates.tif"}
ALL_WATER_FILES = set(MASK_FILES.values()) | {"before-water-score.tif", "after-water-score.tif"}
COUNT_NAMES = {"aoi_pixels", "assessed_pixels", "unknown_pixels", "before_water_pixels",
               "after_water_pixels", "new_water_candidate_pixels", "lost_water_candidate_pixels",
               "unchanged_water_pixels"}
BASES = {"new": "Predicted nonwater before to predicted water after on the joint assessment domain; unvalidated water-state transfer",
         "lost": "Predicted water before to predicted nonwater after on the joint assessment domain; unvalidated water-state transfer"}
SCENARIO = {
    "scenario_is_hypothetical": True,
    "uncertainty_rule": "Every source physical motor-road segment with positive-area (>1e-6 m2) intersection between its 10 m half-width association corridor and a new-water candidate is jointly open/closed in all its source driving directions.",
    "certain_closed_edge_ids": [], "confirmed_closures": 0,
    "all_other_edges": "Retain original baseline OSM edges in every scenario, including unknown and outside-AOI coverage.",
    "state_model": "Every combination of selected physical-segment states is allowed; no probabilities or observed closure assignments.",
    "destination": "Fix each settlement's nearest baseline-reachable node-tagged town/city/hospital by directed road length plus both fixed point-to-node association distances; tie by string destination ID. Never switch targets after closures.",
    "point_association_limit_m": 500,
    "exclusions": ["Unspecified failures, water-depth/flow damage, debris and unmapped hazards are outside this scenario set.",
                   "Unknown radar and absence of candidate overlap cannot establish physical passability.",
                   "Mapped graph routes do not validate facility operability, entrances, off-road point connections or road restrictions."],
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def checked_record(item, parent, label):
    path = Path(item["path"])
    if not path.is_absolute():
        path = Path(parent) / path
    path = path.resolve()
    reject_source_path(path)
    if (not path.is_file() or path.stat().st_size != item["bytes"]
            or sha256(path) != item["sha256"]):
        raise ValueError("Changed checksum-bound " + label)
    return path


def planned_aoi(bbox, grid):
    west, south, east, north = validate_bbox(bbox)
    project = Transformer.from_crs(4326, grid[0], always_xy=True).transform
    perimeter = []
    for start, stop in (((west, south), (east, south)), ((east, south), (east, north)),
                        ((east, north), (west, north)), ((west, north), (west, south))):
        for fraction in np.linspace(0, 1, 41, endpoint=False):
            perimeter.append(project(start[0] + fraction * (stop[0]-start[0]),
                                     start[1] + fraction * (stop[1]-start[1])))
    return geometry_mask([mapping(Polygon(perimeter))], out_shape=(grid[3], grid[2]),
                         transform=grid[1], invert=True, all_touched=False)


def validate_temporal_arrays(arrays):
    if set(arrays) != set(MASK_FILES):
        raise ValueError("Require all six named temporal masks")
    shape = arrays["aoi"].shape
    if any(value.shape != shape for value in arrays.values()):
        raise ValueError("Temporal mask dimensions differ")
    for name, values in arrays.items():
        if values.dtype != np.uint8 or not np.isin(values, (0, 1) if name in ("aoi", "assessed") else (0, 1, 255)).all():
            raise ValueError("Unexpected temporal mask encoding or dtype")
    aoi, assessed = arrays["aoi"] == 1, arrays["assessed"] == 1
    if not aoi.any() or np.any(assessed & ~aoi):
        raise ValueError("Assessment must be within a nonempty AOI")
    for name in ("before", "after", "new", "lost"):
        if not np.array_equal(arrays[name] != 255, assessed):
            raise ValueError("All temporal labels require the identical joint assessment domain; unknown elsewhere")
    before, after = arrays["before"], arrays["after"]
    new = assessed & (before == 0) & (after == 1)
    lost = assessed & (before == 1) & (after == 0)
    if (not np.array_equal(arrays["new"], np.where(assessed, new, 255).astype("uint8"))
            or not np.array_equal(arrays["lost"], np.where(assessed, lost, 255).astype("uint8"))):
        raise ValueError("New/lost candidate masks disagree with before/after water-state predictions")
    return {"aoi_pixels": int(aoi.sum()), "assessed_pixels": int(assessed.sum()),
            "unknown_pixels": int((aoi & ~assessed).sum()),
            "before_water_pixels": int(((before == 1) & assessed).sum()),
            "after_water_pixels": int(((after == 1) & assessed).sum()),
            "new_water_candidate_pixels": int(new.sum()), "lost_water_candidate_pixels": int(lost.sum()),
            "unchanged_water_pixels": int((assessed & (before == 1) & (after == 1)).sum())}


def read_water(manifest_path, plan_path, plan):
    """Pin all eight output hashes before any raster decoding; recount six masks."""
    manifest_path = Path(manifest_path).resolve()
    reject_source_path(manifest_path)
    source_record = record(manifest_path)
    water = json.loads(manifest_path.read_text())
    if (water.get("schema") != "akash-fresh-water-v1" or water.get("available") is not True
            or water.get("reference_labels_loaded") is not False or water.get("case") != plan["case"]
            or water.get("label_encoding") != {"0": "predicted nonwater", "1": "predicted water", "255": "unknown"}
            or set(water.get("outputs", {})) != ALL_WATER_FILES):
        raise ValueError("Require a complete, same-case fresh water-state manifest with explicit unknown labels")
    records = [source_record]
    paths = {}
    for name, item in water["outputs"].items():
        path = checked_record(item, manifest_path.parent, "water output")
        if path.name != name or path.parent != manifest_path.parent:
            raise ValueError("Water output path differs from its named manifest output")
        paths[name] = path
        records.append(record(path))
    # Model bytes are checksum-checked for provenance, never unpickled here.
    record_paths = {}
    for label, item in (("processing record", water["processing_record"]), ("water policy", water["policy"]),
                        ("model", water["model"]), ("model freeze", water["model"]["freeze"])):
        path = checked_record(item, manifest_path.parent, label)
        record_paths[label] = path
        records.append(record(path))
    processing = json.loads(record_paths["processing record"].read_text())
    if (processing.get("schema") != "akash-fresh-s1-processing-v1" or processing.get("status") != "completed"
            or not processing.get("completed_at") or processing.get("case") != plan["case"]
            or checked_record(processing["source_plan"], record_paths["processing record"].parent, "processing plan") != plan_path.resolve()
            or processing["source_plan"]["sha256"] != sha256(plan_path)
            or any(processing["scenes"][scene].get("status") != "completed"
                   or processing["scenes"][scene]["source_datetime"] != water["source_dates"][scene] for scene in ("before", "after"))):
        raise ValueError("Water products are not bound to completed same-plan processing and source dates")
    for item in water["code"]:
        path = checked_record(item, manifest_path.parent, "water source code")
        records.append(record(path))
    arrays, grid = {}, None
    for name, filename in MASK_FILES.items():
        with rasterio.open(paths[filename]) as dataset:
            current = (dataset.crs, dataset.transform, dataset.width, dataset.height)
            validate_grid(current, expected_epsg=plan["projection"]["epsg"])
            if dataset.count != 1 or (grid is not None and current != grid):
                raise ValueError("Temporal water products must share exactly one native metric grid; no resampling")
            if dataset.nodata != (0 if name in ("aoi", "assessed") else 255):
                raise ValueError("Temporal raster nodata differs from the declared encoding")
            grid = current
            arrays[name] = dataset.read(1)
    declared = water["grid"]
    common = processing["common_grid"]
    if (declared.get("resampling_used_for_date_alignment") is not False
            or declared.get("resolution_m") != 10
            or CRS.from_user_input(declared["crs"]) != CRS.from_user_input(grid[0])
            or not np.array_equal(declared["transform"][:6], list(grid[1])[:6])
            or declared["shape"] != [grid[3], grid[2]]
            or CRS.from_user_input(common["crs"]) != CRS.from_user_input(grid[0])
            or not np.array_equal(common["transform"][:6], list(grid[1])[:6])
            or common["shape"] != [grid[3], grid[2]] or common.get("resampling_used") is not False):
        raise ValueError("Water raster grid differs from its manifest or native common-grid provenance")
    counts = validate_temporal_arrays(arrays)
    if (set(water["counts"]) != COUNT_NAMES
            or any(isinstance(value, bool) or not isinstance(value, int) for value in water["counts"].values())
            or counts != water["counts"]):
        raise ValueError("All eight temporal counts must match a fresh raster recount")
    if not np.array_equal(arrays["aoi"] == 1, planned_aoi(plan["case"]["bbox"], grid)):
        raise ValueError("Water AOI mask differs from the case's native-grid pixel-center AOI")
    return arrays, grid, counts, water, records


def read_historical_osm(plan_path, plan):
    osm_plan = plan["osm"]
    if osm_plan.get("status") != "saved_historical_extract_checked" or len(osm_plan.get("local_inputs", [])) != 2:
        raise ValueError("The case plan has no checksum-bound historical OSM extract and provenance")
    event_start, _, snapshot = event_boundary(plan["case"]["event_date"], plan["case"]["osm_snapshot"])
    if utc(osm_plan["snapshot"]) != snapshot:
        raise ValueError("Historical OSM cutoff differs from the requested case")
    paths = [checked_record(item, plan_path.parent, "historical OSM input") for item in osm_plan["local_inputs"]]
    osm, provenance = [json.loads(path.read_text()) for path in paths]
    if (utc(provenance["snapshot"]) != snapshot or snapshot >= event_start
            or provenance.get("event_date", plan["case"]["event_date"]) != plan["case"]["event_date"]
            or not box(*validate_bbox(provenance["bbox"])).covers(box(*validate_bbox(plan["case"]["bbox"])))
            or f'[date:"{normalized_utc(snapshot)}"]' not in provenance.get("query", "")):
        raise ValueError("Historical OSM provenance has an invalid source cutoff, case date or extraction extent")
    canonical_legacy = hashlib.sha256(json.dumps(osm, sort_keys=True).encode()).hexdigest()
    if (provenance.get("sha256") not in (sha256(paths[0]), canonical_legacy)
            or osm.get("remark") or not osm.get("elements")
            or provenance.get("element_count", len(osm["elements"])) != len(osm["elements"])):
        raise ValueError("Historical OSM content differs from its extraction provenance")
    for element in osm["elements"]:
        if element.get("timestamp") and utc(element["timestamp"]) > snapshot:
            raise ValueError("Historical OSM contains a source edit newer than its cutoff")
    records = [record(path) for path in paths]
    return osm, {"snapshot": normalized_utc(snapshot), "source": records[0], "provenance": records[1],
                 "extraction_bbox": provenance["bbox"], "attribution": "© OpenStreetMap contributors; ODbL",
                 "history_evidence": provenance.get("history_evidence", "historical query provenance only"),
                 "post_event_osm_used": False, "independent_historical_server_authenticity_validated": False}, records


class WaterStateCoverage(Coverage):
    """Reuse native-cell intersections while exposing only new/lost semantics."""
    def __init__(self, geometries):
        super().__init__({"aoi": geometries["aoi"], "assessed": geometries["assessed"],
                          "change": geometries["new"], "water": geometries["lost"]})

    def describe(self, geometry, *, measure="area"):
        result = super().describe(geometry, measure=measure)
        names = {"change_candidate": "new_water_candidate", "water_candidate": "lost_water_candidate",
                 "assessed_no_change_candidate": "assessed_no_new_water_candidate",
                 "assessed_no_water_candidate": "assessed_no_lost_water_candidate",
                 "fully_assessed_no_change_candidate": "fully_assessed_no_new_water_candidate",
                 "fully_assessed_no_water_candidate": "fully_assessed_no_lost_water_candidate"}
        result = {names.get(key, key): value for key, value in result.items()}
        result["fractions_of_total"] = {names.get(key, key): value for key, value in result["fractions_of_total"].items()}
        result["interpretation"] = "Provisional water-state candidate geometry only. Unknown and outside-AOI coverage remain unknown; no candidate is not verified absence of flood/debris or physical passability."
        return result


def mask_regions(arrays, grid):
    masks = {"aoi": arrays["aoi"] == 1, "assessed": arrays["assessed"] == 1,
             "new": arrays["new"] == 1, "lost": arrays["lost"] == 1}
    polygons = {name: pixel_polygons(value, grid[1]) for name, value in masks.items()}
    unions = {name: unary_union(value) if value else GeometryCollection() for name, value in polygons.items()}
    return masks, polygons, unions


def overlap_features(osm, grid, regions):
    project = Transformer.from_crs(4326, grid[0], always_xy=True).transform
    unproject = Transformer.from_crs(grid[0], 4326, always_xy=True).transform
    coverage = WaterStateCoverage(regions)
    graph = build_road_graph(osm)
    roads, road_audit = physical_segments(graph)
    corridors = []
    for feature in roads["features"]:
        line = transform(project, shape(feature["geometry"]))
        corridor = line.buffer(10, cap_style="flat", join_style="mitre")
        properties = feature["properties"]
        properties.update(centerline_coverage=coverage.describe(line, measure="length"), coverage=coverage.describe(corridor),
                          road_buffer_half_width_m=10, observed_road_width=False, geometry_association_only=True,
                          candidate_overlap_basis=BASES, verified_no_hazard=False)
        properties["new_water_candidate_overlap"] = properties["coverage"]["new_water_candidate"] > EPSILON
        properties["lost_water_candidate_overlap"] = properties["coverage"]["lost_water_candidate"] > EPSILON
        properties["hypothetical_access_state"] = "uncertain_open_or_closed" if properties["new_water_candidate_overlap"] else "retain_baseline_edge_assumption"
        properties["physical_passability_observed"] = False
        corridors.append({"type": "Feature", "id": feature["id"], "geometry": mapping(transform(unproject, corridor)),
                          "properties": {"physical_segment_id": feature["id"], "buffer_half_width_m": 10,
                              "buffer_end_caps": "flat", "buffer_join_style": "mitre", "observed_road_width": False,
                              "coverage": properties["coverage"], "road_closure_assigned": False,
                              "hypothetical_access_state": properties["hypothetical_access_state"]}})
    buildings, building_audit = building_footprints(osm)
    for feature in buildings["features"]:
        polygon = transform(project, shape(feature["geometry"]))
        properties = feature["properties"]
        properties.update(coverage=coverage.describe(polygon), candidate_overlap_basis=BASES,
                          geometry_association_only=True, confirmed_damage=False, verified_no_hazard=False)
        properties["new_water_candidate_overlap"] = properties["coverage"]["new_water_candidate"] > EPSILON
        properties["lost_water_candidate_overlap"] = properties["coverage"]["lost_water_candidate"] > EPSILON
    collections = {"roads": roads, "road-corridors": feature_collection(corridors),
                   "bridges": feature_collection([item for item in roads["features"] if item["properties"]["bridge"]]),
                   "buildings": buildings}
    return collections, {"roads": road_audit, "buildings": building_audit}, graph


def summarize_features(collection):
    coverages = [feature["properties"]["coverage"] for feature in collection["features"]]
    return {"mapped_feature_count": len(coverages),
            "features_outside_aoi": sum(item["within_aoi"] <= EPSILON for item in coverages),
            "features_partly_outside_aoi": sum(item["outside_aoi"] > EPSILON and item["within_aoi"] > EPSILON for item in coverages),
            "features_with_unknown_water_state": sum(item["unassessed"] > EPSILON for item in coverages),
            "features_with_assessed_water_state": sum(item["assessed"] > EPSILON for item in coverages),
            "features_fully_assessed": sum(item["status"] == "fully_assessed" for item in coverages),
            "features_with_new_water_candidate_overlap": sum(item["new_water_candidate"] > EPSILON for item in coverages),
            "features_with_lost_water_candidate_overlap": sum(item["lost_water_candidate"] > EPSILON for item in coverages),
            "features_fully_assessed_no_new_water_candidate": sum(item["fully_assessed_no_new_water_candidate"] for item in coverages),
            "features_fully_assessed_no_lost_water_candidate": sum(item["fully_assessed_no_lost_water_candidate"] for item in coverages)}


def candidate_features(polygons, masks, grid):
    unproject = Transformer.from_crs(grid[0], 4326, always_xy=True).transform
    collections, summary = {}, {}
    pixel_area = abs(grid[1].determinant)
    for kind in ("new", "lost"):
        features = [{"type": "Feature", "id": f"{kind}-water-candidate-{index}",
                     "geometry": mapping(transform(unproject, polygon)),
                     "properties": {"candidate_kind": kind + "_water_state", "candidate_basis": BASES[kind],
                                    "native_grid_area_m2": polygon.area, "confirmed_flood": False,
                                    "confirmed_debris": False, "confirmed_damage": False, "road_closure_assigned": False}}
                    for index, polygon in enumerate(polygons[kind])]
        count = int(masks[kind].sum())
        area = sum(polygon.area for polygon in polygons[kind])
        if not math.isclose(area, count * pixel_area, abs_tol=1e-5, rel_tol=1e-10):
            raise ValueError("Water-state polygons changed native pixel-cell area")
        collections[kind + "-water-candidates"] = feature_collection(features)
        summary[kind] = {"candidate_pixel_count": count, "polygon_component_count": len(features),
                         "candidate_area_m2": area, "native_grid_pixel_area_m2": pixel_area,
                         "candidate_basis": BASES[kind], "confirmed_hazard": False}
    return collections, summary


def access_scenario(graph, osm, roads):
    """Candidate-selected uncertainty groups, with no certainly closed roads."""
    groups = {}
    for feature in roads["features"]:
        properties = feature["properties"]
        if properties["new_water_candidate_overlap"]:
            groups[feature["id"]] = sorted(item["edge_id"] for item in properties["source_directed_edges"])
    settlements, destinations = named_points(osm)
    result = analyze_nearest_access(graph, settlements, destinations, uncertain_links=groups, max_snap_m=500)
    checks = verify_certificates(graph, result)
    fixed_target_checks = {"paths": 0, "cuts": 0}
    for row in result["settlements"]:
        selected = row["nearest_baseline_destination"]
        if selected:
            for key in ("path_witness", "optimistic_path_witness"):
                witness = row[key]
                if witness:
                    if (witness["destination_id"] != selected["destination_id"]
                            or witness["destination_node_id"] != selected["destination_snap_node_id"]):
                        raise AssertionError("Route certificate switched away from the fixed baseline target")
                    fixed_target_checks["paths"] += 1
            for key in ("unreachable_certificate", "pessimistic_cut_certificate"):
                certificate = row[key]
                if certificate:
                    if certificate["mapped_destination_node_ids"] != [selected["destination_snap_node_id"]]:
                        raise AssertionError("Cut certificate does not concern the fixed baseline target")
                    fixed_target_checks["cuts"] += 1
    if result["certain_closed_edge_ids"]:
        raise AssertionError("This scenario assigns no certainly closed or confirmed closed edges")
    result.pop("hazard_inputs_used", None)  # The adapter's original bridge-only scope differs from this caller.
    result.update(schema="akash-fresh-access-scenario-v1", scenario=SCENARIO,
                  satellite_water_state_candidates_used_for_hypothetical_scenario=True,
                  observed_hazard_or_closures_used=False, certificate_checks=checks,
                  fixed_destination_certificate_checks=fixed_target_checks,
                  named_settlement_count=len(settlements), node_tagged_destination_count=len(destinations))
    return result, settlements, destinations


def point_features(points, *, proof_rows=None, bbox=None):
    rows = {row["id"]: row for row in proof_rows or []}
    return feature_collection([{"type": "Feature", "id": point["id"],
                               "geometry": {"type": "Point", "coordinates": [point["lon"], point["lat"]]},
                               "properties": {**point, "within_requested_aoi": box(*bbox).covers(shape({"type": "Point", "coordinates": [point["lon"], point["lat"]]})),
                                              "hypothetical_access_result": rows.get(point["id"]),
                                              "physical_passability_observed": False,
                                              "facility_operability_validated": False}}
                              for point in points])


def run(water_manifest, plan_path, out):
    water_manifest, plan_path, out = [Path(path).resolve() for path in (water_manifest, plan_path, out)]
    if out.exists():
        raise FileExistsError("Use a fresh exposure output directory; preserve earlier science and proofs")
    reject_source_path(plan_path)
    started = time.perf_counter()
    plan_record = record(plan_path)
    plan = json.loads(plan_path.read_text())
    arrays, grid, counts, water, water_records = read_water(water_manifest, plan_path, plan)
    osm, historical_osm, osm_records = read_historical_osm(plan_path, plan)
    inputs = [plan_record, *water_records, *osm_records]
    out.mkdir(parents=True)
    write_json(out / "scenario-policy.json", {**SCENARIO, "recorded_before_feature_overlap_and_access_at": datetime.now(timezone.utc).isoformat(),
                                              "water_source": record(water_manifest), "case_id": plan["case"]["case_id"],
                                              "positive_overlap_area_tolerance_m2": EPSILON})
    masks, polygons, regions = mask_regions(arrays, grid)
    collections, audits, graph = overlap_features(osm, grid, regions)
    candidate_collections, candidates = candidate_features(polygons, masks, grid)
    collections.update(candidate_collections)
    proof, settlements, destinations = access_scenario(graph, osm, collections["roads"])
    proof.update(case=plan["case"], temporal_counts=counts, water_source=record(water_manifest),
                 source_plan=plan_record, historical_osm=historical_osm, source_feature_audits=audits)
    proof_path = out / "access-proof.json"
    write_json(proof_path, proof)
    collections["settlements"] = point_features(settlements, proof_rows=proof["settlements"], bbox=plan["case"]["bbox"])
    collections["destinations"] = point_features(destinations, bbox=plan["case"]["bbox"])
    for name, collection in collections.items():
        write_json(out / (name + ".geojson"), collection)
    selection_counts = Counter(row["nearest_baseline_destination"]["destination_id"] for row in proof["settlements"] if row["nearest_baseline_destination"])
    scenario_summary = {"schema": proof["schema"], "scenario_is_hypothetical": True,
        "status_counts": proof["status_counts"], "n_uncertain_physical_segments": len(proof["uncertain_links"]),
        "n_uncertain_directed_edges": sum(map(len, proof["uncertain_links"].values())),
        "n_settlements": len(settlements), "n_destinations": len(destinations), "confirmed_closures": 0,
        "fixed_selected_destination_counts": dict(sorted(selection_counts.items())),
        "destination_rule": proof["destination_rule"], "destination_switching_after_closure": False,
        "certificate_checks": proof["certificate_checks"], "fixed_destination_certificate_checks": proof["fixed_destination_certificate_checks"],
        "proof": record(proof_path), "scenario_policy": record(out / "scenario-policy.json"),
        "interpretation": "Conditional graph bounds for the declared candidate-selected uncertainty set; unspecified failures excluded. Unknown radar and candidate absence do not establish passability."}
    summary = {"schema": "akash-fresh-exposure-v1", "status": "completed", "case": plan["case"],
        "water_source": record(water_manifest), "plan_source": plan_record, "source_dates": water["source_dates"],
        "historical_osm": historical_osm, "temporal_counts": counts,
        "grid": {"crs": grid[0].to_string(), "transform": list(grid[1]), "shape": [grid[3], grid[2]],
                 "pixel_area_m2": abs(grid[1].determinant), "resampling_used": False,
                 "polygonization": "Exact four-connected native pixel cells; no smoothing, simplification or geometry repair"},
        "road_association": {"half_width_m": 10, "full_width_m": 20, "end_caps": "flat", "join_style": "mitre",
                             "observed_road_width": False, "positive_overlap_area_tolerance_m2": EPSILON},
        **{name: summarize_features(collections[name]) for name in ("roads", "bridges", "buildings")},
        "candidates": candidates, "source_feature_audits": audits, "access_scenario": scenario_summary,
        "scope": "All mapped source roads and valid building footprints retained, including unknown/outside-AOI coverage; bridge-tagged physical segments are a subset of roads.",
        "reference_labels_loaded": False, "model_inference_run": False, "network_requests": 0, "gpu_used": False,
        "limitations": ["Water-state changes remain provisional; permanent and flood water were inseparable in the training labels.",
                        "No debris, confirmed flood, structure damage, observed closure or operational safety inference.",
                        "The 20 m association corridor is an explicit geometric choice, not a measured road width.",
                        "OSM footprints/facilities and road restrictions remain incomplete or unvalidated; source audits are retained.",
                        "Access robustness is conditional on baseline edges and this declared uncertainty set; unspecified/unknown failures are excluded.",
                        "Pixel-cell geometry preserves grid spacing; model context and unvalidated geolocation limit independent spatial detail."],
        "wall_seconds": time.perf_counter() - started}
    write_json(out / "summary.json", summary)
    for item in inputs:
        checked_record(item, Path(item["path"]).parent, "input during execution")
    outputs = {path.name: record(path) for path in sorted(out.iterdir()) if path.is_file()}
    code_paths = [Path(__file__), ROOT / "scripts/map_radar_exposure.py", ROOT / "scripts/prove_access_bounds.py",
                  *[ROOT / ("src/akash_flood/" + name) for name in ("access.py", "access_bounds.py", "nearest_access.py", "buildings.py", "osm.py", "snap_products.py")]]
    binding = {"schema": "akash-fresh-exposure-binding-v1", "status": "completed", "case": plan["case"],
               "created_at": datetime.now(timezone.utc).isoformat(), "summary": record(out / "summary.json"),
               "water_source": record(water_manifest), "plan_source": plan_record,
               "inputs": inputs, "outputs": outputs, "code": [record(path) for path in code_paths],
               "reference_labels_loaded": False, "model_inference_run": False, "network_requests": 0}
    write_json(out / "source-and-output-manifest.json", binding)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--water-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.water_manifest, args.plan, args.out)
    print(json.dumps({name: summary[name] for name in ("temporal_counts", "roads", "bridges", "buildings", "access_scenario", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
