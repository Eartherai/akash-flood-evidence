"""Intersect saved radar candidates with historical mapped features, on CPU.

Candidate overlap is not confirmed flood/debris, damage or road closure. Masked
radar remains unknown. This script neither reads reference labels nor changes a
road graph; its geometry associations must not be used as automatic closures.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.features import shapes
from shapely.geometry import GeometryCollection, LineString, mapping, shape
from shapely.ops import transform, unary_union
from shapely.prepared import prep

from akash_flood.access import build_road_graph
from akash_flood.buildings import building_footprints


ROOT = Path(__file__).resolve().parents[1]
RASTER_FILES = {
    "aoi": "aoi-mask.tif", "assessed": "valid-mask.tif",
    "change": "change-candidates.tif", "water": "water-candidates.tif",
}
BASES = {
    "change": "intersection with unvalidated same-sign dual-polarization change candidate",
    "water": "intersection with unvalidated new-darkening/water-screen candidate",
}
EPSILON = 1e-6


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def feature_collection(features):
    return {"type": "FeatureCollection", "features": features}


def read_evidence(folder):
    """Read only four named rasters; never resample or infer valid zeroes."""
    folder = Path(folder)
    if any(word in folder.as_posix().casefold() for word in ("emsr", "evaluation-only", "reference-map")):
        raise ValueError("Official evaluation products must not be input evidence")
    arrays, grid = {}, None
    for key, filename in RASTER_FILES.items():
        with rasterio.open(folder / filename) as dataset:
            if dataset.count != 1 or dataset.crs is None:
                raise ValueError("Evidence must have one band and an explicit CRS")
            crs = CRS.from_user_input(dataset.crs)
            if not crs.is_projected or any(axis.unit_name != "metre" for axis in crs.axis_info):
                raise ValueError("Exact metric intersections require a projected CRS with metre units")
            signature = (crs, dataset.transform, dataset.width, dataset.height)
            if grid is not None and signature != grid:
                raise ValueError("Evidence rasters must share exactly the same grid; no resampling allowed")
            grid = signature
            if not all(math.isfinite(value) for value in dataset.transform[:6]) or abs(dataset.transform.determinant) <= 0:
                raise ValueError("Evidence affine must be finite and invertible")
            array = dataset.read(1)
            allowed = (0, 1) if key in ("aoi", "assessed") else (0, 1, 255)
            if not np.all(np.isin(array, allowed)):
                raise ValueError(f"Unexpected {key} mask encoding")
            arrays[key] = array
    aoi, valid = arrays["aoi"] == 1, arrays["assessed"] == 1
    if not aoi.any():
        raise ValueError("AOI has no pixels")
    if np.any(valid & ~aoi):
        raise ValueError("Assessed pixels must be within the AOI")
    for key in ("change", "water"):
        if np.any((arrays[key] == 1) & ~valid):
            raise ValueError("Candidate pixels must have assessed radar support")
        if np.any(valid & (arrays[key] == 255)):
            raise ValueError("Assessed mask disagrees with candidate nodata")
    return arrays, grid


def pixel_polygons(mask, affine):
    """Four-connected exact pixel-cell boundaries, with holes and no smoothing."""
    mask = np.asarray(mask, dtype=bool)
    result = []
    for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=affine, connectivity=4):
        polygon = shape(geometry)
        if value != 1 or polygon.is_empty or not polygon.is_valid:
            raise ValueError("Invalid mask polygon; no automatic geometry repair")
        result.append(polygon)
    return result


def mask_geometries(arrays, grid):
    affine = grid[1]
    masks = {
        "aoi": arrays["aoi"] == 1,
        "assessed": arrays["assessed"] == 1,
        "change": arrays["change"] == 1,
        "water": arrays["water"] == 1,
    }
    polygons = {key: pixel_polygons(mask, affine) for key, mask in masks.items()}
    unions = {key: unary_union(value) if value else GeometryCollection() for key, value in polygons.items()}
    return masks, polygons, unions


class Coverage:
    """Exact planar intersections with native raster-cell unions, in metres."""

    def __init__(self, geometries):
        self.geometries = geometries
        self.prepared = {key: prep(value) for key, value in geometries.items()}

    def within(self, geometry, key, measure):
        region, prepared = self.geometries[key], self.prepared[key]
        if region.is_empty or not prepared.intersects(geometry):
            return 0.0
        if prepared.covers(geometry):
            return getattr(geometry, measure)
        return getattr(geometry.intersection(region), measure)

    def describe(self, geometry, *, measure="area"):
        if measure not in ("area", "length"):
            raise ValueError("Coverage measure must be area or length")
        total = getattr(geometry, measure)
        if not math.isfinite(total) or total <= EPSILON:
            raise ValueError("Feature has no measurable geometry")
        aoi = min(total, self.within(geometry, "aoi", measure))
        assessed = min(aoi, self.within(geometry, "assessed", measure))
        change = min(assessed, self.within(geometry, "change", measure))
        water = min(assessed, self.within(geometry, "water", measure))
        values = {
            "total": total, "within_aoi": aoi, "outside_aoi": max(0.0, total - aoi),
            "assessed": assessed, "unassessed": max(0.0, aoi - assessed),
            "change_candidate": change, "water_candidate": water,
            "assessed_no_change_candidate": max(0.0, assessed - change),
            "assessed_no_water_candidate": max(0.0, assessed - water),
        }
        status = ("outside_aoi" if aoi <= EPSILON else
                  "fully_assessed" if values["outside_aoi"] <= EPSILON and values["unassessed"] <= EPSILON else
                  "within_aoi_fully_unassessed" if assessed <= EPSILON and values["outside_aoi"] <= EPSILON else
                  "partial_or_unknown_coverage")
        return {
            "measure": "area_m2" if measure == "area" else "length_m",
            "status": status,
            **{key: round(value, 6) for key, value in values.items()},
            "fractions_of_total": {key: value / total for key, value in values.items() if key != "total"},
            "fully_assessed_no_change_candidate": status == "fully_assessed" and change <= EPSILON,
            "fully_assessed_no_water_candidate": status == "fully_assessed" and water <= EPSILON,
            "interpretation": "No candidate in assessed coverage is not verified absence of hazard; masked and outside-AOI coverage remain unknown.",
        }


def physical_segments(graph):
    """Group only the same source way/endpoints; retain all directed identities."""
    groups = defaultdict(list)
    for start, end, key, data in graph.edges(keys=True, data=True):
        groups[(data["way_id"], min(start, end), max(start, end))].append((start, end, data))
    features, excluded = [], []
    for (way, left, right), edges in sorted(groups.items()):
        identity = f"way-{way}-nodes-{left}-{right}"
        geometry = LineString([(graph.nodes[node]["x"], graph.nodes[node]["y"]) for node in (left, right)])
        if geometry.is_empty or geometry.length == 0:
            excluded.append({"physical_segment_id": identity, "reason": "zero-length source geometry"})
            continue
        data = edges[0][2]
        features.append({"type": "Feature", "id": identity, "geometry": mapping(geometry), "properties": {
            "physical_segment_id": identity, "source_way_id": way, "source_endpoint_node_ids": [left, right],
            "source_directed_edges": [{"edge_id": item[2]["edge_id"], "from_node_id": item[0], "to_node_id": item[1]}
                                      for item in sorted(edges, key=lambda item: item[2]["edge_id"])],
            "source_directed_edge_count": len(edges), "bridge": bool(data["bridge"]),
            "tunnel": bool(data["tunnel"]), "highway": data["highway"], "layer": data["layer"],
            "source_tags": data["tags"], "confirmed_damage": False, "road_closure_assigned": False,
        }})
    audit = {"source_directed_edges": graph.number_of_edges(), "mapped_physical_segments": len(features),
             "deduplication_key": "source way ID and unordered endpoint node IDs; parallel source ways remain distinct",
             "missing_source_node_ids": graph.graph["missing_node_ids"],
             "unresolved_oneway_way_ids": graph.graph["unresolved_oneway_way_ids"], "excluded": excluded,
             "network_filter": "existing build_road_graph drive filter; explicit no/private motor access excluded"}
    return feature_collection(features), audit


def overlap_features(osm, grid, geometries, *, road_buffer_m=10):
    if road_buffer_m not in (0, 10):
        raise ValueError("Road buffer must be 0 m (centerline) or explicit 10 m half-width")
    to_metric = Transformer.from_crs(4326, grid[0], always_xy=True).transform
    to_wgs84 = Transformer.from_crs(grid[0], 4326, always_xy=True).transform
    coverage = Coverage(geometries)
    roads, road_audit = physical_segments(build_road_graph(osm))
    corridors = []
    for feature in roads["features"]:
        line = transform(to_metric, shape(feature["geometry"]))
        properties = feature["properties"]
        properties["centerline_coverage"] = coverage.describe(line, measure="length")
        if road_buffer_m:
            corridor = line.buffer(road_buffer_m, cap_style="flat", join_style="mitre")
            properties["coverage"] = coverage.describe(corridor)
            corridors.append({"type": "Feature", "id": feature["id"],
                              "geometry": mapping(transform(to_wgs84, corridor)),
                              "properties": {"physical_segment_id": feature["id"], "buffer_half_width_m": road_buffer_m,
                                             "buffer_end_caps": "flat", "coverage": properties["coverage"],
                                             "observed_road_width": False, "road_closure_assigned": False}})
        else:
            properties["coverage"] = properties["centerline_coverage"]
        properties.update(road_buffer_half_width_m=road_buffer_m, candidate_overlap_basis=BASES,
                          geometry_association_only=True, verified_no_hazard=False)
    buildings, building_audit = building_footprints(osm)
    for feature in buildings["features"]:
        polygon = transform(to_metric, shape(feature["geometry"]))
        feature["properties"].update(coverage=coverage.describe(polygon), candidate_overlap_basis=BASES,
                                      geometry_association_only=True, verified_no_hazard=False)
    bridges = feature_collection([feature for feature in roads["features"] if feature["properties"]["bridge"]])
    return {"roads": roads, "road-corridors": feature_collection(corridors), "bridges": bridges,
            "buildings": buildings}, {"roads": road_audit, "buildings": building_audit}


def summarize_features(collection):
    coverages = [feature["properties"]["coverage"] for feature in collection["features"]]
    return {
        "mapped_feature_count": len(coverages),
        "features_outside_aoi": sum(c["within_aoi"] <= EPSILON for c in coverages),
        "features_partly_outside_aoi": sum(c["outside_aoi"] > EPSILON and c["within_aoi"] > EPSILON for c in coverages),
        "features_with_unassessed_radar": sum(c["unassessed"] > EPSILON for c in coverages),
        "features_with_assessed_radar": sum(c["assessed"] > EPSILON for c in coverages),
        "features_fully_assessed": sum(c["status"] == "fully_assessed" for c in coverages),
        "features_with_change_candidate_overlap": sum(c["change_candidate"] > EPSILON for c in coverages),
        "features_with_water_candidate_overlap": sum(c["water_candidate"] > EPSILON for c in coverages),
        "features_fully_assessed_no_change_candidate": sum(c["fully_assessed_no_change_candidate"] for c in coverages),
        "features_fully_assessed_no_water_candidate": sum(c["fully_assessed_no_water_candidate"] for c in coverages),
        "features_no_change_candidate_in_assessed_portion": sum(c["assessed"] > EPSILON and c["change_candidate"] <= EPSILON for c in coverages),
        "features_no_water_candidate_in_assessed_portion": sum(c["assessed"] > EPSILON and c["water_candidate"] <= EPSILON for c in coverages),
        "coverage_measure": coverages[0]["measure"] if coverages else None,
        "sum_feature_coverage": {key: round(sum(c[key] for c in coverages), 6) for key in
                                 ("total", "within_aoi", "outside_aoi", "assessed", "unassessed", "change_candidate",
                                  "water_candidate", "assessed_no_change_candidate", "assessed_no_water_candidate")},
        "sum_interpretation": "Per-feature intersections; overlapping mapped features/corridors may count the same pixels more than once. Not a union area or number of damaged structures.",
    }


def candidate_features(polygons, masks, grid):
    to_wgs84 = Transformer.from_crs(grid[0], 4326, always_xy=True).transform
    pixel_area = abs(grid[1].determinant)
    collections, summary = {}, {}
    for kind in ("change", "water"):
        features = []
        for index, polygon in enumerate(polygons[kind]):
            features.append({"type": "Feature", "id": f"{kind}-candidate-{index}",
                             "geometry": mapping(transform(to_wgs84, polygon)), "properties": {
                                 "candidate_kind": kind, "candidate_basis": BASES[kind],
                                 "native_grid_area_m2": polygon.area, "confirmed_hazard": False,
                                 "confirmed_damage": False, "road_closure_assigned": False}})
        count = int(masks[kind].sum())
        native_area = sum(polygon.area for polygon in polygons[kind])
        if not math.isclose(native_area, count * pixel_area, abs_tol=1e-5, rel_tol=1e-10):
            raise ValueError("Candidate polygons do not preserve native pixel-cell area")
        collections[kind + "-candidates"] = feature_collection(features)
        summary[kind] = {"candidate_pixel_count": count, "polygon_component_count": len(features),
                         "native_grid_pixel_area_m2": pixel_area, "candidate_area_m2": native_area,
                         "confirmed_hazard": False, "candidate_basis": BASES[kind]}
    return collections, summary


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run(folder, osm_path, out, *, road_buffer_m=10):
    folder, osm_path, out = Path(folder), Path(osm_path), Path(out)
    if out.exists():
        raise FileExistsError("Use a fresh output directory; preserve earlier evidence")
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "run.started.json", {"created_at": datetime.now(timezone.utc).isoformat(),
                                         "evidence_folder": str(folder), "osm_path": str(osm_path),
                                         "road_buffer_half_width_m": road_buffer_m})
    inputs = [folder / filename for filename in RASTER_FILES.values()] + [osm_path]
    for optional in (folder / "manifest.json", osm_path.with_suffix(".provenance.json")):
        if optional.exists():
            inputs.append(optional)
    hashes = {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in inputs}
    arrays, grid = read_evidence(folder)
    masks, polygons, geometries = mask_geometries(arrays, grid)
    osm = json.loads(osm_path.read_text())
    collections, audits = overlap_features(osm, grid, geometries, road_buffer_m=road_buffer_m)
    candidate_collections, candidate_summary = candidate_features(polygons, masks, grid)
    collections.update(candidate_collections)
    for filename, collection in collections.items():
        write_json(out / (filename + ".geojson"), collection)
    aoi, valid = masks["aoi"], masks["assessed"]
    summary = {
        "status": "mapped-feature intersections with unvalidated radar candidates; not damage or passability",
        "scope": "All mapped source features are retained, including outside-AOI and unassessed coverage. No closures are assigned.",
        "grid": {"crs": grid[0].to_string(), "affine": list(grid[1]), "width": grid[2], "height": grid[3],
                 "native_pixel_area_m2": abs(grid[1].determinant), "resampling": False,
                 "polygonization": "four-connected native pixel-cell boundaries; no smoothing, simplification or geometry repair"},
        "coverage_pixels": {"aoi": int(aoi.sum()), "assessed": int(valid.sum()), "unassessed": int((aoi & ~valid).sum())},
        "candidate_polygons": candidate_summary,
        "roads": summarize_features(collections["roads"]), "bridges": summarize_features(collections["bridges"]),
        "buildings": summarize_features(collections["buildings"]), "source_feature_audits": audits,
        "road_buffer": {"half_width_m": road_buffer_m, "total_width_m": road_buffer_m * 2,
                        "end_caps": "flat" if road_buffer_m else None, "joins": "mitre" if road_buffer_m else None,
                        "meaning": "An explicit geometric association corridor, not observed road width or calibrated positional uncertainty; centerline lengths are recorded separately."},
        "reference_labels_loaded": False, "road_closures_assigned": False, "gpu_used": False,
        "source_eligibility": "Not inferred by this mapper; review the recorded source manifest and rules independently.",
        "limitations": [
            "Candidate overlap is not confirmed flood, debris, physical structure damage or road closure.",
            "No candidate in assessed coverage is not verified absence of hazard; outside-AOI and masked coverage remain unknown.",
            "OSM polygon features are not verified physical building counts; missing/incomplete footprints are recorded by the existing audit.",
            "Bridge-tagged segments are a separate subset of roads; change and water overlaps are never collapsed into a bridge-damage label.",
            "Grid-cell geometry preserves raster spacing; averaging and geolocation uncertainty still limit independent spatial detail.",
            "No road graph is mutated and no access or routing conclusions are produced.",
        ],
        "wall_seconds": time.perf_counter() - start,
    }
    write_json(out / "summary.json", summary)
    # Detect source changes during the analysis rather than publish mixed provenance.
    if any(sha256(path) != hashes[str(path)]["sha256"] for path in inputs):
        raise ValueError("An input changed during analysis; this output is incomplete")
    outputs = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file()}
    manifest = {"status": "complete", "created_at": datetime.now(timezone.utc).isoformat(), "inputs": hashes,
                "code": {str(path): sha256(path) for path in (Path(__file__), ROOT / "src/akash_flood/access.py", ROOT / "src/akash_flood/buildings.py")},
                "outputs": outputs, "manifest_self_hash": "not included; compute externally", "reference_labels_loaded": False}
    write_json(out / "source-and-output-manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True, help="Folder containing four same-grid evidence masks")
    parser.add_argument("--osm", type=Path, default=ROOT / "data/rasuwa-pilot/osm.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--road-buffer-m", type=int, choices=(0, 10), default=10,
                        help="Corridor half-width: 10 m default; 0 uses exact centerline lengths")
    args = parser.parse_args()
    summary = run(args.evidence, args.osm, args.out, road_buffer_m=args.road_buffer_m)
    print(json.dumps({key: summary[key] for key in ("status", "coverage_pixels", "roads", "bridges", "buildings", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
