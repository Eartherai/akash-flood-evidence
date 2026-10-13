"""Prepared-raster prototype. Raw Sentinel preprocessing is a separate stage."""
from dataclasses import fields
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform, unary_union

from .access import analyze_access, build_road_graph
from .buildings import building_footprints, intersect_buildings
from .catalog import utc
from .flood import FloodConfig, classify_change
from .raster import candidate_geojson, export_labels, load_aligned, metric_pixel_area


def checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contract(config):
    event = date.fromisoformat(config["event_date"])
    before, after = config["before"], config["after"]
    if not (utc(before["datetime"]).date() < event < utc(after["datetime"]).date()):
        raise ValueError("Before/after observations must strictly bracket the event day")
    for key in ("relative_orbit", "orbit_state", "polarization", "backscatter"):
        if not before.get(key) or before[key] != after.get(key):
            raise ValueError(f"Input metadata must contain matching {key}")
    for source in (before, after):
        if (source.get("units") != "dB" or source.get("calibrated") is not True
                or source.get("terrain_corrected") is not True):
            raise ValueError("Inputs must be calibrated, terrain-corrected dB backscatter; raw amplitude is unsupported")
        if source.get("backscatter") not in {"sigma0", "gamma0"}:
            raise ValueError("Backscatter must be sigma0 or gamma0")
        if not source.get("scene_id") or not source.get("processing_record"):
            raise ValueError("A source scene ID and preprocessing record are required")
    if config.get("osm") and date.fromisoformat(config["osm_snapshot"]) >= event:
        raise ValueError("OSM snapshot must predate event")


def intersect_roads(graph, hazard_geojson, metric_crs):
    """Find intersecting physical segments; flag bridges for separate review."""
    hazard = unary_union([shape(feature["geometry"]) for feature in hazard_geojson["features"]])
    project = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True).transform
    seen, features, closures = set(), [], set()
    projected_hazard = transform(project, hazard)
    for u, v, key, edge in graph.edges(keys=True, data=True):
        physical_id = (edge["way_id"], min(u, v), max(u, v))
        if physical_id in seen:
            continue
        seen.add(physical_id)
        if hazard.is_empty or not edge["geometry"].intersects(hazard):
            continue
        overlap_m = transform(project, edge["geometry"]).intersection(projected_hazard).length
        if overlap_m <= 0:
            continue
        # A water intersection is only a closure *scenario*. Bridge/tunnel
        # intersections are never automatically treated as closures.
        scenario = not edge["bridge"] and not edge["tunnel"]
        if scenario:
            for a, b in ((u, v), (v, u)):
                for other_key, other in graph.get_edge_data(a, b, default={}).items():
                    if other["way_id"] == edge["way_id"]:
                        closures.add(other["edge_id"])
        features.append({"type": "Feature", "id": f"road-{edge['way_id']}-{u}-{v}",
                         "geometry": mapping(edge["geometry"]),
                         "properties": {"way_id": edge["way_id"], "overlap_m": round(overlap_m, 2),
                                        "bridge": edge["bridge"], "tunnel": edge["tunnel"],
                                        "scenario_closed": scenario, "confirmed_damage": False}})
    return {"type": "FeatureCollection", "features": features}, closures


def run(config_path, out_dir):
    started = time.perf_counter()
    config_path, out = Path(config_path).resolve(), Path(out_dir)
    config = json.loads(config_path.read_text())
    validate_contract(config)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be empty; preserve completed runs")
    out.mkdir(parents=True, exist_ok=True)
    def local(value):
        return (config_path.parent / value).resolve()
    names = ["before", "after"]
    paths = [local(config[name]["path"]) for name in names]
    for name in ("slope", "radar_invalid"):
        if config.get(name):
            names.append(name)
            paths.append(local(config[name]))
    arrays, valid, profile = load_aligned(paths)
    source = dict(zip(names, arrays))
    pixel_area = metric_pixel_area(profile)
    params = config.get("thresholds", {})
    allowed = {f.name for f in fields(FloodConfig)}
    if set(params) - allowed:
        raise ValueError("Unknown flood threshold settings")
    labels, metrics = classify_change(source["before"], source["after"], valid=valid,
                                     slope_deg=source.get("slope"),
                                     radar_invalid=source.get("radar_invalid"), config=FloodConfig(**params))
    export_labels(out / "hazard_candidates.tif", labels, profile)
    hazard = candidate_geojson(labels, profile)
    metrics["candidate_area_km2"] = metrics["new_water_candidate_pixels"] * pixel_area / 1e6
    metrics["assessed_area_km2"] = metrics["assessed_pixels"] * pixel_area / 1e6
    metrics["pixel_area_m2"] = pixel_area
    roads = {"type": "FeatureCollection", "features": []}
    buildings = {"type": "FeatureCollection", "features": []}
    building_audit = {"status": "not assessed; no OSM supplied"}
    access, settlements, destinations = [], config.get("settlements", []), config.get("destinations", [])
    if config.get("osm"):
        osm_path = local(config["osm"])
        osm = json.loads(osm_path.read_text())
        graph = build_road_graph(osm)
        footprints, building_audit = building_footprints(osm)
        buildings = intersect_buildings(footprints, hazard, profile["crs"])
        metrics["building_features_exposed"] = len(buildings["features"])
        metrics["mapped_building_features"] = building_audit["mapped_building_features"]
        metrics["building_features_excluded"] = len(building_audit["excluded"])
        roads, closed_edges = intersect_roads(graph, hazard, profile["crs"])
        access = analyze_access(graph, settlements, destinations, set(),
                                max_snap_m=config.get("max_snap_m", 500), closed_edge_ids=closed_edges)
        metrics["road_segments_exposed"] = len(roads["features"])
        metrics["newly_disconnected_under_scenario"] = sum(row["status"] == "newly_disconnected" for row in access)
        metrics["graph_missing_nodes"] = graph.graph["missing_node_ids"]
        paths.append(osm_path)
    metrics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    inputs = [{"path": str(path), "sha256": checksum(path)} for path in paths]
    manifest = {"run_id": hashlib.sha256(json.dumps({"config": config, "inputs": inputs}, sort_keys=True).encode()).hexdigest()[:16],
                "created_at": datetime.now(timezone.utc).isoformat(), "config": config, "inputs": inputs,
                "data_kind": config.get("data_kind", "user_supplied_preprocessed"),
                "stage": "prepared-raster CPU baseline", "gpu_used": False,
                "limitations": ["Water-change candidates are not verified flood or damage.",
                                "Debris detection and the AI copilot are not implemented in this prototype.",
                                "Road closures are an explicit intersection scenario, not observed passability.",
                                "Building exposure counts mapped polygon features, not damage or verified physical structures.",
                                "Missing roads, map boundaries, barriers and unmodeled turn restrictions affect access.",
                                "OSM snapshot and raster processing declarations require source-provenance review."]}
    outputs = {"run_manifest.json": manifest, "metrics.json": metrics, "hazards.geojson": hazard,
               "exposed_roads.geojson": roads, "settlement_access.json": access,
               "exposed_buildings.geojson": buildings, "building_geometry_audit.json": building_audit}
    for name, obj in outputs.items():
        (out / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    from .report import write_report
    write_report(out, manifest, metrics, hazard, roads, access, settlements, destinations, buildings=buildings)
    return metrics
