"""Freeze then execute one reference-free native Sentinel-2 optical pair.

Preparation reads only saved catalogue/plan JSON. Execution is explicit, bounded,
credential-free, and never reuses a historical optical pair. Spectral thresholds
and the native-grid reader are reused unchanged from the original implementation.
The output is a provisional surface-change screen, never a complete debris map.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
from pyproj import Transformer
from rasterio.features import geometry_mask, shapes
from shapely.geometry import mapping, shape
from shapely.ops import transform

from akash_flood.catalog import utc, validate_bbox
from akash_flood.fresh_case import file_record, parse_discovery, plan_projection, reject_source_path
from akash_flood.optical_change import POLICY, select_optical_pair, screen
from akash_flood.optical_composite import AuditedCompositeRangeHTTP, source_reflectance_support, validate_asset_url
from build_fresh_report import write_png
from build_s2_debris_candidates import asset_grid, read_band, window_for, write_layer

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_SCHEMA = "akash-fresh-optical-adapter-20261012-v3"
CODE = (
    "scripts/run_fresh_optical_20261012_v2.py", "scripts/build_s2_debris_candidates.py",
    "scripts/build_fresh_report.py", "src/akash_flood/catalog.py",
    "src/akash_flood/fresh_case.py", "src/akash_flood/optical_change.py",
    "src/akash_flood/optical_composite.py", "src/akash_flood/optical_observation.py",
    "src/akash_flood/s1_crop.py", "src/akash_flood/water_inference.py",
)
BAND_FILENAMES = {"blue": "B02.tif", "red": "B04.tif", "nir": "B08.tif",
                  "swir16": "B11.tif", "scl": "SCL.tif"}
DEPENDENCIES = ("numpy", "scipy", "rasterio", "shapely", "pyproj", "requests", "affine")


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load(path):
    path = Path(path).resolve(strict=True)
    reject_source_path(path)
    if not path.is_file() or path.stat().st_size > 40_000_000:
        raise ValueError("Expected a bounded source JSON file, at most 40 MB")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field: " + key)
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Source JSON must be an object")
    return value


def verify(binding):
    path = Path(binding["path"]).resolve(strict=True)
    reject_source_path(path)
    if file_record(path) != binding:
        raise ValueError("Frozen file identity changed: " + path.name)
    return path


def strict_date(value):
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("Use the exact YYYY-MM-DD event date")
    return value


def bounded_int(value, upper, name):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
        raise ValueError(f"{name} must be an integer from 1 to {upper}")
    return value


def software_inventory():
    return {"python": sys.version, "packages": {name: importlib.metadata.version(name) for name in DEPENDENCIES}}


def covers_bbox(outer, inner):
    outer, inner = validate_bbox(outer), validate_bbox(inner)
    return outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]


def validate_catalog_window(document, bbox, event_date, max_days):
    """Verify any explicit saved query coverage; never infer it from cloud outcomes."""
    if "max_days" in document:
        saved_days = bounded_int(document["max_days"], 90, "saved catalogue max_days")
        if saved_days < max_days:
            raise ValueError("Saved catalogue date window is narrower than the frozen optical window")
    if "queried_context_bbox" in document and not covers_bbox(document["queried_context_bbox"], bbox):
        raise ValueError("Declared provider query context does not cover the optical AOI")
    records = document.get("search_records")
    if records is None:
        return "Explicit saved window checked when present; complete provider query bounds are not independently recorded"
    if not isinstance(records, list) or not records:
        raise ValueError("Saved search_records must contain the complete optical query")
    relevant = [r for r in records if isinstance(r, dict) and r.get("collection") == "sentinel-2-c1-l2a"]
    if len(relevant) != 1:
        raise ValueError("Require one unambiguous saved Collection-1 optical query")
    query = relevant[0].get("query", {})
    if relevant[0].get("search_complete") is not True or not covers_bbox(query.get("bbox", []), bbox):
        raise ValueError("Saved optical query is incomplete or does not cover the AOI")
    if "queried_context_bbox" in document and query["bbox"] != document["queried_context_bbox"]:
        raise ValueError("Saved provider query bbox contradicts declared catalogue query context")
    interval = query.get("datetime", "").split("/")
    if len(interval) != 2:
        raise ValueError("Explicit saved optical query datetime interval required")
    event = utc(event_date + "T00:00:00Z")
    if utc(interval[0]) > event - timedelta(days=max_days) or utc(interval[1]) < event + timedelta(days=max_days + 1):
        raise ValueError("Saved optical provider interval does not cover the frozen pre/post date window")
    return "Saved complete optical provider query covers the frozen AOI and pre/post date window; provider inventory independently unverified"


def validate_contract(document, bbox, event_date, plan=None, max_days=24):
    """Reject incomplete discovery and an explicitly different saved case."""
    bbox = validate_bbox(bbox)
    strict_date(event_date)
    _, incomplete = parse_discovery(document)
    if incomplete:
        raise ValueError("Unresolved catalogue pagination or incomplete saved search")
    catalog_bbox = validate_bbox(document["bbox"]) if "bbox" in document else None
    if catalog_bbox is not None and not covers_bbox(catalog_bbox, bbox):
        raise ValueError("Saved catalogue bbox does not cover the requested AOI")
    if "event_date" in document and document["event_date"] != event_date:
        raise ValueError("Saved catalogue event date differs from the requested case")
    if plan is not None:
        if not (plan.get("schema") == "akash-fresh-case-runner-v1"
                or plan.get("schema_version") == 1):
            raise ValueError("Require a current fresh-case plan contract")
        case = plan.get("case", {})
        if case.get("bbox") != bbox or case.get("event_date") != event_date:
            raise ValueError("Case plan bbox/event date differs from optical request")
        context = validate_bbox(case.get("context_bbox", bbox))
        if not covers_bbox(context, bbox):
            raise ValueError("Case plan context does not cover its AOI")
        if catalog_bbox is not None and catalog_bbox not in (bbox, context):
            raise ValueError("Saved catalogue bbox contradicts both the plan context and exact AOI query")
        if "queried_context_bbox" in document and document["queried_context_bbox"] not in (bbox, context):
            raise ValueError("Declared provider query context contradicts the supplied case plan")
        if plan.get("reference_labels_loaded") is True:
            raise ValueError("Reference labels are forbidden optical inputs")
    validate_catalog_window(document, bbox, event_date, max_days)
    return bbox


def validate_plan_projection(plan, grid):
    if plan is None:
        return
    projection = plan.get("projection") or plan_projection(plan["case"].get("context_bbox", plan["case"]["bbox"]))
    if projection.get("status") != "single_utm_zone" or projection.get("epsg") != grid[0].to_epsg():
        raise ValueError("Selected optical projection differs from the supported case-plan water projection")


def source_assessment(scene, scl, aoi):
    """Per-date observation support, with the existing fixed radiometry limits."""
    support = source_reflectance_support(scene, scl) & aoi
    denominator = scene["nir"] + scene["red"]
    bsi_denominator = sum(scene.values())
    return support & np.isfinite(denominator) & np.isfinite(bsi_denominator) & (denominator > 1e-6) & (bsi_denominator > 1e-6)


def validate_pair_assets(pair, bbox, max_cells):
    """Inspect saved STAC geometry/radiometry without opening pixels or URLs."""
    baseline = None
    for item in pair.values():
        window, grid, geometry = window_for(item, bbox)
        if grid[2] * grid[3] > max_cells:
            raise ValueError("Native optical crop exceeds frozen maximum cell count")
        if baseline is None:
            baseline = (window, grid, geometry)
        elif window != baseline[0] or grid != baseline[1]:
            raise ValueError("Selected optical date grids differ; resampling is forbidden")
        scl_grid = asset_grid(item, "scl")
        for key in POLICY["bands"]:
            crs, affine, _, _ = asset_grid(item, key)
            asset = item["assets"][key]
            validate_asset_url(asset["href"])
            if asset["href"].rsplit("/", 1)[-1] != BAND_FILENAMES[key]:
                raise ValueError("Asset filename does not identify the declared spectral band")
            if (crs != grid[0] or affine.c != scl_grid[1].c or affine.f != scl_grid[1].f
                    or isinstance(asset.get("file:size"), bool)
                    or not isinstance(asset.get("file:size"), int) or asset["file:size"] <= 0):
                raise ValueError("Invalid exact native-band grid or declared object length")
            band = asset["raster:bands"][0]
            if band.get("data_type") != ("uint8" if key == "scl" else "uint16"):
                raise ValueError("Native SCL/reflectance raster datatype differs")
            nodata = band.get("nodata")
            if isinstance(nodata, bool) or not isinstance(nodata, (int, float)) or not math.isfinite(nodata):
                raise ValueError("Explicit finite native nodata required")
            if key != "scl":
                scale, offset = band.get("scale"), band.get("offset")
                if (any(isinstance(v, bool) or not isinstance(v, (int, float))
                        or not math.isfinite(v) for v in (scale, offset)) or scale <= 0):
                    raise ValueError("Explicit finite positive reflectance scale and offset required")
    return baseline


def prepare(args):
    if args.out.exists():
        raise FileExistsError("Preparation requires a new folder; preserved runs cannot be overwritten")
    reject_source_path(args.out)
    bbox = validate_bbox(args.bbox)
    event_date = strict_date(args.event_date)
    max_days = bounded_int(args.max_days, 24, "max-days")
    limits = {"maximum_new_http_body_bytes": bounded_int(args.max_body_bytes, 100_000_000, "max-body-bytes"),
              "maximum_attempted_gets": bounded_int(args.max_attempts, 256, "max-attempts"),
              "maximum_acquisition_seconds": bounded_int(args.max_seconds, 600, "max-seconds"),
              "maximum_native20m_cells": bounded_int(args.max_cells, 4_000_000, "max-cells")}
    catalog_binding = file_record(args.catalog)
    document = load(args.catalog)
    plan_binding = file_record(args.plan) if args.plan else None
    plan = load(args.plan) if args.plan else None
    validate_contract(document, bbox, event_date, plan, max_days)
    pair = select_optical_pair(document, bbox, event_date, max_days=max_days)
    window, grid, _ = validate_pair_assets(pair, bbox, limits["maximum_native20m_cells"])
    verify(catalog_binding)
    if plan_binding is not None:
        verify(plan_binding)
    validate_plan_projection(plan, grid)
    args.out.mkdir(parents=True, exist_ok=False)
    items = {}
    for branch, item in pair.items():
        path = args.out / (branch + ".stac.json")
        dump(path, item)
        items[branch] = file_record(path)
    protocol = {
        "schema": "akash-s2-surface-change-protocol-v1", "adapter_schema": ADAPTER_SCHEMA,
        "recorded_before_pixel_acquisition_at": now(), "catalog": catalog_binding,
        "bbox": bbox, "event_date": event_date, "max_days": max_days, "policy": POLICY,
        "case_plan": plan_binding, "source_items": items,
        "code": [file_record(ROOT / name) for name in CODE], "limits": limits,
        "software": software_inventory(),
        "native_window": list(window.flatten()),
        "grid": {"crs": str(grid[0]), "transform": list(grid[1]),
                 "shape": [int(grid[3]), int(grid[2])]},
        "source_selection": POLICY["selector"], "crop_reuse": None,
        "saved_discovery_completeness": "No unresolved provider pagination in supplied saved search; provider inventory independently unverified",
        "saved_query_bbox": document.get("bbox"), "provider_query_context_bbox": document.get("queried_context_bbox"),
        "saved_query_coverage_scope": validate_catalog_window(document, bbox, event_date, max_days),
        "known_context": "Fresh adapter reuses previously developed engineering thresholds; no reference labels consumed and no blind accuracy claim",
        "reference_labels_loaded": False, "new_raw_pixels_read_before_freeze": False,
        "network_requests_before_freeze": 0, "original_gate_unchanged": True,
        "gpu_used": False, "maximum_new_http_body_bytes": limits["maximum_new_http_body_bytes"],
        "attribution": {"observations": "Copernicus Sentinel-2 / ESA", "public_catalogue_and_COGs": "Element84 Earth Search"},
    }
    path = args.out / "protocol.json"
    dump(path, protocol)
    receipt = {"status": "protocol_frozen; execution_not_run", "adapter_schema": ADAPTER_SCHEMA,
               "protocol": file_record(path), "selected": {k: v["id"] for k, v in pair.items()},
               "network_requests": 0, "pixels_read": False, "gpu_used": False}
    dump(args.out / "prepare-receipt.json", receipt)
    return receipt


def read_frozen(out):
    receipt = load(out / "prepare-receipt.json")
    protocol_path = verify(receipt["protocol"])
    protocol = load(protocol_path)
    if (protocol.get("adapter_schema") != ADAPTER_SCHEMA
            or protocol.get("schema") != "akash-s2-surface-change-protocol-v1"
            or protocol.get("policy") != POLICY or protocol.get("reference_labels_loaded") is not False):
        raise ValueError("Require this exact reference-free frozen optical protocol")
    for binding in (*protocol["code"], protocol["catalog"]):
        verify(binding)
    if protocol.get("software") != software_inventory():
        raise ValueError("Frozen optical execution environment changed")
    plan = load(verify(protocol["case_plan"])) if protocol["case_plan"] else None
    document = load(verify(protocol["catalog"]))
    validate_contract(document, protocol["bbox"], protocol["event_date"], plan, protocol["max_days"])
    pair = {branch: load(verify(binding)) for branch, binding in protocol["source_items"].items()}
    if pair != select_optical_pair(document, protocol["bbox"], protocol["event_date"], max_days=protocol["max_days"]):
        raise ValueError("Frozen optical pair differs from deterministic catalogue selection")
    window, grid, geometry = validate_pair_assets(pair, protocol["bbox"], protocol["limits"]["maximum_native20m_cells"])
    validate_plan_projection(plan, grid)
    actual_grid = {"crs": str(grid[0]), "transform": list(grid[1]), "shape": [int(grid[3]), int(grid[2])]}
    if actual_grid != protocol["grid"] or list(window.flatten()) != protocol["native_window"]:
        raise ValueError("Frozen native window/grid changed")
    return protocol, pair, window, grid, geometry


def execute(args, *, client_factory=AuditedCompositeRangeHTTP, band_reader=read_band):
    reject_source_path(args.out)
    if any((args.out / name).exists() for name in ("execution-started.json", "manifest.json", "failed.json", "transport.json")):
        raise FileExistsError("Execution already attempted; preserve it and prepare a fresh folder")
    protocol, pair, window, grid, geometry = read_frozen(args.out)
    if args.bbox is not None and validate_bbox(args.bbox) != protocol["bbox"]:
        raise ValueError("Execution bbox differs from frozen optical case")
    if args.event_date is not None and args.event_date != protocol["event_date"]:
        raise ValueError("Execution event date differs from frozen optical case")
    started, cpu_started = time.perf_counter(), time.process_time()
    dump(args.out / "execution-started.json", {"started_at": now(), "protocol": file_record(args.out / "protocol.json"), "automatic_retry": False})
    limits = protocol["limits"]
    client = client_factory([item["assets"][key]["href"] for item in pair.values() for key in POLICY["bands"]],
                            max_bytes=limits["maximum_new_http_body_bytes"], max_attempts=limits["maximum_attempted_gets"],
                            deadline=started + limits["maximum_acquisition_seconds"])
    scenes, scls, receipts = {}, {}, {}
    try:
        dims = (int(grid[3]), int(grid[2]))
        aoi = geometry_mask([mapping(geometry)], out_shape=dims, transform=grid[1], invert=True, all_touched=False)
        if not aoi.any():
            raise ValueError("Requested AOI contains no native20m pixel centers")
        # Observe both SCL sources first; source selection cannot change from observed coverage.
        for branch, item in pair.items():
            folder = args.out / branch
            folder.mkdir()
            scenes[branch], receipts[branch] = {}, {}
            scls[branch], receipt = band_reader(item, "scl", window, grid, client, folder)
            receipts[branch]["scl"] = receipt
            dump(folder / "scl-crop-receipt.json", receipt)
        raw_clear = {branch: int((aoi & np.isin(scl, (4, 5, 6))).sum()) for branch, scl in scls.items()}
        clear = {branch: {"raw_scl_clear_pixels": count, "fraction": count / int(aoi.sum()),
                         "original70percentgate_passes": count / int(aoi.sum()) >= .7}
                 for branch, count in raw_clear.items()}
        dump(args.out / "scl-coverage.json", {"aoi_pixels": int(aoi.sum()), "sources": clear,
                                               "source_selection_changed": False, "raw_scl_is_ground_truth": False})
        for branch, item in pair.items():
            for key in POLICY["bands"]:
                if key == "scl":
                    continue
                scenes[branch][key], receipt = band_reader(item, key, window, grid, client, args.out / branch)
                receipts[branch][key] = receipt
                dump(args.out / branch / (key + "-crop-receipt.json"), receipt)
        labels, assessed, indices, counts, _ = screen(scenes["before"], scenes["after"], scls["before"], scls["after"], aoi)
        source_assessed = {branch: source_assessment(scenes[branch], scls[branch], aoi) for branch in scenes}
        if not np.array_equal(assessed, source_assessed["before"] & source_assessed["after"]):
            raise ValueError("Per-date assessment masks disagree with joint fixed screen")
        for branch, valid in source_assessed.items():
            write_layer(args.out / (branch + "-assessment-mask.tif"), valid.astype("uint8"), grid, 0)
        write_layer(args.out / "surface-change-candidates.tif", labels, grid, 255)
        write_layer(args.out / "valid-mask.tif", assessed.astype("uint8"), grid, 0)
        write_layer(args.out / "aoi-mask.tif", aoi.astype("uint8"), grid, 0)
        for branch in indices:
            for key, values in indices[branch].items():
                write_layer(args.out / (branch + "-" + key + ".tif"), np.where(assessed, values, -9999).astype("float32"), grid, -9999)
        unproject = Transformer.from_crs(grid[0], 4326, always_xy=True).transform
        features = []
        for geo, value in shapes(labels, mask=labels == 1, transform=grid[1], connectivity=4):
            poly = shape(geo)
            if value != 1 or not poly.is_valid:
                raise ValueError("Candidate polygon fails native geometry validation")
            features.append({"type": "Feature", "geometry": mapping(transform(unproject, poly)), "properties": {
                "native_grid_area_m2": poly.area, "candidate_basis": POLICY["scope"], "confirmed_debris": False,
                "confirmed_landslide": False, "road_closure_assigned": False}})
        dump(args.out / "surface-change-candidates.geojson", {"type": "FeatureCollection", "features": features})
        rgb = np.full((*dims, 3), 255, dtype="uint8")
        rgb[aoi], rgb[assessed], rgb[labels == 1] = [193, 200, 207], [232, 240, 230], [147, 78, 35]
        write_png(args.out / "overview.png", rgb)
        dump(args.out / "transport.json", client.audit())
        manifest = {"schema": "akash-s2-surface-change-v1", "adapter_schema": ADAPTER_SCHEMA,
            "status": "completed", "completed_at": now(), "protocol": file_record(args.out / "protocol.json"),
            "bbox": protocol["bbox"], "event_date": protocol["event_date"], "grid": protocol["grid"],
            "source_dates": {branch: item["properties"]["datetime"] for branch, item in pair.items()},
            "source_item_ids": {branch: item["id"] for branch, item in pair.items()},
            "source_assessment_counts": {branch: {"aoi_pixels": counts["aoi_pixels"], "assessed_pixels": int(valid.sum()),
                                                    "unknown_pixels": counts["aoi_pixels"] - int(valid.sum())}
                                         for branch, valid in source_assessed.items()},
            "source_crops": receipts, "counts": counts, "raw_scl_single_scene_clear": clear,
            "joint_assessed_fraction": counts["joint_assessed_pixels"] / counts["aoi_pixels"],
            "candidate_area_m2": counts["candidate_pixels"] * 400,
            "scene_date_alignment_resampling_used": False, "10m_reflectance_aggregated_to20m": True,
            "partial_exploration_is_original_gate_pass": False, "original_single_scene_gate_changed": False,
            "transport": file_record(args.out / "transport.json"), "network_received_body_bytes": client.bytes_read,
            "network_requests": len(client.requests), "imported_crop_count": 0,
            "outputs": {path.name: file_record(path) for path in sorted(args.out.iterdir()) if path.suffix in (".tif", ".geojson", ".png")},
            "scope": POLICY["scope"], "local_debris_accuracy_established": False,
            "reference_labels_loaded": False, "gpu_used": False, "surface_change_thresholds_retuned": False,
            "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
            "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
            "rss_scope": "Maximum lifetime resident set of this execution process; Linux KiB converted to bytes, macOS already bytes",
            "limitations": [
                "Untuned spectral engineering hypotheses; no local debris accuracy or operational damage claim.",
                "Vegetation loss/exposed surface may be agriculture, clearing, moisture, season, shadow or sensor effects.",
                "Cloud, shadow, snow, water, invalid pixels and their 60m neighborhoods stay unknown; no cloud reconstruction.",
                "The selected pair never changes from coverage or candidate outcomes; a cloudy earliest post scene can yield little supported area.",
                "Complete TIFF metadata agreement is not absolute geolocation or inter-date registration validation.",
                "Observed single-scene raw SCL fractions do not establish a complete debris map or a benchmark accuracy score.",
                "Range checksums bind received ranges and ETag/length consistency; provider whole-object checksums are not asserted verified."]}
        dump(args.out / "manifest.json", manifest)
        return {"status": "completed", "counts": counts, "raw_scl_single_scene_clear": clear,
                "network_received_body_bytes": client.bytes_read, "manifest": file_record(args.out / "manifest.json")}
    except Exception as error:
        if not (args.out / "transport.json").exists():
            dump(args.out / "transport.json", client.audit())
        dump(args.out / "failed.json", {"status": "failed", "failed_at": now(),
            "error_type": type(error).__name__, "message": str(error)[:2000],
            "protocol": file_record(args.out / "protocol.json"), "transport": file_record(args.out / "transport.json"),
            "received_body_bytes": client.bytes_read, "attempted_get_count": len(client.requests),
            "wall_seconds": time.perf_counter() - started, "no_automatic_retry": True,
            "missing_observations_are_safe_or_negative": False, "completed_optical_map": False})
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    p.add_argument("--catalog", type=Path)
    p.add_argument("--plan", type=Path, help="Optional exact case plan.json or execution-plan.json binding")
    p.add_argument("--bbox", type=float, nargs=4)
    p.add_argument("--event-date")
    p.add_argument("--max-days", type=int, default=24)
    p.add_argument("--max-body-bytes", "--budget-bytes", dest="max_body_bytes", type=int, default=60_000_000)
    p.add_argument("--max-attempts", type=int, default=256)
    p.add_argument("--max-seconds", "--timeout-seconds", dest="max_seconds", type=int, default=180)
    p.add_argument("--max-cells", type=int, default=1_000_000)
    p.add_argument("--out", type=Path, required=True)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.prepare_only and (args.catalog is None or args.bbox is None or args.event_date is None):
        p.error("Preparation requires --catalog, --bbox and --event-date")
    if args.execute and (args.catalog is not None or args.plan is not None):
        p.error("Execution consumes its frozen catalogue/plan; supply only --out and optional matching bbox/date")
    result = prepare(args) if args.prepare_only else execute(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
