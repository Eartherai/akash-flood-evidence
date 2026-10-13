"""Freeze and execute a bounded CPU-only, SCL-selected optical composite.

Run once with --prepare-only to freeze metadata and cache hashes. Then execute
that exact protocol with --execute; source selection precedes new reflectance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.features import geometry_mask, shapes
from shapely.geometry import mapping, shape
from shapely.ops import transform

from akash_flood.catalog import utc, validate_bbox
from akash_flood.fresh_case import reject_source_path
from akash_flood.optical_change import reflectance, average_2x2
from akash_flood.optical_composite import (POLICY, BANDS, AuditedCompositeRangeHTTP,
    select_observation_sources, select_source_index, install_selected_reflectance, screen_composite,
    validate_asset_url)
from build_fresh_report import record, write_png
from build_s2_debris_candidates import asset_grid, read_band, window_for, write_layer

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify(binding):
    path = Path(binding["path"]).resolve()
    reject_source_path(path)
    if record(path) != binding:
        raise ValueError("Frozen file changed: " + path.name)
    return path


def validate_cached_metadata(path, item, key, window, grid):
    crs, affine, _, _ = asset_grid(item, key)
    factor = 1 if key in ("scl", "swir16") else 2
    expected = affine * rasterio.Affine.translation(window.col_off * factor, window.row_off * factor)
    band = item["assets"][key]["raster:bands"][0]
    with rasterio.open(path) as ds:
        if (ds.count != 1 or ds.crs != crs or ds.transform != expected
                or ds.width != grid[2] * factor or ds.height != grid[3] * factor
                or ds.nodata != band["nodata"] or ds.dtypes[0] != ("uint8" if key == "scl" else "uint16")):
            raise ValueError("Archived crop metadata differs from native source")


def freeze(args):
    if args.out.exists():
        raise FileExistsError("Use a new protocol folder; preserved runs cannot be overwritten")
    bbox = validate_bbox(args.bbox)
    reject_source_path(args.catalog)
    document = json.loads(args.catalog.read_text())
    sources, excluded = select_observation_sources(document, bbox, args.event_date)
    ordered = [(branch, item) for branch in ("before", "after") for item in sources[branch]]
    window, grid, _ = window_for(ordered[0][1], bbox)
    for _, item in ordered:
        other_window, other_grid, _ = window_for(item, bbox)
        if window != other_window or grid != other_grid:
            raise ValueError("Exact native source grids differ")
        for key in (*BANDS, "scl"):
            asset_grid(item, key)
            asset = item["assets"][key]
            validate_asset_url(asset["href"])
            band = asset["raster:bands"][0]
            if key != "scl" and (not all(k in band for k in ("scale", "offset", "nodata"))
                                  or band.get("data_type") != "uint16"):
                raise ValueError("Explicit reflectance type, scaling and nodata required")
    cache = {}
    cached_manifest = json.loads((args.reuse_pair / "manifest.json").read_text())
    if (cached_manifest.get("schema") != "akash-s2-surface-change-v1"
            or cached_manifest.get("status") != "completed" or cached_manifest["bbox"] != bbox
            or cached_manifest["event_date"] != args.event_date
            or cached_manifest.get("reference_labels_loaded") is not False):
        raise ValueError("Require the completed same-case archived optical pair")
    prior_protocol = json.loads(verify(cached_manifest["protocol"]).read_text())
    verify(cached_manifest["transport"])
    chosen = {item["id"]: item for _, item in ordered}
    for branch in ("before", "after"):
        item_binding = prior_protocol["source_items"][branch]
        item = json.loads(verify(item_binding).read_text())
        if chosen.get(item["id"]) != item:
            raise ValueError("Archived pair source differs from frozen catalogue source")
        cache[item["id"]] = {}
        for key in (*BANDS, "scl"):
            receipt = cached_manifest["source_crops"][branch][key]
            receipt_path = args.reuse_pair / branch / (key + "-crop-receipt.json")
            if json.loads(receipt_path.read_text()) != receipt or receipt["asset"] != item["assets"][key]:
                raise ValueError("Archived crop receipt differs")
            path = verify(receipt["crop"])
            validate_cached_metadata(path, item, key, window, grid)
            cache[item["id"]][key] = {"crop": receipt["crop"], "receipt": record(receipt_path),
                "source_item": item_binding, "asset": receipt["asset"],
                "remote_identity": receipt["remote_identity"],
                "whole_object_checksum_verified": False,
                "legacy_provenance": "Original completed pair receipts; some crops came from a failed acquisition with later captured crop hashes; successful-range receipts omit legacy failed attempts"}
    args.out.mkdir(parents=True)
    frozen = []
    for index, (branch, item) in enumerate(ordered, 1):
        path = args.out / f"source-{index:02d}.stac.json"
        dump(path, item)
        frozen.append({"source_index": index, "branch": branch, "item_id": item["id"],
                       "datetime": item["properties"]["datetime"], "source_item": record(path)})
    protocol = {"schema": "akash-optical-composite-protocol-v1", "frozen_at": datetime.now(timezone.utc).isoformat(),
        "policy": POLICY, "catalog": record(args.catalog), "bbox": bbox, "event_date": args.event_date,
        "ordered_sources": frozen, "excluded_item_ids": excluded, "cached_crops": cache,
        "cached_manifest": record(args.reuse_pair / "manifest.json"),
        "grid": {"crs": str(grid[0]), "transform": list(grid[1]), "shape": [int(grid[3]), int(grid[2])]},
        "native_window": list(window.flatten()),
        "code": [record(ROOT / name) for name in ("src/akash_flood/optical_composite.py", "scripts/run_optical_composite.py",
            "src/akash_flood/optical_change.py", "src/akash_flood/optical_observation.py", "src/akash_flood/s1_crop.py",
            "scripts/build_s2_debris_candidates.py", "scripts/build_fresh_report.py")],
        "prior_coverage_metadata": {"existing_assessed_cells": 79555, "aoi_cells": 706991,
            "eleven_source_raw_clear_union_upper_bound_fraction": .5936115169782925,
            "pre_20260812_raw_clear_fraction": .9879376116527651},
        "known_context": "Designed after earlier optical coverage and checking outcomes were known; no blind validation claim, no checking/reference pixels loaded",
        "new_raw_pixels_read_before_freeze": False, "reference_labels_loaded": False, "gpu_used": False,
        "decision": "Improve actual observation support with fixed SCL-selected dates; do not claim complete 70% optical mapping",
        "attribution": {"observations": "Copernicus Sentinel-2 / ESA", "public_catalogue_and_COGs": "Element84 Earth Search"}}
    dump(args.out / "protocol.json", protocol)
    print(json.dumps({"status": "protocol_frozen", "source_count": len(frozen), "new_pixel_reads": 0,
                      "new_network_requests": 0, "protocol": record(args.out / "protocol.json")}, indent=2))


def load_crop(item, key, index, protocol, window, grid, client, out):
    folder = out / f"source-{index:02d}"
    folder.mkdir(exist_ok=True)
    entry = protocol["cached_crops"].get(item["id"], {}).get(key)
    if entry is None:
        values, receipt = read_band(item, key, window, grid, client, folder)
    else:
        path = verify(entry["crop"])
        verify(entry["receipt"])
        validate_cached_metadata(path, item, key, window, grid)
        with rasterio.open(path) as ds:
            dn = ds.read(1)
        destination = folder / (key + "-dn.tif")
        shutil.copyfile(path, destination)
        receipt = {"crop": record(destination), "asset": item["assets"][key], "imported_from": entry,
                   "source_object_full_checksum_verified": False, "remote_identity": entry["remote_identity"]}
        if key == "scl":
            values = dn
        else:
            band = item["assets"][key]["raster:bands"][0]
            values, valid = reflectance(dn, band["scale"], band["offset"], band["nodata"])
            if key != "swir16":
                values, _ = average_2x2(values, valid)
    dump(folder / (key + "-crop-receipt.json"), receipt)
    return values, receipt


def execute(args):
    protocol_path = args.out / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema") != "akash-optical-composite-protocol-v1" or protocol["policy"] != POLICY:
        raise ValueError("Require the exact frozen composite protocol")
    if (args.out / "manifest.json").exists() or (args.out / "transport.json").exists():
        raise FileExistsError("Execution already recorded; preserve this run")
    for binding in (*protocol["code"], protocol["catalog"], protocol["cached_manifest"]):
        verify(binding)
    started = time.perf_counter()
    cpu_started = time.process_time()
    ordered = [(frozen, json.loads(verify(frozen["source_item"]).read_text())) for frozen in protocol["ordered_sources"]]
    window, grid, geometry = window_for(ordered[0][1], protocol["bbox"])
    allowed = [item["assets"][key]["href"] for _, item in ordered for key in (*BANDS, "scl")]
    client = AuditedCompositeRangeHTTP(allowed, deadline=started + 580)
    dims = (int(grid[3]), int(grid[2]))
    aoi = geometry_mask([mapping(geometry)], out_shape=dims, transform=grid[1], invert=True, all_touched=False)
    selected = {branch: np.zeros(dims, dtype="uint8") for branch in ("before", "after")}
    scls, receipts, source_counts = {}, {}, []
    stop_reason = None
    try:
        for frozen, item in ordered:
            index, branch = frozen["source_index"], frozen["branch"]
            values, receipt = load_crop(item, "scl", index, protocol, window, grid, client, args.out)
            scls[index] = values
            receipts[str(index)] = {"source": frozen, "scl": receipt}
            counts = select_source_index(selected[branch], values, aoi, index)
            counts.update({"source_index": index, "branch": branch, "item_id": item["id"], "datetime": frozen["datetime"]})
            source_counts.append(counts)
            print(json.dumps({"phase": "SCL_selection", **counts}), flush=True)
    except Exception as error:
        dump(args.out / "transport.json", client.audit())
        dump(args.out / "failed.json", {"status": "failed_before_complete_SCL_selection", "error_type": type(error).__name__,
             "message": str(error), "wall_seconds": time.perf_counter() - started, "new_body_bytes": client.bytes_read})
        raise
    joint_scl = aoi & (selected["before"] != 0) & (selected["after"] != 0)
    for branch in selected:
        write_layer(args.out / (branch + "-selected-source-index.tif"), selected[branch], grid, 0)
    # Persist selection before any new reflectance read; acquisition cannot change it.
    selection_receipt = {"selection_frozen_at": datetime.now(timezone.utc).isoformat(),
        "joint_scl_selected_cells": int(joint_scl.sum()), "source_counts": source_counts,
        "selection_outputs": {branch: record(args.out / (branch + "-selected-source-index.tif")) for branch in selected}}
    for counts in source_counts:
        counts["joint_selected_pixels"] = int((joint_scl & (selected[counts["branch"]] == counts["source_index"])).sum())
    acquisition_order = sorted((counts for counts in source_counts if counts["joint_selected_pixels"]),
        key=lambda counts: (counts["item_id"] not in protocol["cached_crops"], -counts["joint_selected_pixels"], counts["source_index"]))
    selection_receipt["reflectance_acquisition_order"] = [counts["source_index"] for counts in acquisition_order]
    dump(args.out / "selection.frozen.json", selection_receipt)
    composite = {branch: {key: np.full(dims, np.nan, dtype="float32") for key in BANDS} for branch in selected}
    available = {branch: np.zeros(dims, dtype=bool) for branch in selected}
    installed = []
    source_by_index = {frozen["source_index"]: (frozen, item) for frozen, item in ordered}
    for counts in acquisition_order:
        index, branch = counts["source_index"], counts["branch"]
        frozen, item = source_by_index[index]
        if time.perf_counter() - started >= 560:
            stop_reason = "Frozen time budget reserve reached; remaining selected cells unknown"
            break
        try:
            scene = {}
            for key in BANDS:
                scene[key], receipt = load_crop(item, key, index, protocol, window, grid, client, args.out)
                receipts[str(index)][key] = receipt
            result = install_selected_reflectance(composite[branch], available[branch], index, selected[branch], scene, scls[index], aoi)
            installed.append({"source_index": index, "branch": branch, **result})
            print(json.dumps({"phase": "reflectance_installed", "source_index": index, "branch": branch,
                              **result, "new_body_bytes": client.bytes_read, "wall_seconds": time.perf_counter() - started}), flush=True)
            del scene
        except Exception as error:
            stop_reason = {"source_index": index, "error_type": type(error).__name__, "message": str(error),
                "selected_pixels_remain_unknown": True, "automatic_retry": False}
            break
    labels, assessed, indices, counts = screen_composite(composite["before"], composite["after"], available["before"], available["after"], aoi)
    write_layer(args.out / "surface-change-candidates.tif", labels, grid, 255)
    write_layer(args.out / "valid-mask.tif", assessed.astype("uint8"), grid, 0)
    write_layer(args.out / "aoi-mask.tif", aoi.astype("uint8"), grid, 0)
    write_layer(args.out / "joint-scl-selection.tif", joint_scl.astype("uint8"), grid, 0)
    age_histograms = {}
    for branch in selected:
        age = np.zeros(dims, dtype="uint16")
        event = utc(protocol["event_date"] + "T00:00:00Z")
        for frozen, _ in ordered:
            if frozen["branch"] == branch:
                age[selected[branch] == frozen["source_index"]] = abs((utc(frozen["datetime"]).date() - event.date()).days)
        write_layer(args.out / (branch + "-selected-age-days.tif"), age, grid, 0)
        values, frequencies = np.unique(age[assessed], return_counts=True)
        age_histograms[branch] = dict(zip(map(str, values), map(int, frequencies)))
        for key, array in indices[branch].items():
            write_layer(args.out / (branch + "-" + key + ".tif"), np.where(assessed, array, -9999).astype("float32"), grid, -9999)
        for key, array in composite[branch].items():
            write_layer(args.out / (branch + "-" + key + "-reflectance.tif"), np.where(available[branch], array, -9999).astype("float32"), grid, -9999)
    unproject = Transformer.from_crs(grid[0], 4326, always_xy=True).transform
    features = []
    for geo, value in shapes(labels, mask=labels == 1, transform=grid[1], connectivity=4):
        polygon = shape(geo)
        if value != 1 or not polygon.is_valid:
            raise ValueError("Candidate polygon fails native geometry validation")
        features.append({"type": "Feature", "geometry": mapping(transform(unproject, polygon)), "properties": {
            "native_grid_area_m2": polygon.area, "candidate_basis": POLICY["scope"], "confirmed_debris": False,
            "confirmed_landslide": False, "road_closure_assigned": False, "multi_date_composite": True}})
    dump(args.out / "surface-change-candidates.geojson", {"type": "FeatureCollection", "features": features})
    rgb = np.full((*dims, 3), 255, dtype="uint8")
    rgb[aoi] = (193, 200, 207)
    rgb[assessed] = (232, 240, 230)
    rgb[labels == 1] = (147, 78, 35)
    write_png(args.out / "overview.png", rgb)
    dump(args.out / "transport.json", client.audit())
    manifest = {"schema": "akash-optical-composite-v1", "status": "completed", "protocol": record(protocol_path),
        "completed_at": datetime.now(timezone.utc).isoformat(), "selection": record(args.out / "selection.frozen.json"),
        "bbox": protocol["bbox"], "event_date": protocol["event_date"], "grid": protocol["grid"],
        "counts": counts, "joint_assessed_fraction": counts["joint_assessed_pixels"] / counts["aoi_pixels"],
        "joint_scl_selected_cells": int(joint_scl.sum()), "joint_scl_selected_fraction": int(joint_scl.sum()) / int(aoi.sum()),
        "candidate_area_m2": counts["candidate_pixels"] * 400, "source_counts": source_counts,
        "source_crops": receipts, "installed_sources": installed, "acquisition_stop_reason": stop_reason,
        "assessed_age_days_histograms": age_histograms, "original_pair_assessed_cells": 79555,
        "assessed_cell_count_difference_from_original_pair": counts["joint_assessed_pixels"] - 79555,
        "original_single_scene_gate_changed": False, "complete_optical_map_claim": False,
        "surface_change_thresholds_retuned": False, "local_debris_accuracy_established": False,
        "reference_labels_loaded": False, "gpu_used": False, "scene_date_alignment_resampling_used": False,
        "new_body_bytes": client.bytes_read, "network_attempts": len(client.requests),
        "transport": record(args.out / "transport.json"), "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "rss_platform_units": "macOS ru_maxrss reports bytes", "scope": POLICY["scope"],
        "outputs": {path.name: record(path) for path in args.out.iterdir() if path.suffix in (".tif", ".geojson", ".png")},
        "limitations": ["Different cells have different dates and ages; this is not a synoptic damage map or forecast.",
            "SCL is imperfect; cloud, shadow, snow, water, invalid and unacquired selected cells stay unknown.",
            "Vegetation loss and exposed surface can reflect agriculture, clearing, moisture, season or sensor effects; thresholds are engineering hypotheses.",
            "No local accuracy or debris coverage evaluation was performed; prior failed checking results remain unchanged.",
            "Old crop provenance retains original limitations; new successful ranges are hashed and ETag-bound, whole provider objects are not checksum verified."]}
    dump(args.out / "manifest.json", manifest)
    print(json.dumps({"status": "completed", "counts": counts, "new_body_bytes": client.bytes_read,
                      "wall_seconds": manifest["wall_seconds"], "peak_rss_bytes": manifest["peak_process_rss_bytes"],
                      "manifest": record(args.out / "manifest.json")}, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "outputs/fresh-public-catalog/catalog.json")
    parser.add_argument("--bbox", type=float, nargs=4, default=[85.3, 28.12, 85.43, 28.32])
    parser.add_argument("--event-date", default="2026-08-26")
    parser.add_argument("--reuse-pair", type=Path, default=ROOT / "outputs/s2-surface-change-resumed")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/optical-composite-2026-10-10")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    reject_source_path(args.out)
    execute(args) if args.execute else freeze(args)


if __name__ == "__main__":
    main()
