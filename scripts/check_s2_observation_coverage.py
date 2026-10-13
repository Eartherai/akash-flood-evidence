"""Freeze or run a separate SCL-only native20m optical observation gate.

Default: metadata/protocol only, with no pixel reads or network requests.
Execution consumes SCL only and produces date-index availability masks, never
reflectance indices, flood/debris labels, road closures or accuracy estimates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping

from akash_flood.catalog import utc, validate_bbox
from akash_flood.fresh_case import reject_source_path
from akash_flood.optical_observation import (AuditedSCLRangeHTTP, POLICY,
    select_observation_sources, summarize_availability, update_date_index)
from build_fresh_report import record
from build_s2_debris_candidates import asset_grid, read_band, window_for, write_layer

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_json(path):
    path = Path(path).resolve()
    reject_source_path(path)
    if path.stat().st_size > 20_000_000:
        raise ValueError("Saved source JSON exceeds the20MB gate cap")
    return json.loads(path.read_text())


def verify_binding(binding):
    path = Path(binding["path"]).resolve()
    reject_source_path(path)
    if record(path) != binding:
        raise ValueError("Saved source binding changed: " + path.name)
    return path


def validate_crop_metadata(path, item, window, grid):
    """Read metadata only; default protocol construction does not read pixels."""
    asset = item["assets"]["scl"]
    if list(window.flatten()) != [int(x) for x in window.flatten()]:
        raise ValueError("Integer native SCL crop is required")
    with rasterio.open(path) as ds:
        if (ds.count != 1 or ds.crs != grid[0] or ds.transform != grid[1]
                or ds.width != grid[2] or ds.height != grid[3]
                or ds.dtypes[0] != "uint8" or ds.nodata != asset["raster:bands"][0]["nodata"]):
            raise ValueError("Reused SCL crop differs from the exact native source grid/type/nodata")


def verify_range_identity(ranges, asset, identity):
    if (not ranges or identity["bytes"] != asset["file:size"] or not identity.get("etag")
            or any(entry.get("status") != 206 or entry.get("file_bytes") != identity["bytes"]
                   or entry.get("etag") != identity["etag"]
                   or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64
                   for entry in ranges)):
        raise ValueError("Reused SCL source lacks consistent successful range identities")


def prepare_scl_reuse(folder, sources, bbox, event_date, window, grid):
    """Import exactly the two SCL crops from the preserved completed pair.

    Original crop receipts, item records, protocol and successful source range
    ledgers are checked. Legacy failed-run hashes remain explicitly later-captured
    identities; no full provider object checksum is asserted verified.
    """
    if folder is None:
        return None
    folder = Path(folder).resolve()
    reject_source_path(folder)
    manifest_path = folder / "manifest.json"
    manifest = load_json(manifest_path)
    if (manifest.get("schema") != "akash-s2-surface-change-v1" or manifest.get("status") != "completed"
            or manifest.get("reference_labels_loaded") is not False
            or manifest.get("bbox") != bbox or manifest.get("event_date") != event_date):
        raise ValueError("Reuse requires the completed, same-case optical pair manifest")
    protocol = load_json(verify_binding(manifest["protocol"]))
    if (protocol.get("schema") != "akash-s2-surface-change-protocol-v1"
            or protocol.get("reference_labels_loaded") is not False
            or protocol.get("bbox") != bbox or protocol.get("event_date") != event_date):
        raise ValueError("Reused source protocol identity differs")
    transport_path = verify_binding(manifest["transport"])
    transport = load_json(transport_path)
    result = {"source_manifest": record(manifest_path), "source_protocol": manifest["protocol"],
              "source_transport": manifest["transport"], "crops": {},
              "pixel_reads_during_protocol_preparation": False,
              "source_whole_object_checksums_verified": False,
              "legacy_source_transport_limits": "Original pair successful-range ledgers omit failed request counts/partial bytes; no retrospective correction to the original runs",
              "scope": "Exact two archived SCL crops only; no reflectance raster read or reference input"}
    selected = {item["id"]: item for group in sources.values() for item in group}
    for branch in ("before", "after"):
        item_record = protocol["source_items"][branch]
        item = load_json(verify_binding(item_record))
        if item["id"] not in selected or selected[item["id"]] != item:
            raise ValueError("Archived SCL scene differs from the exact selected source metadata")
        crop_receipt = manifest["source_crops"][branch]["scl"]
        receipt_path = folder / branch / "scl-crop-receipt.json"
        if load_json(receipt_path) != crop_receipt:
            raise ValueError("Archived SCL crop receipt differs from completed manifest")
        asset, identity = item["assets"]["scl"], crop_receipt["remote_identity"]
        if (crop_receipt["asset"] != asset or crop_receipt.get("source_object_full_checksum_verified") is not False
                or crop_receipt.get("horizontal_aggregation") != "none"
                or crop_receipt.get("native_window") != list(window.flatten())):
            raise ValueError("Archived SCL crop source/native-window contract differs")
        crop_path = verify_binding(crop_receipt["crop"])
        validate_crop_metadata(crop_path, item, window, grid)
        ledger = transport["requests"]
        legacy_link = None
        if "imported_from" in crop_receipt:
            imported = crop_receipt["imported_from"]
            reuse = protocol.get("crop_reuse")
            if not reuse or reuse["crops"][branch]["scl"] != imported:
                raise ValueError("Archived imported SCL crop is not bound to the original reuse freeze")
            original_item = load_json(verify_binding(imported["source_item"]))
            if original_item != item or imported["remote_identity"] != identity:
                raise ValueError("Original imported SCL source identity differs")
            original_crop = verify_binding(imported["crop"])
            if (record(original_crop)["sha256"] != crop_receipt["crop"]["sha256"]
                    or record(original_crop)["bytes"] != crop_receipt["crop"]["bytes"]):
                raise ValueError("Imported SCL crop differs from the original exact bytes")
            failure_path = verify_binding(reuse["source_failure"])
            original_protocol_path = verify_binding(reuse["source_protocol"])
            failure, original_protocol = load_json(failure_path), load_json(original_protocol_path)
            if failure.get("status") != "failed" or original_protocol["source_items"][branch] != imported["source_item"]:
                raise ValueError("Legacy archived source run differs")
            ledger = failure["requests"]
            legacy_link = {"source_failure": reuse["source_failure"], "source_protocol": reuse["source_protocol"],
                           "hash_capture_scope": reuse["hash_capture_scope"]}
        ranges = [entry for entry in ledger if entry["url"] == asset["href"]]
        verify_range_identity(ranges, asset, identity)
        result["crops"][item["id"]] = {
            "crop": crop_receipt["crop"], "crop_receipt": record(receipt_path),
            "source_item": item_record, "source_asset": asset, "remote_identity": identity,
            "successful_range_count": len(ranges), "legacy_link": legacy_link,
            "source_whole_object_checksum_verified": False,
        }
    return result


def import_scl_crop(item, entry, window, grid, out):
    source = verify_binding(entry["crop"])
    if load_json(verify_binding(entry["source_item"])) != item or entry["source_asset"] != item["assets"]["scl"]:
        raise ValueError("Frozen reuse source item changed")
    verify_binding(entry["crop_receipt"])
    validate_crop_metadata(source, item, window, grid)
    with rasterio.open(source) as ds:
        values = ds.read(1)
    destination = out / "scl-dn.tif"
    shutil.copyfile(source, destination)
    return values, {
        "crop": record(destination), "asset": item["assets"]["scl"], "imported_from": entry,
        "remote_identity": entry["remote_identity"], "native_window": list(window.flatten()),
        "horizontal_aggregation": "none", "source_object_full_checksum_verified": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--bbox", type=float, nargs=4, required=True)
    parser.add_argument("--event-date", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse-scl", type=Path, help="Explicitly import exactly two verified archived SCL crops")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("Preserve prior optical experiments; a fresh output folder is required")
    reject_source_path(args.out)
    reject_source_path(args.catalog)
    bbox = validate_bbox(args.bbox)
    sources, excluded = select_observation_sources(load_json(args.catalog), bbox, args.event_date)
    ordered = [(branch, item) for branch in ("before", "after") for item in sources[branch]]
    window, grid, geometry = window_for(ordered[0][1], bbox)
    for _, item in ordered:
        other_window, other_grid, _ = window_for(item, bbox)
        if grid != other_grid or window != other_window:
            raise ValueError("Native date windows differ; no resampling/alignment is allowed")
        asset_grid(item, "scl")
    reuse = prepare_scl_reuse(args.reuse_scl, sources, bbox, args.event_date, window, grid)
    args.out.mkdir(parents=True)
    frozen_items = []
    for index, (branch, item) in enumerate(ordered, 1):
        path = args.out / f"source-{index:02d}.stac.json"
        dump(path, item)
        frozen_items.append({"source_index": index, "branch": branch, "item_id": item["id"],
                             "datetime": item["properties"]["datetime"], "source_item": record(path)})
    protocol = {
        "schema": "akash-scl-observation-protocol-v1", "execute_requested": args.execute,
        "recorded_before_pixel_acquisition_at": datetime.now(timezone.utc).isoformat(),
        "catalog": record(args.catalog), "bbox": bbox, "event_date": args.event_date,
        "policy": POLICY, "ordered_sources": frozen_items, "excluded_optical_item_ids": excluded,
        "grid": {"crs": str(grid[0]), "transform": list(grid[1]), "shape": [int(grid[3]), int(grid[2])]},
        "native_window": list(window.flatten()), "scl_reuse": reuse,
        "code": [record(Path(__file__)), record(ROOT / "src/akash_flood/optical_observation.py"),
                 record(ROOT / "scripts/build_s2_debris_candidates.py"), record(ROOT / "src/akash_flood/s1_crop.py")],
        "known_context": "Designed after known Rasuwa single-pair optical coverage, earlier multi-date SCL coverage and checking results; this is observation-only feasibility, never blind hazard validation",
        "reference_labels_loaded": False, "reflectance_pixels_loaded": False, "gpu_used": False,
        "maximum_new_http_body_bytes": 20_000_000, "maximum_attempted_gets": 256,
        "source_full_object_checksum_verified": False,
        "attribution": {"SCL": "Copernicus Sentinel-2 / ESA", "public_catalogue_and_COGs": "Element84 Earth Search",
                        "SCL_documentation": "https://sentiwiki.copernicus.eu/web/s2-processing"},
    }
    dump(args.out / "protocol.json", protocol)
    if not args.execute:
        print(json.dumps({"status": "protocol_only", "network_requests": 0, "pixels_read": False,
                          "selected_source_counts": {branch: len(items) for branch, items in sources.items()},
                          "protocol": record(args.out / "protocol.json")}, indent=2))
        return
    client = AuditedSCLRangeHTTP()
    started = time.perf_counter()
    indexes = {branch: np.zeros((int(grid[3]), int(grid[2])), dtype="uint8") for branch in sources}
    aoi = geometry_mask([mapping(geometry)], out_shape=indexes["before"].shape,
                        transform=grid[1], invert=True, all_touched=False)
    receipts, source_counts = {}, []
    try:
        for frozen, (branch, item) in zip(frozen_items, ordered):
            index = frozen["source_index"]
            folder = args.out / f"source-{index:02d}"
            folder.mkdir()
            entry = reuse["crops"].get(item["id"]) if reuse else None
            if entry:
                values, receipt = import_scl_crop(item, entry, window, grid, folder)
            else:
                values, receipt = read_band(item, "scl", window, grid, client, folder)
            dump(folder / "scl-crop-receipt.json", receipt)
            receipts[str(index)] = {"source": frozen, "scl": receipt,
                                    "crop_receipt": record(folder / "scl-crop-receipt.json")}
            counts = update_date_index(indexes[branch], values, aoi, index)
            when = utc(item["properties"]["datetime"])
            counts.update({"source_index": index, "branch": branch, "item_id": item["id"],
                           "datetime": item["properties"]["datetime"],
                           "calendar_days_from_event_start": (when.date() - utc(args.event_date + "T00:00:00Z").date()).days})
            source_counts.append(counts)
        joint, counts, fraction = summarize_availability(indexes["before"], indexes["after"], aoi)
        for branch in sources:
            write_layer(args.out / (branch + "-date-index.tif"), indexes[branch], grid, 0)
        write_layer(args.out / "aoi-mask.tif", aoi.astype("uint8"), grid, 0)
        write_layer(args.out / "joint-availability.tif", joint.astype("uint8"), grid, 0)
        transport = client.audit()
        dump(args.out / "transport.json", transport)
        manifest = {
            "schema": "akash-scl-observation-availability-v1", "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(), "protocol": record(args.out / "protocol.json"),
            "bbox": bbox, "event_date": args.event_date, "grid": protocol["grid"],
            "ordered_sources": frozen_items, "source_crops": receipts, "source_counts": source_counts,
            "counts": counts, "joint_scl_availability_fraction": fraction,
            "counts_mean_accuracy": False, "candidate_layer_generated": False,
            "joint_scl_availability_is_reflectance_validity": False,
            "original_single_scene_gate_pass_asserted": False, "synoptic_assessment": False,
            "reference_labels_loaded": False, "reflectance_pixels_loaded": False, "gpu_used": False,
            "date_alignment_resampling_used": False, "source_whole_object_checksums_verified": False,
            "network_attempted_gets": transport["attempted_get_count"],
            "network_received_body_bytes": transport["received_body_bytes"], "transport": record(args.out / "transport.json"),
            "imported_scl_crop_count": sum("imported_from" in value["scl"] for value in receipts.values()),
            "outputs": {path.name: record(path) for path in args.out.glob("*.tif")},
            "wall_seconds": time.perf_counter() - started,
            "limitations": [
                "SCL is an imperfect classifier, not cloud/land truth or hazard truth.",
                "Each source is eroded independently by60m around non-land/invalid classes; water/shadow/cloud/snow are unsupported.",
                "Per-cell source dates differ. Latest pre and earliest post are acquisition rules, not a single synoptic observation.",
                "Sources up to24days before/after the event-date UTC start may miss transient floods or reflect later recovery/seasonal changes.",
                "SCL-only opportunity is an upper bound: actual four-band coherent same-source valid reflectance may reduce availability.",
                "No reflectance change, debris/flood prediction, absolute geolocation accuracy or road passability is established.",
                "The known Rasuwa case is not an untouched validation, and the fixed earlier single-pair result remains preserved.",
                "Range hashes and ETag/length consistency bind delivered bytes; whole provider objects/digests are not independently verified.",
            ],
        }
        dump(args.out / "manifest.json", manifest)
        print(json.dumps({"status": "completed", "counts": counts,
                          "joint_scl_availability_fraction": fraction,
                          "network_attempted_gets": transport["attempted_get_count"],
                          "network_received_body_bytes": transport["received_body_bytes"],
                          "coverage_is_accuracy": False, "manifest": record(args.out / "manifest.json")}, indent=2))
    except Exception as error:
        dump(args.out / "transport.json", client.audit())
        dump(args.out / "failed.json", {
            "status": "failed", "error_type": type(error).__name__, "message": str(error),
            "protocol": record(args.out / "protocol.json"), "transport": record(args.out / "transport.json"),
            "received_body_bytes": client.bytes_read, "attempted_get_count": len(client.requests),
            "no_automatic_retry": True, "source_crops_completed_before_failure": receipts,
            "reference_labels_loaded": False, "reflectance_pixels_loaded": False,
        })
        raise


if __name__ == "__main__":
    main()
