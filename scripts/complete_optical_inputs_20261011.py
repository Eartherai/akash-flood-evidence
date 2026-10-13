"""Complete only the already frozen Oct10 optical reflectance acquisitions.

The original SCL pixel/date selection and spectral rule are preserved. Freeze
first, then execute once into fresh data/output folders; no reference is read.
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
from rasterio.features import shapes
from shapely.geometry import mapping, shape
from shapely.ops import transform

from akash_flood.catalog import utc
from akash_flood.optical_change import reflectance, average_2x2
from akash_flood.optical_composite import (BANDS, POLICY, AuditedCompositeRangeHTTP,
    install_selected_reflectance, screen_composite, select_source_index)
from build_fresh_report import record, write_png
from build_s2_debris_candidates import read_band, window_for, write_layer
from run_optical_composite import validate_cached_metadata, verify

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cached_entry(folder, item, key, window, grid):
    receipt_path = folder / (key + "-crop-receipt.json")
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt["asset"] != item["assets"][key]:
        raise ValueError("Completed crop receipt differs from the frozen asset")
    crop = verify(receipt["crop"])
    validate_cached_metadata(crop, item, key, window, grid)
    return {"crop": receipt["crop"], "receipt": record(receipt_path)}


def acquisition_plan(order, entries):
    """The preserved acquisition order cannot depend on spectra or outcomes."""
    return [{"source_index": index, "missing_bands": [key for key in BANDS
                if key not in entries[str(index)]]}
            for index in order if any(key not in entries[str(index)] for key in BANDS)]


def freeze(args):
    if args.research.exists() or args.data.exists() or args.out.exists():
        raise FileExistsError("Completion requires three fresh folders")
    original_protocol = json.loads((args.prior / "protocol.json").read_text())
    prior = json.loads((args.prior / "manifest.json").read_text())
    selection = json.loads((args.prior / "selection.frozen.json").read_text())
    if (prior["schema"] != "akash-optical-composite-v1" or prior["status"] != "completed"
            or original_protocol["policy"] != POLICY or prior["reference_labels_loaded"] is not False
            or prior["surface_change_thresholds_retuned"] is not False
            or prior["acquisition_stop_reason"]["source_index"] != 11):
        raise ValueError("Require the preserved unchanged Oct10 partial acquisition")
    verify(prior["protocol"])
    verify(prior["selection"])
    verify(prior["transport"])
    for binding in original_protocol["code"]:
        verify(binding)
    ordered = original_protocol["ordered_sources"]
    first_item = json.loads(verify(ordered[0]["source_item"]).read_text())
    window, grid, _ = window_for(first_item, original_protocol["bbox"])
    entries = {}
    for frozen in ordered:
        index = frozen["source_index"]
        item = json.loads(verify(frozen["source_item"]).read_text())
        entries[str(index)] = {}
        for key in (*BANDS, "scl"):
            entry = cached_entry(args.prior / f"source-{index:02d}", item, key, window, grid)
            if entry is not None:
                entries[str(index)][key] = entry
        if "scl" not in entries[str(index)]:
            raise ValueError("All thirteen selected SCL crops must remain available")
    bindings = {name: record(args.prior / name) for name in (
        "protocol.json", "selection.frozen.json", "manifest.json", "transport.json",
        "before-selected-source-index.tif", "after-selected-source-index.tif",
        "joint-scl-selection.tif", "aoi-mask.tif", "valid-mask.tif", "surface-change-candidates.tif")}
    for name in ("before-selected-source-index.tif", "after-selected-source-index.tif",
                 "joint-scl-selection.tif", "aoi-mask.tif", "valid-mask.tif", "surface-change-candidates.tif"):
        if prior["outputs"][name] != bindings[name]:
            raise ValueError("Preserved output hash changed before completion freeze")
    args.research.mkdir(parents=True)
    args.data.mkdir(parents=True)
    args.out.mkdir(parents=True)
    protocol = {"schema": "akash-optical-completion-protocol-v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "decision": "Complete original fixed SCL-selected reflectance acquisitions; never select a replacement date",
        "policy": POLICY, "original_bindings": bindings, "ordered_sources": ordered,
        "cached_crops": entries, "acquisition_order": selection["reflectance_acquisition_order"],
        "missing_acquisitions": acquisition_plan(selection["reflectance_acquisition_order"], entries),
        "bbox": original_protocol["bbox"], "event_date": original_protocol["event_date"],
        "grid": original_protocol["grid"], "native_window": original_protocol["native_window"],
        "code": [record(Path(__file__).resolve()), *original_protocol["code"]],
        "data_folder": str(args.data.resolve()), "output_folder": str(args.out.resolve()),
        "maximum_new_body_bytes": 200_000_000, "maximum_acquisition_processing_seconds": 600,
        "maximum_attempted_gets": 1024, "automatic_retry": False,
        "maximum_additional_assessed_cells_scl_bound": 20570,
        "source_dates_and_pixel_selection_changed": False, "thresholds_changed": False,
        "known_context": "Designed after earlier production support counts and failed acquisition were known; not blind validation",
        "reference_labels_loaded": False, "new_remote_pixels_read_before_freeze": False,
        "gpu_used": False, "credentials_used": False,
        "cache_provenance": "Preserved original receipts and failed transport remain authoritative; no retrospective checksum upgrade",
        "attribution": {"observations": "Copernicus Sentinel-2 / ESA", "public_catalogue_and_COGs": "Element84 Earth Search"}}
    dump(args.research / "protocol.json", protocol)
    shutil.copyfile(args.research / "protocol.json", args.out / "protocol.json")
    print(json.dumps({"status": "frozen_before_new_remote_pixels", "protocol": record(args.research / "protocol.json"),
                      "missing_acquisitions": protocol["missing_acquisitions"]}, indent=2))


def read_frozen_crop(entry, item, key, window, grid, folder):
    path = verify(entry["crop"])
    verify(entry["receipt"])
    validate_cached_metadata(path, item, key, window, grid)
    destination = folder / (key + "-dn.tif")
    shutil.copyfile(path, destination)
    with rasterio.open(destination) as ds:
        dn = ds.read(1)
    receipt = {"crop": record(destination), "asset": item["assets"][key],
               "imported_from": entry, "source_object_full_checksum_verified": False,
               "remote_identity": json.loads(Path(entry["receipt"]["path"]).read_text())["remote_identity"]}
    if key == "scl":
        return dn, receipt
    band = item["assets"][key]["raster:bands"][0]
    values, valid = reflectance(dn, band["scale"], band["offset"], band["nodata"])
    if key != "swir16":
        values, _ = average_2x2(values, valid)
    return values, receipt


def execute(args):
    protocol_path = args.research / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if (protocol["schema"] != "akash-optical-completion-protocol-v1" or protocol["policy"] != POLICY
            or str(args.out.resolve()) != protocol["output_folder"]
            or str(args.data.resolve()) != protocol["data_folder"]
            or record(args.out / "protocol.json") != {**record(protocol_path), "path": str((args.out / "protocol.json").resolve())}):
        raise ValueError("Require the exact frozen completion protocol and folders")
    if any((args.out / name).exists() for name in ("manifest.json", "transport.json", "failed.json")):
        raise FileExistsError("Execution is already recorded; preserve this run")
    for binding in (*protocol["code"], *protocol["original_bindings"].values()):
        verify(binding)
    started, cpu_started = time.perf_counter(), time.process_time()
    ordered = [(f, json.loads(verify(f["source_item"]).read_text())) for f in protocol["ordered_sources"]]
    source_by_index = {f["source_index"]: (f, item) for f, item in ordered}
    window, grid, _ = window_for(ordered[0][1], protocol["bbox"])
    arrays = {}
    for key in ("before-selected-source-index.tif", "after-selected-source-index.tif", "aoi-mask.tif", "joint-scl-selection.tif"):
        source = verify(protocol["original_bindings"][key])
        shutil.copyfile(source, args.out / key)
        with rasterio.open(source) as ds:
            arrays[key] = ds.read(1)
    aoi = arrays["aoi-mask.tif"].astype(bool)
    joint_scl = arrays["joint-scl-selection.tif"].astype(bool)
    selected = {b: arrays[b + "-selected-source-index.tif"] for b in ("before", "after")}
    allowed = [item["assets"][key]["href"] for _, item in ordered for key in BANDS]
    client = AuditedCompositeRangeHTTP(allowed, max_bytes=protocol["maximum_new_body_bytes"], deadline=started + 580)
    composite = {b: {key: np.full(aoi.shape, np.nan, dtype="float32") for key in BANDS} for b in selected}
    available = {b: np.zeros(aoi.shape, dtype=bool) for b in selected}
    scls, receipts = {}, {}
    for f, item in ordered:
        index = f["source_index"]
        folder = args.data / f"source-{index:02d}"
        folder.mkdir()
        values, receipt = read_frozen_crop(protocol["cached_crops"][str(index)]["scl"], item, "scl", window, grid, folder)
        dump(folder / "scl-crop-receipt.json", receipt)
        scls[index] = values
        receipts[str(index)] = {"source": f, "scl": receipt}
    # Recompute selection from SCL alone and reject drift before acquiring bands.
    recomputed = {b: np.zeros(aoi.shape, dtype="uint8") for b in selected}
    for f, _ in ordered:
        select_source_index(recomputed[f["branch"]], scls[f["source_index"]], aoi, f["source_index"])
    if any(not np.array_equal(selected[b], recomputed[b]) for b in selected):
        raise ValueError("Preserved selected date indices disagree with their SCL-only rule")
    if not np.array_equal(joint_scl, aoi & (selected["before"] != 0) & (selected["after"] != 0)):
        raise ValueError("Preserved joint SCL selection differs")
    installed, stop_reason = [], None
    for index in protocol["acquisition_order"]:
        f, item = source_by_index[index]
        branch = f["branch"]
        folder = args.data / f"source-{index:02d}"
        if time.perf_counter() - started >= 560:
            stop_reason = "Frozen completion deadline reserve reached; unacquired selected cells remain unknown"
            break
        try:
            scene = {}
            for key in BANDS:
                entry = protocol["cached_crops"][str(index)].get(key)
                values, receipt = (read_frozen_crop(entry, item, key, window, grid, folder) if entry
                    else read_band(item, key, window, grid, client, folder))
                dump(folder / (key + "-crop-receipt.json"), receipt)
                receipts[str(index)][key] = receipt
                scene[key] = values
            installed.append({"source_index": index, "branch": branch,
                **install_selected_reflectance(composite[branch], available[branch], index, selected[branch], scene, scls[index], aoi)})
            print(json.dumps({"phase": "fixed_source_installed", **installed[-1], "new_body_bytes": client.bytes_read}), flush=True)
        except Exception as error:
            stop_reason = {"source_index": index, "error_type": type(error).__name__, "message": str(error),
                           "automatic_retry": False, "selected_pixels_remain_unknown": True}
            break
    labels, assessed, indices, counts = screen_composite(composite["before"], composite["after"], available["before"], available["after"], aoi)
    with rasterio.open(verify(protocol["original_bindings"]["valid-mask.tif"])) as ds:
        old_assessed = ds.read(1).astype(bool)
    with rasterio.open(verify(protocol["original_bindings"]["surface-change-candidates.tif"])) as ds:
        old_labels = ds.read(1)
    retained = int((old_assessed & assessed).sum())
    lost = int((old_assessed & ~assessed).sum())
    if lost or not np.array_equal(labels[old_assessed], old_labels[old_assessed]):
        raise ValueError("Completion changed or removed previously assessed production classifications")
    write_layer(args.out / "surface-change-candidates.tif", labels, grid, 255)
    write_layer(args.out / "valid-mask.tif", assessed.astype("uint8"), grid, 0)
    age_histograms = {}
    for branch in selected:
        age = np.zeros(aoi.shape, dtype="uint16")
        event = utc(protocol["event_date"] + "T00:00:00Z")
        for f, _ in ordered:
            if f["branch"] == branch:
                age[selected[branch] == f["source_index"]] = abs((utc(f["datetime"]).date() - event.date()).days)
        write_layer(args.out / (branch + "-selected-age-days.tif"), age, grid, 0)
        values, freq = np.unique(age[assessed], return_counts=True)
        age_histograms[branch] = dict(zip(map(str, values), map(int, freq)))
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
    rgb = np.full((*aoi.shape, 3), 255, dtype="uint8")
    rgb[aoi], rgb[assessed], rgb[labels == 1] = (193, 200, 207), (232, 240, 230), (147, 78, 35)
    write_png(args.out / "overview.png", rgb)
    dump(args.out / "transport.json", client.audit())
    prior = json.loads(verify(protocol["original_bindings"]["manifest.json"]).read_text())
    selection = json.loads(verify(protocol["original_bindings"]["selection.frozen.json"]).read_text())
    shutil.copyfile(verify(protocol["original_bindings"]["selection.frozen.json"]), args.out / "selection.original.frozen.json")
    manifest = {"schema": "akash-optical-composite-v1", "status": "completed",
        "completion_schema": "akash-fixed-selection-optical-completion-v1", "protocol": record(args.out / "protocol.json"),
        "completed_at": datetime.now(timezone.utc).isoformat(), "selection": record(args.out / "selection.original.frozen.json"),
        "bbox": protocol["bbox"], "event_date": protocol["event_date"], "grid": protocol["grid"],
        "counts": counts, "joint_assessed_fraction": counts["joint_assessed_pixels"] / counts["aoi_pixels"],
        "joint_scl_selected_cells": int(joint_scl.sum()), "joint_scl_selected_fraction": float(joint_scl.sum() / aoi.sum()),
        "candidate_area_m2": counts["candidate_pixels"] * 400, "source_counts": selection["source_counts"],
        "source_crops": receipts, "installed_sources": installed, "acquisition_stop_reason": stop_reason,
        "assessed_age_days_histograms": age_histograms, "retention": {"previous_assessed_cells": int(old_assessed.sum()),
            "retained_assessed_cells": retained, "lost_assessed_cells": lost,
            "newly_assessed_cells": int((assessed & ~old_assessed).sum()), "previous_classifications_changed": 0},
        "original_pair_assessed_cells": 79555,
        "assessed_cell_count_difference_from_original_pair": counts["joint_assessed_pixels"] - 79555,
        "original_single_scene_gate_changed": False, "complete_optical_map_claim": False,
        "surface_change_thresholds_retuned": False, "local_debris_accuracy_established": False,
        "reference_labels_loaded": False, "gpu_used": False, "scene_date_alignment_resampling_used": False,
        "source_dates_and_pixel_selection_changed": False, "new_body_bytes": client.bytes_read,
        "prior_composite_new_body_bytes": prior["new_body_bytes"],
        "recorded_composite_campaign_body_bytes": prior["new_body_bytes"] + client.bytes_read,
        "network_attempts": len(client.requests), "transport": record(args.out / "transport.json"),
        "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "rss_platform_units": "macOS ru_maxrss reports bytes", "scope": POLICY["scope"],
        "outputs": {p.name: record(p) for p in args.out.iterdir() if p.suffix in (".tif", ".geojson", ".png")},
        "limitations": prior["limitations"] + ["This is completion of a known partial acquisition, not new blind validation."]}
    dump(args.out / "manifest.json", manifest)
    dump(args.research / "execution-receipt.json", {"manifest": record(args.out / "manifest.json"),
         "transport": record(args.out / "transport.json"), "protocol": record(protocol_path)})
    print(json.dumps({"status": "completed", "counts": counts, "retention": manifest["retention"],
                      "new_body_bytes": client.bytes_read, "stop_reason": stop_reason}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=ROOT / "outputs/optical-composite-2026-10-10")
    parser.add_argument("--research", type=Path, default=ROOT / "research/optical-completion-2026-10-11")
    parser.add_argument("--data", type=Path, default=ROOT / "data/optical-completion-2026-10-11")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/optical-completion-2026-10-11")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    freeze(args) if args.freeze else execute(args)


if __name__ == "__main__":
    main()
