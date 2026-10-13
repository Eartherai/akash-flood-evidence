"""Process a checksum-bound fresh case with local SNAP and eligible CopDEM.

This runner acquires nothing. It requires completed native acquisition, prepared
ellipsoidal Copernicus DEM and the separately proved local DEM-reader overlay.
By default it writes a checked processing plan only; --execute runs SNAP.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import numpy as np
from pyproj import CRS
import rasterio
from affine import Affine
from rasterio.windows import from_bounds, Window

from shapely.geometry import box, shape

from akash_flood.catalog import utc, validate_bbox
from akash_flood.fresh_case import (canonical_sha, event_boundary, plan_projection,
                                   reject_source_path, validate_item_source)
from akash_flood.snap_products import read_snap_product, product_summary, integer_overlap
from prepare_copdem_ellipsoid import GRID_SHA256, GRID_BYTES
from process_s1_copdem import verified_reader_patch
from snap_gpt import command

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def bound_path(item, parent, *, label="input"):
    path = Path(item["path"])
    if not path.is_absolute():
        path = parent / path
    path = path.resolve()
    reject_source_path(path)
    if (not path.is_file() or sha256(path) != item["sha256"]
            or ("bytes" in item and path.stat().st_size != item["bytes"])):
        raise ValueError(f"Checksum or size mismatch for {label}")
    return path


def load_plan(path):
    reject_source_path(path)
    plan = json.loads(Path(path).read_text())
    case = plan["case"]
    bbox, context = validate_bbox(case["bbox"]), validate_bbox(case["context_bbox"])
    if not (context[0] <= bbox[0] < bbox[2] <= context[2]
            and context[1] <= bbox[1] < bbox[3] <= context[3]):
        raise ValueError("Processing context must cover the entire requested AOI")
    projection = plan_projection(context)
    if projection["status"] != "single_utm_zone":
        raise ValueError("Processing requires a reviewed single UTM zone and hemisphere")
    # The planner derives the AOI projection; the wider processing context must
    # fit that same zone/hemisphere, rather than silently changing its CRS.
    if (plan["projection"].get("epsg") != projection["epsg"]
            or plan["projection"].get("epsg") != plan_projection(bbox).get("epsg")):
        raise ValueError("Planned projection differs from the independently derived case/context UTM")
    if projection["proposed_grid"]["estimated_pixels"] > 20_000_000:
        raise ValueError("Processing context exceeds the explicit 20 million pixel local limit")
    pair = plan["sentinel1"].get("selected_pair")
    if not pair:
        raise ValueError("The plan has no selected native Sentinel-1 pair")
    orbit = pair.get("relative_orbit")
    if (isinstance(orbit, bool) or not isinstance(orbit, int) or not 1 <= orbit <= 175
            or pair.get("orbit_state") not in ("ascending", "descending")
            or pair.get("mode") != "IW" or set(pair.get("polarizations", [])) != {"VV", "VH"}):
        raise ValueError("Unsupported selected native orbit/mode/polarization pair")
    event_start, event_end, _ = event_boundary(case["event_date"], case["osm_snapshot"])
    items = {}
    for scene in ("before", "after"):
        selected = pair[scene]
        item_path = bound_path({"path": selected["item_path"], "sha256": selected["item_sha256"]},
                               Path(path).parent, label=scene + " STAC item")
        item = json.loads(item_path.read_text())
        properties = item["properties"]
        if (item["id"] != selected["item_id"]
                or properties["datetime"] != selected["datetime"]
                or canonical_sha(item) != selected["catalogue_item_canonical_sha256"]):
            raise ValueError("Selected STAC content differs from the case plan")
        if (validate_item_source(item) != "sentinel1"
                or properties.get("sat:relative_orbit") != orbit
                or properties.get("sat:orbit_state") != pair["orbit_state"]
                or properties.get("sar:instrument_mode") != "IW"
                or str(properties.get("platform", "")).casefold() != pair["platform"].casefold()
                or set(properties.get("sar:polarizations", [])) != {"VV", "VH"}
                or not shape(item["geometry"]).covers(box(*context))):
            raise ValueError("Native STAC source does not meet planned compatible context/orbit constraints")
        acquired = utc(properties["datetime"])
        if not (acquired < event_start if scene == "before" else acquired >= event_end):
            raise ValueError("Native source is not on the correct side of the excluded event day")
        items[scene] = (item_path, item)
    separation = (utc(items["after"][1]["properties"]["datetime"]) -
                  utc(items["before"][1]["properties"]["datetime"])).total_seconds() / 86400
    if abs(separation / 6 - round(separation / 6)) * 6 > .1:
        raise ValueError("Native pair separation is not compatible with the repeat-cycle rule")
    return plan, bbox, context, projection, items


def acquired_sources(index_path, plan_path, plan, items):
    reject_source_path(index_path)
    index = json.loads(index_path.read_text())
    if (index.get("schema") != "akash-fresh-native-acquisition-v1"
            or index.get("status") != "completed"):
        raise ValueError("Native acquisition is not complete")
    source_plan = bound_path(index["source_plan"], index_path.parent, label="acquisition source plan")
    if source_plan != plan_path.resolve() or index["source_plan"]["sha256"] != sha256(plan_path):
        raise ValueError("Acquisition is bound to a different case plan")
    sources = {}
    for scene in ("before", "after"):
        entry = index["scenes"][scene]
        selected = plan["sentinel1"]["selected_pair"][scene]
        stac_record = index["input_stac"][scene]
        stac = bound_path(stac_record, index_path.parent, label=scene + " acquisition STAC")
        if (stac != items[scene][0].resolve() or stac_record["sha256"] != selected["item_sha256"]
                or entry["source_item_id"] != selected["item_id"]
                or entry["datetime"] != selected["datetime"]
                or entry.get("not_original_esa_safe") is not True):
            raise ValueError("Acquired scene identity differs from the planned derived-wrapper source")
        safe = Path(entry["derived_safe"])
        if not safe.is_absolute():
            safe = index_path.parent / safe
        safe = safe.resolve()
        manifest = bound_path({"path": entry["derived_manifest"], "sha256": entry["derived_manifest_sha256"]},
                              index_path.parent, label=scene + " derived SAFE manifest")
        if manifest != safe / "manifest.safe":
            raise ValueError("Derived SAFE manifest is outside its declared wrapper")
        original = bound_path({"path": entry["original_manifest"], "sha256": entry["original_manifest_sha256"]},
                              index_path.parent, label=scene + " original provider manifest")
        aliases = set()
        if not entry.get("files"):
            raise ValueError("Derived SAFE has no checksum-bound source objects")
        for item in entry["files"]:
            alias = Path(item["alias"])
            if alias.is_absolute() or alias in aliases or ".." in alias.parts:
                raise ValueError("Unsafe or duplicate derived-wrapper alias")
            aliases.add(alias)
            target = (safe / alias).resolve()
            if not target.is_relative_to(safe):
                raise ValueError("Derived-wrapper object resolves outside its wrapper")
            bound_path({**item, "path": str(target)}, safe, label=scene + " wrapper object")
        sources[scene] = {"manifest": manifest, "entry": entry, "original_manifest": original}
    return sources, index


def prepared_dem(manifest_path, context):
    reject_source_path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "akash-fresh-dem-v1" or not manifest.get("completed_at")
            or manifest.get("context_bbox") != list(context)
            or manifest.get("snap_settings", {}).get("externalDEMApplyEGM") is not False
            or manifest["output"].get("vertical_datum") != "WGS84 ellipsoidal height"
            or manifest["output"].get("horizontal_crs") != "EPSG:4326"):
        raise ValueError("Require a completed context-matched ellipsoidal Copernicus DEM preparation")
    tiles = manifest.get("source_tiles")
    if not tiles:
        raise ValueError("Prepared DEM has no eligible Copernicus source tiles")
    for tile in tiles:
        url = urlsplit(tile["url"])
        if url.scheme != "https" or url.hostname != "copernicus-dem-30m.s3.amazonaws.com":
            raise ValueError("Prepared DEM contains an unsupported elevation source")
        bound_path(tile, manifest_path.parent, label="Copernicus source tile")
    geoid = manifest.get("geoid", {})
    if geoid.get("sha256") != GRID_SHA256 or geoid.get("bytes") != GRID_BYTES:
        raise ValueError("Prepared DEM requires the reviewed official NGA EGM2008 geoid")
    bound_path(geoid, manifest_path.parent, label="EGM2008 geoid")
    diagnostic_path = bound_path(manifest["diagnostics"], manifest_path.parent, label="DEM conversion diagnostics")
    diagnostic = json.loads(diagnostic_path.read_text())
    if (diagnostic.get("formula") != "h_WGS84 = H_EGM2008 + N_EGM2008"
            or manifest["output"].get("formula") != diagnostic["formula"]
            or diagnostic.get("mandatory_local_grid") is not True
            or diagnostic.get("fallback_allowed") is not False
            or diagnostic.get("network_enabled_for_proj") is not False
            or diagnostic.get("independent_sample_count", 0) < 100
            or not 0 <= diagnostic.get("max_proj_vs_independent_error_m", float("inf")) < 1e-6):
        raise ValueError("Prepared DEM lacks the checksum-bound numerical datum conversion proof")
    output = bound_path(manifest["output"], manifest_path.parent, label="ellipsoidal DEM")
    with rasterio.open(output) as dataset:
        if (dataset.crs != rasterio.crs.CRS.from_epsg(4326) or dataset.count != 1
                or dataset.transform.b != 0 or dataset.transform.d != 0
                or dataset.transform.a <= 0 or dataset.transform.e >= 0):
            raise ValueError("Prepared DEM must be a single north-up geographic WGS84 band")
        nodata = dataset.nodata
        if (nodata is None or not np.isfinite(nodata)
                or nodata != manifest["output"]["nodata"]):
            raise ValueError("Prepared DEM needs matching explicit finite nodata")
        west, south, east, north = context
        # A margin is required for interpolation; extent equality alone is not
        # evidence that the full context can be sampled without missing values.
        window = from_bounds(west, south, east, north, dataset.transform)
        left, top = int(np.floor(window.col_off)) - 1, int(np.floor(window.row_off)) - 1
        right = int(np.ceil(window.col_off + window.width)) + 1
        bottom = int(np.ceil(window.row_off + window.height)) + 1
        if left < 0 or top < 0 or right > dataset.width or bottom > dataset.height:
            raise ValueError("Prepared DEM does not cover processing context plus interpolation margin")
        values = dataset.read(1, window=Window(left, top, right-left, bottom-top), masked=True)
        if np.ma.getmaskarray(values).any() or not np.isfinite(values.data).all():
            raise ValueError("Prepared DEM has missing samples in the processing context")
    return output, float(nodata), manifest


def graph(scene, source, target, context, epsg, dem, nodata):
    west, south, east, north = context
    doc = ET.Element("graph", id="fresh-native-s1-geometric-" + scene)
    ET.SubElement(doc, "version").text = "1.0"

    def node(name, operator, source_id=None, source_alias="sourceProduct", **parameters):
        element = ET.SubElement(doc, "node", id=name)
        ET.SubElement(element, "operator").text = operator
        if source_id:
            ET.SubElement(ET.SubElement(element, "sources"), source_alias, refid=source_id)
        params = ET.SubElement(element, "parameters")
        for key, value in parameters.items():
            ET.SubElement(params, key).text = str(value).lower() if isinstance(value, bool) else str(value)

    node("read", "Read", file=source.resolve())
    node("subset", "Subset", "read", geoRegion=f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))", copyMetadata=True)
    node("noise", "ThermalNoiseRemoval", "subset", selectedPolarisations="VV,VH", removeThermalNoise=True, clipNegativeValues=False)
    node("calibrate", "Calibration", "noise", source_alias="source", selectedPolarisations="VV,VH", outputSigmaBand=True, outputBetaBand=False, outputImageScaleInDb=False, auxFile="Product Auxiliary File")
    node("terrain", "Terrain-Correction", "calibrate", source_alias="source", sourceBands="Sigma0_VV,Sigma0_VH", demName="External DEM", externalDEMFile=dem.resolve(), externalDEMNoDataValue=nodata, externalDEMApplyEGM=False, demResamplingMethod="BILINEAR_INTERPOLATION", imgResamplingMethod="BILINEAR_INTERPOLATION", pixelSpacingInMeter=10, mapProjection=CRS.from_epsg(epsg).to_wkt(version="WKT1_GDAL"), alignToStandardGrid=True, standardGridOriginX=0, standardGridOriginY=0, saveDEM=True, saveLocalIncidenceAngle=True, saveProjectedLocalIncidenceAngle=True, saveIncidenceAngleFromEllipsoid=True, saveLayoverShadowMask=True, nodataValueAtSea=True, applyRadiometricNormalization=False, saveSelectedSourceBand=True)
    node("write", "Write", "terrain", file=target.resolve(), formatName="BEAM-DIMAP")
    ET.indent(doc)
    return ET.ElementTree(doc)


def run(plan_path, index_path, dem_manifest, patch_manifest, patch_proof, output, *, execute=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; previous case evidence is preserved")
    plan_path, index_path, dem_manifest = [Path(p).resolve() for p in (plan_path, index_path, dem_manifest)]
    plan, bbox, context, projection, items = load_plan(plan_path)
    sources, index = acquired_sources(index_path, plan_path, plan, items)
    dem, nodata, conversion = prepared_dem(dem_manifest, context)
    overlay, patch_record = verified_reader_patch(Path(patch_manifest).resolve(), Path(patch_proof).resolve())
    # Verifies that this local portable runtime is prepared before creating any
    # checked-run artifact. There is no runtime or source download here.
    command([])
    output.mkdir(parents=True)
    report = {"schema": "akash-fresh-s1-processing-v1", "status": "preflight_only",
              "started_at": datetime.now(timezone.utc).isoformat(), "case": plan["case"],
              "bbox": list(bbox), "context_bbox": list(context), "projection": projection,
              "source_plan": record(plan_path), "source_index": record(index_path),
              "dem_conversion_manifest": record(dem_manifest), "dem": record(dem),
              "dem_conversion": conversion, "dem_reader_patch": patch_record, "scenes": {},
              "tool_sources": [record(Path(__file__)), record(ROOT / "src/akash_flood/snap_products.py"),
                               record(ROOT / "src/akash_flood/fresh_case.py"), record(ROOT / "scripts/process_s1_copdem.py"),
                               record(ROOT / "scripts/snap_gpt.py"), record(ROOT / "uv.lock")],
              "processing": {"thermal_noise_removed": True, "negative_noise_corrected_powers_clipped": False,
                  "calibration": "sigma0 linear power", "terrain_correction": "Range-Doppler geometric correction",
                  "dem": "Copernicus GLO-30 converted from EGM2008 orthometric to WGS84 ellipsoidal height",
                  "externalDEMApplyEGM": False, "SRTM_used": False,
                  "projection": f"EPSG:{projection['epsg']} WKT1_GDAL", "spacing_m": 10,
                  "standard_grid_pixel_center_origin": [0, 0],
                  "orbit": "Embedded source state vectors; no applied precise orbit",
                  "radiometric_terrain_flattening": False, "gpt_parallelism": 4, "cache": "512M", "heap": "6G",
                  "ocean_nodata_enabled": True},
              "classification_or_hazard_claim": False,
              "limitations": ["Output consistency is not independent absolute satellite geolocation or complete terrain-mask validation.",
                              "Provider-converted COGs in derived reader wrappers, not original ESA SAFE packages.",
                              "Local numerical DEM-reader correction; ESA validation or endorsement is not established.",
                              "No flood/debris model inference, reference labels or passability decisions are consumed by this stage."]}
    destination = output / "run-metadata.json"

    def save():
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for scene in ("before", "after"):
        source = sources[scene]
        xml = output / (scene + "-terrain.xml")
        graph(scene, source["manifest"], output / (scene + "-terrain.dim"), context,
              projection["epsg"], dem, nodata).write(xml, encoding="utf-8", xml_declaration=True)
        entry = {"source_item_id": source["entry"]["source_item_id"],
                 "source_datetime": source["entry"]["datetime"], "source_manifest": record(source["manifest"]),
                 "original_provider_manifest": record(source["original_manifest"]),
                 "source_wrapper_provenance": record(index_path), "graph": record(xml), "status": "not_executed"}
        report["scenes"][scene] = entry
        save()
        if not execute:
            continue
        argv = command([str(xml), "-q", "4", "-c", "512M"])
        cp_index = argv.index("-cp") + 1
        argv[cp_index] = str(overlay.resolve()) + ":" + argv[cp_index]
        entry.update(status="running", started_at=datetime.now(timezone.utc).isoformat())
        report["status"] = "running"
        save()
        log = output / (scene + "-terrain.log")
        start = time.perf_counter()
        with log.open("w") as stream:
            result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
        entry.update(exit_code=result.returncode, wall_seconds=time.perf_counter()-start, log=record(log))
        save()
        text = log.read_text(errors="replace")
        if result.returncode != 0 or "done." not in text or "Cannot create layover/shadow" in text:
            entry["status"] = report["status"] = "failed"
            save()
            raise RuntimeError("SNAP failed or could not generate its terrain mask; partial outputs and log retained")
        product = read_snap_product(output / (scene + "-terrain.dim"), projection["epsg"])
        pair = plan["sentinel1"]["selected_pair"]
        if (int(product.attributes["REL_ORBIT"]) != pair["relative_orbit"]
                or product.attributes["PASS"] != pair["orbit_state"].upper()):
            entry["status"] = report["status"] = "failed"
            save()
            raise ValueError("Processed native orbit/pass differs from the selected catalogue pair")
        entry["validation"] = product_summary(product)
        del product
        entry["artifacts"] = [record(p) for p in sorted(output.glob(scene + "-terrain.*")) if p.is_file()] + [
            record(p) for p in sorted((output / (scene + "-terrain.data")).rglob("*")) if p.is_file()]
        entry.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
        save()
        print(scene, "completed", entry["validation"]["positive_geometry_supported_pixels"], "positive geometry-supported pixels", flush=True)
    if execute:
        grids = []
        for scene in ("before", "after"):
            summary = report["scenes"][scene]["validation"]
            height, width = summary["shape"]
            grids.append((rasterio.crs.CRS.from_string(summary["crs"]), Affine(*summary["transform"]), width, height))
        try:
            before_window, after_window, common = integer_overlap(*grids)
        except ValueError:
            report["status"] = "failed"
            save()
            raise
        report["common_grid"] = {"crs": common[0].to_string(), "transform": list(common[1])[:6],
                                 "shape": [common[3], common[2]], "resampling_used": False,
                                 "before_window": [before_window.col_off, before_window.row_off, before_window.width, before_window.height],
                                 "after_window": [after_window.col_off, after_window.row_off, after_window.width, after_window.height],
                                 "validity_scope": "Shared native grid cells; observation and geometry validity still require both-date masks"}
        report.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat())
    else:
        report.update(preflight_completed_at=datetime.now(timezone.utc).isoformat(),
                      execution_required="Use a separate fresh output directory with --execute; this artifact is not a processed raster.")
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--dem-manifest", type=Path, required=True)
    parser.add_argument("--dem-reader-patch", type=Path, required=True)
    parser.add_argument("--dem-reader-proof", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Run SNAP; default writes checked graph/metadata only")
    args = parser.parse_args()
    run(args.plan, args.source_index, args.dem_manifest, args.dem_reader_patch, args.dem_reader_proof, args.out, execute=args.execute)


if __name__ == "__main__":
    main()
