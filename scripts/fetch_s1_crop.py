"""Read/calibrate a small real Sentinel-1 GRD crop, with a hard public-data budget.

Default AOI is a small Rasuwa engineering window, not an event impact footprint.
Run: uv run python scripts/fetch_s1_crop.py --output outputs/s1-raw-crop-proof
No credentials, full scenes, paid processing, terrain correction or flood claims.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import rasterio
from rasterio.transform import GCPTransformer

from akash_flood.s1_crop import (BoundedHTTP, SingleRasterOpener, calibrate_sigma0,
                               gcp_records, interpolate_sigma_lut, native_window_for_bbox,
                               parse_calibration_xml, shifted_gcps, unsigned_s3_https)


ROOT = Path(__file__).resolve().parents[1]
ITEMS = {
    "before": "S1D_IW_GRDH_1SDV_20260816T122141_20260816T122206_004151_007980",
    "after": "S1D_IW_GRDH_1SDV_20260828T122141_20260828T122206_004326_007FA4",
}
CALIBRATION_SOURCES = [
    "https://sentinels.copernicus.eu/documents/247904/685163/S1-Radiometric-Calibration-V1.0.pdf",
    "https://github.com/senbox-org/s1tbx/blob/master/s1tbx-op-calibration-ui/src/main/resources/org/esa/s1tbx/calibration/docs/operators/CalibrationOp.html",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(array, mask):
    values = np.asarray(array)[np.asarray(mask)]
    return {"count": int(len(values)), "min": float(values.min()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "max": float(values.max()) if len(values) else None}


def write_native(path, data, gcps, gcp_crs, nodata, description):
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0],
                       count=1, dtype=data.dtype, gcps=gcps, crs=gcp_crs, nodata=nodata,
                       compress="deflate") as destination:
        destination.write(data, 1)
        destination.set_band_description(1, description)
        destination.update_tags(GEOLOCATION="shifted original source GCPs; native measurement grid",
                                NOT_DONE="thermal_noise_removal,terrain_correction,coregistration,shadow_layover_mask,flood_mapping")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/s1-raw-crop-proof")
    parser.add_argument("--bbox", type=float, nargs=4, default=[85.34, 28.18, 85.35, 28.19])
    parser.add_argument("--max-download-bytes", type=int, default=50_000_000)
    args = parser.parse_args()
    if args.max_download_bytes > 50_000_000:
        raise SystemExit("This proof is capped at 50,000,000 bytes")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Output is nonempty; preserve the evidence and use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    client = BoundedHTTP(args.max_download_bytes)
    results = []
    provenance = {"status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "requested_bbox": args.bbox, "event_date": "2026-08-26", "relative_orbit": 85,
                  "pass": "ascending", "platform": "sentinel-1d", "source": "anonymous EarthSearch Sentinel-1 GRD S3",
                  "max_download_bytes": args.max_download_bytes, "gpu_used": False, "account_used": False,
                  "paid_processing_used": False, "calibration_sources": CALIBRATION_SOURCES,
                  "formula": "sigma0_power = float64(DN)^2 / bilinearly_interpolated_sigmaNought_LUT^2; dB = 10*log10(power)",
                  "interpolation_coordinates": "original zero-based measurement rows and columns, before crop-offset subtraction",
                  "scope": "actual decoded raw amplitude and radiometric calibration proof only",
                  "not_established": ["thermal noise removal", "terrain correction", "radiometric terrain flattening",
                                      "co-registration", "shadow/layover validity", "flood/debris accuracy", "road passability"],
                  "geolocation_limitation": "Source GCP polynomial approximation; no DEM terrain correction. STAC proj affine is not used.",
                  "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "rasterio": rasterio.__version__,
                              "machine": platform.machine()},
                  "code_sha256": {"scripts/fetch_s1_crop.py": digest(Path(__file__)),
                                  "src/akash_flood/s1_crop.py": digest(ROOT / "src/akash_flood/s1_crop.py")}}
    try:
        for scene, item_id in ITEMS.items():
            item_url = "https://earth-search.aws.element84.com/v1/collections/sentinel-1-grd/items/" + item_id
            item_bytes = client.get(item_url)
            item = json.loads(item_bytes)
            properties = item["properties"]
            if (item["id"] != item_id or properties.get("sat:relative_orbit") != 85
                    or properties.get("sat:orbit_state") != "ascending" or properties.get("platform") != "sentinel-1d"):
                raise ValueError("Catalogue source does not match the declared acquisition")
            (args.output / f"{scene}.source-stac.json").write_bytes(item_bytes)
            for polarization in ["vv", "vh"]:
                prefix = f"{scene}-{polarization}"
                asset_url = unsigned_s3_https(item["assets"][polarization]["href"])
                calibration_url = unsigned_s3_https(item["assets"]["schema-calibration-" + polarization]["href"])
                xml = client.get(calibration_url)
                (args.output / f"{prefix}.calibration.xml").write_bytes(xml)
                lut = parse_calibration_xml(xml)
                if lut.header.get("missionId") != "S1D" or lut.header.get("polarisation", "").lower() != polarization:
                    raise ValueError("Calibration XML mission/polarization differs from the measurement")
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_PAM_ENABLED="NO"):
                    with rasterio.open(asset_url, opener=SingleRasterOpener(asset_url, client)) as source:
                        gcps, gcp_crs = source.gcps
                        if source.crs is not None or not gcps or gcp_crs.to_epsg() != 4326 or source.dtypes != ("uint16",):
                            raise ValueError("Expected unprojected uint16 amplitude with source EPSG:4326 GCPs")
                        window = native_window_for_bbox(gcps, args.bbox, source.width, source.height)
                        raw = source.read(1, window=window)
                        source_mask = source.read_masks(1, window=window)
                        rows = np.arange(int(window.row_off), int(window.row_off + window.height))
                        columns = np.arange(int(window.col_off), int(window.col_off + window.width))
                        sigma_lut = interpolate_sigma_lut(lut, rows, columns)
                        power, db, numeric_valid = calibrate_sigma0(raw, sigma_lut, source_mask)
                        original_rows, original_cols = np.meshgrid(rows, columns, indexing="ij")
                        with GCPTransformer(gcps) as transformer:
                            longitude, latitude = transformer.xy(original_rows.ravel(), original_cols.ravel(), offset="center")
                        longitude = np.asarray(longitude).reshape(raw.shape)
                        latitude = np.asarray(latitude).reshape(raw.shape)
                        west, south, east, north = args.bbox
                        inside = (longitude >= west) & (longitude <= east) & (latitude >= south) & (latitude <= north)
                        shifted = shifted_gcps(gcps, window)
                        write_native(args.output / f"{prefix}.raw-amplitude.tif", raw, shifted, gcp_crs, source.nodata,
                                     "Original uint16 Sentinel-1 GRD digital-number amplitude")
                        write_native(args.output / f"{prefix}.sigma0-power.tif", power.astype(np.float32), shifted, gcp_crs, np.nan,
                                     "Radiometrically calibrated sigma0 linear power; not terrain corrected")
                        write_native(args.output / f"{prefix}.sigma0-db.tif", db.astype(np.float32), shifted, gcp_crs, np.nan,
                                     "10 log10 sigma0 power in dB; not terrain corrected")
                        np.savez_compressed(args.output / f"{prefix}.native-coordinates-and-lut.npz", original_rows=rows,
                                            original_columns=columns, sigma_nought_lut=sigma_lut,
                                            approximate_longitude=longitude, approximate_latitude=latitude,
                                            inside_requested_bbox=inside, numeric_valid=numeric_valid)
                        geolocation = {"original_gcps": gcp_records(gcps), "shifted_crop_gcps": gcp_records(shifted),
                                       "gcp_crs": gcp_crs.to_string(), "source_dataset_crs": None,
                                       "source_transform": list(source.transform), "source_transform_is_geolocation": False,
                                       "window": {"row_off": int(window.row_off), "col_off": int(window.col_off),
                                                  "height": int(window.height), "width": int(window.width)},
                                       "stac_projection_used_for_window": False, "source_shape": [source.height, source.width],
                                       "source_nodata": source.nodata, "boundary_samples_per_side": 33, "native_halo_pixels": 8}
                        (args.output / f"{prefix}.geolocation.json").write_text(json.dumps(geolocation, indent=2) + "\n")
                        results.append({"scene": scene, "polarization": polarization, "item_id": item_id,
                                        "datetime": properties["datetime"], "canonical_unsigned_asset_url": asset_url,
                                        "canonical_unsigned_calibration_url": calibration_url,
                                        "window": geolocation["window"], "source_gcp_count": len(gcps),
                                        "raw_crop_pixels": int(raw.size), "inside_requested_bbox_pixels": int(inside.sum()),
                                        "source_mask_nonzero_pixels": int((source_mask != 0).sum()),
                                        "raw_nonzero_pixels": int((raw > 0).sum()), "numeric_valid_pixels": int(numeric_valid.sum()),
                                        "inside_bbox_numeric_valid_pixels": int((inside & numeric_valid).sum()),
                                        "uint16_saturated_pixels": int((raw == 65535).sum()),
                                        "raw_amplitude": stats(raw, numeric_valid), "sigma0_power": stats(power, numeric_valid),
                                        "sigma0_db": stats(db, numeric_valid), "calibration_vector_count": len(lut.lines),
                                        "numeric_valid_does_not_mean_hazard_valid": True})
        provenance["status"] = "complete"
    except Exception as error:
        provenance["status"] = "failed"
        provenance["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        provenance["results"] = results
        provenance["network_bytes_read"] = client.bytes_read
        provenance["http_requests"] = client.requests
        provenance["wall_seconds"] = time.perf_counter() - wall_start
        provenance["process_cpu_seconds"] = time.process_time() - cpu_start
        provenance["files"] = [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
                               for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "provenance.json"]
        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: provenance[k] for k in ["status", "network_bytes_read", "wall_seconds", "process_cpu_seconds", "results"]}, indent=2))


if __name__ == "__main__":
    main()
