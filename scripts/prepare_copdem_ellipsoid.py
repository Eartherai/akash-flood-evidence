"""Convert allowed Copernicus GLO-30 EGM2008 heights to ellipsoidal heights.

The geographic sample grid is preserved. An explicit PROJ +multiplier=1 applies
h = H_EGM2008 + N_EGM2008, with mandatory local grids and networking disabled.
The resulting external SNAP DEM must use externalDEMApplyEGM=false.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyproj
from pyproj import Transformer, network
import rasterio
from rasterio.windows import Window
import requests

ROOT = Path(__file__).resolve().parents[1]
GRID_URL = "https://cdn.proj.org/us_nga_egm08_25.tif"
# Pinned to the official PROJ CDN catalog inspected on 2026-10-06.
GRID_SHA256 = "4191d471eefebf24091b56dbc604353cb3b8cf8cc70e448bb9ae56a272bef17a"
GRID_BYTES = 80585622
DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N28_00_E085_00_DEM/Copernicus_DSM_COG_10_N28_00_E085_00_DEM.tif"
LICENSE_URL = "https://dataspace.copernicus.eu/sites/default/files/media/files/2025-06/copernicus_contributing_mission_data_access_v2_cop_dem_licenses.pdf"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path):
    p = Path(path)
    return {"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha256(p)}


class DownloadBudget:
    def __init__(self, maximum=200_000_000):
        self.maximum = maximum
        self.used = 0

    def download(self, url, target, per_file_cap, expected_sha=None, expected_bytes=None):
        target = Path(target)
        if target.exists():
            raise FileExistsError(target)
        partial = target.with_suffix(target.suffix + ".part")
        start = time.perf_counter()
        print("Downloading", target.name, flush=True)
        with requests.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            length = response.headers.get("Content-Length")
            if length and int(length) > min(per_file_cap, self.maximum - self.used):
                raise ValueError("Advertised download exceeds budget")
            size = 0
            with partial.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    self.used += len(chunk)
                    size += len(chunk)
                    if size > per_file_cap or self.used > self.maximum:
                        raise ValueError("Download budget exceeded; partial file retained")
                    stream.write(chunk)
            if length and size != int(length):
                raise ValueError("Truncated source download")
            if expected_bytes is not None and size != expected_bytes:
                raise ValueError("Downloaded size differs from official catalog")
            digest = sha256(partial)
            if expected_sha is not None and digest != expected_sha:
                raise ValueError("Downloaded SHA256 differs from official catalog")
            partial.replace(target)
            return {**file_record(target), "url": url, "resolved_url": response.url, "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified"), "wall_seconds": time.perf_counter() - start}


def mandatory_geoid_transform(grid_path):
    grid_path = Path(grid_path).resolve()
    if not grid_path.is_file():
        raise FileNotFoundError("Mandatory local EGM2008 grid is absent")
    network.set_network_enabled(False)
    # PROJ handles quoted paths, including the workspace's spaces. No @optional
    # grid, null grid, ballpark CRS transformation, or automatic fallback exists.
    operation = (
        "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad "
        f"+step +proj=vgridshift +grids={json.dumps(str(grid_path))} +multiplier=1 "
        "+step +proj=unitconvert +xy_in=rad +xy_out=deg"
    )
    return Transformer.from_pipeline(operation), operation


def bilinear_geoid(dataset, longitude, latitude):
    """Independent raster interpolation of geoid samples at pixel centers."""
    lon, lat = np.broadcast_arrays(np.asarray(longitude, dtype=np.float64), np.asarray(latitude, dtype=np.float64))
    inverse = ~dataset.transform
    col = inverse.a * lon + inverse.b * lat + inverse.c - .5
    row = inverse.d * lon + inverse.e * lat + inverse.f - .5
    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    if np.any(c0 < 0) or np.any(r0 < 0) or np.any(c0 + 1 >= dataset.width) or np.any(r0 + 1 >= dataset.height):
        raise ValueError("Independent geoid samples exceed mandatory grid coverage")
    left, right = int(c0.min()), int(c0.max()) + 2
    top, bottom = int(r0.min()), int(r0.max()) + 2
    samples = dataset.read(1, window=Window(left, top, right - left, bottom - top), masked=True)
    dx, dy = col - c0, row - r0
    x, y = c0 - left, r0 - top
    a, b, c, d = samples[y, x], samples[y, x + 1], samples[y + 1, x], samples[y + 1, x + 1]
    if any(np.ma.getmaskarray(v).any() or not np.isfinite(np.ma.getdata(v)).all() for v in [a, b, c, d]):
        raise ValueError("Mandatory geoid interpolation encountered nodata")
    return (1 - dy) * ((1 - dx) * a + dx * b) + dy * ((1 - dx) * c + dx * d)


def convert_dem(source, grid, target, source_vertical_datum="EGM2008"):
    if source_vertical_datum != "EGM2008":
        raise ValueError("Require an explicitly verified EGM2008 orthometric input datum")
    if Path(target).exists():
        raise FileExistsError("Refuse to overwrite a converted DEM")
    transform, operation = mandatory_geoid_transform(grid)
    sample_rows = []
    shift_min, shift_max = np.inf, -np.inf
    valid_pixels = 0
    nodata_pixels = 0
    with rasterio.open(source) as src, rasterio.open(grid) as geoid:
        if src.crs != rasterio.crs.CRS.from_epsg(4326) or src.count != 1 or src.transform.b != 0 or src.transform.d != 0:
            raise ValueError("Require one-band north-up geographic EPSG:4326 DEM")
        if geoid.crs not in (rasterio.crs.CRS.from_epsg(4326), rasterio.crs.CRS.from_epsg(4979)) or geoid.count != 1:
            raise ValueError("Require one-band geographic EGM2008 grid")
        if "EGM2008" not in str(geoid.tags()) + str(geoid.tags(1)) and "egm08" not in str(Path(grid).name):
            raise ValueError("Geoid metadata does not identify EGM2008")
        profile = {"driver": "GTiff", "width": src.width, "height": src.height, "count": 1, "dtype": "float32", "crs": src.crs, "transform": src.transform, "nodata": -32767.0, "tiled": False}
        with rasterio.open(target, "w", **profile) as dst:
            dst.update_tags(AREA_OR_POINT=src.tags().get("AREA_OR_POINT", "Area"), VERTICAL_DATUM="WGS84 ellipsoidal height", SOURCE_VERTICAL_DATUM="EGM2008 (EPSG:3855)", TRANSFORMATION="h = H_EGM2008 + N_EGM2008", SNAP_EXTERNAL_DEM_APPLY_EGM="false")
            dst.set_band_unit(1, "metre")
            for row0 in range(0, src.height, 256):
                height = min(256, src.height - row0)
                window = Window(0, row0, src.width, height)
                orthometric = src.read(1, window=window).astype(np.float64)
                valid = (src.read_masks(1, window=window) > 0) & np.isfinite(orthometric)
                rows, cols = np.meshgrid(np.arange(row0, row0 + height), np.arange(src.width), indexing="ij")
                lon = src.transform.c + (cols + .5) * src.transform.a
                lat = src.transform.f + (rows + .5) * src.transform.e
                _, _, ellipsoid = transform.transform(lon, lat, orthometric, errcheck=True)
                ellipsoid = np.asarray(ellipsoid, dtype=np.float64)
                if not np.isfinite(ellipsoid[valid]).all():
                    raise ValueError("Mandatory EGM2008 conversion produced missing/nonfinite elevations")
                shift = ellipsoid - orthometric
                if valid.any():
                    shift_min = min(shift_min, float(shift[valid].min()))
                    shift_max = max(shift_max, float(shift[valid].max()))
                valid_pixels += int(valid.sum())
                nodata_pixels += int((~valid).sum())
                out = np.full(orthometric.shape, profile["nodata"], dtype=np.float32)
                out[valid] = ellipsoid[valid]
                dst.write(out, 1, window=window)
            if not valid_pixels:
                raise ValueError("Source DEM has no valid pixels")
        # Fixed deterministic samples across the full source; include only valid
        # input samples and verify PROJ independently of its interpolation code.
        rr = np.linspace(0, src.height - 1, 17).round().astype(int)
        cc = np.linspace(0, src.width - 1, 17).round().astype(int)
        with rasterio.open(target) as dst:
            for row in rr:
                for col in cc:
                    window = Window(int(col), int(row), 1, 1)
                    if src.read_masks(1, window=window)[0, 0] == 0:
                        continue
                    h0 = float(src.read(1, window=window)[0, 0])
                    if not np.isfinite(h0):
                        continue
                    lon, lat = src.xy(int(row), int(col))
                    n = float(bilinear_geoid(geoid, lon, lat))
                    _, _, proj_h = transform.transform(lon, lat, h0, errcheck=True)
                    saved_h = float(dst.read(1, window=window)[0, 0])
                    if abs(proj_h - (h0 + n)) > 2e-5:
                        raise ValueError("PROJ conversion differs from independent H+N interpolation")
                    if abs(saved_h - (h0 + n)) > 5e-4:
                        raise ValueError("Saved float32 elevation differs from independent H+N interpolation")
                    sample_rows.append({"row": int(row), "column": int(col), "longitude": lon, "latitude": lat, "orthometric_m": h0, "independent_geoid_undulation_m": n, "proj_ellipsoid_m": proj_h, "saved_float32_ellipsoid_m": saved_h})
            if dst.compression is not None or dst.crs != src.crs or dst.transform != src.transform or dst.shape != src.shape:
                raise ValueError("External DEM altered geographic grid or is compressed")
            nodata_mask = dst.read_masks(1) == 0
            if int(nodata_mask.sum()) != nodata_pixels:
                raise ValueError("Saved nodata coverage changed")
    if not sample_rows:
        raise ValueError("No independent datum verification samples")
    return {"operation": operation, "formula": "h_WGS84 = H_EGM2008 + N_EGM2008", "multiplier": 1, "network_enabled_for_proj": network.is_network_enabled(), "mandatory_local_grid": True, "fallback_allowed": False, "shift_range_m": [shift_min, shift_max], "valid_pixels": valid_pixels, "nodata_pixels": nodata_pixels, "independent_sample_count": len(sample_rows), "max_proj_vs_independent_error_m": max(abs(s["proj_ellipsoid_m"] - s["orthometric_m"] - s["independent_geoid_undulation_m"]) for s in sample_rows), "max_float32_vs_independent_error_m": max(abs(s["saved_float32_ellipsoid_m"] - s["orthometric_m"] - s["independent_geoid_undulation_m"]) for s in sample_rows), "samples": sample_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/copdem-ellipsoid")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("Select a fresh conversion directory")
    out.mkdir(parents=True)
    budget = DownloadBudget()
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "code": file_record(Path(__file__)), "pyproj_version": pyproj.__version__, "proj_version": pyproj.proj_version_str, "dem_source": {"url": DEM_URL, "vertical_datum": "EGM2008 (EPSG:3855)", "instance": "Copernicus GLO-30 Public", "horizontal_crs": "EPSG:4326", "source_scope": "full public provider COG tile, not original ESA delivery packaging"}, "downloads": {}, "download_budget_bytes": budget.maximum, "snap_settings": {"externalDEMApplyEGM": False, "externalDEMNoDataValue": -32767.0, "demName": "External DEM"}}
    def save():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    assets = [
        ("proj_catalog", "https://cdn.proj.org/files.geojson", "proj-grid-catalog.geojson", 10_000_000),
        ("geoid_license_readme", "https://cdn.proj.org/us_nga_README.txt", "us_nga_README.txt", 100_000),
        ("dem_readme", "https://copernicus-dem-30m.s3.amazonaws.com/readme.html", "copernicus-public-readme.html", 1_000_000),
        ("dem_license", LICENSE_URL, "copernicus-dem-licenses.pdf", 5_000_000),
    ]
    for key, url, name, cap in assets:
        manifest["downloads"][key] = budget.download(url, out / name, cap)
        save()
    features = json.loads((out / "proj-grid-catalog.geojson").read_text())["features"]
    grid_metadata = next(f["properties"] for f in features if f["properties"].get("name") == "us_nga_egm08_25.tif")
    if grid_metadata["sha256sum"] != GRID_SHA256 or grid_metadata["file_size"] != GRID_BYTES:
        raise ValueError("Official EGM2008 catalog changed; require explicit review")
    manifest["geoid_source"] = {**grid_metadata, "license": "Public Domain; official us_nga_README.txt", "purpose": "Geodetic datum conversion auxiliary, not a hazard/evaluation input"}
    manifest["downloads"]["dem_orthometric"] = budget.download(DEM_URL, out / "copdem-n28-e085-orthometric.tif", 65_000_000)
    save()
    manifest["downloads"]["geoid_grid"] = budget.download(GRID_URL, out / "us_nga_egm08_25.tif", 85_000_000, GRID_SHA256, GRID_BYTES)
    save()
    output = out / "copdem-n28-e085-ellipsoid.tif"
    diagnostics = convert_dem(out / "copdem-n28-e085-orthometric.tif", out / "us_nga_egm08_25.tif", output)
    diagnostic_path = out / "conversion-diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    manifest["diagnostics"] = file_record(diagnostic_path)
    manifest["output"] = {**file_record(output), "vertical_datum": "WGS84 ellipsoidal height", "horizontal_crs": "EPSG:4326", "dtype": "float32", "compression": None, "nodata": -32767.0, "formula": diagnostics["formula"]}
    manifest["network_downloaded_bytes"] = budget.used
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["limitations"] = ["2.5 arc-minute EGM2008 geoid grid bilinear interpolation; no claim of a finer geoid", "Geodetic conversion changes height datum but does not remove DSM vegetation/building heights", "Source and corrected DEM grids have the same pixel centers; no spatial resampling", "Source grid is the provider COG, whose original eastern/southern duplicate edges were removed"]
    save()
    print(json.dumps({"output": manifest["output"], "shift_range_m": diagnostics["shift_range_m"], "verification_samples": diagnostics["independent_sample_count"], "proj_error_m": diagnostics["max_proj_vs_independent_error_m"], "download_bytes": budget.used}, indent=2))


if __name__ == "__main__":
    main()
