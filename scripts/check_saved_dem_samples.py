"""Compare SNAP's saved elevation band with the DEM at nominal output centers.

Fixed 25-by-25 target samples; independent raster bilinear interpolation. This
checks DEM sampling/datum consistency, not absolute satellite geolocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.windows import Window

from prepare_snap_external_dem import ROOT, record, digest


def bilinear(dataset, lon, lat):
    corner_col, corner_row = (~dataset.transform) * (lon, lat)
    col, row = corner_col - .5, corner_row - .5
    c0, r0 = int(np.floor(col)), int(np.floor(row))
    if not (0 <= c0 < dataset.width - 1 and 0 <= r0 < dataset.height - 1):
        raise ValueError("Nominal target center exceeds source DEM interpolation support")
    window = Window(c0, r0, 2, 2)
    values = dataset.read(1, window=window).astype("float64")
    if not (dataset.read_masks(1, window=window) > 0).all() or not np.isfinite(values).all():
        raise ValueError("Independent interpolation encountered source nodata")
    dx, dy = col - c0, row - r0
    return float((1 - dy) * ((1 - dx) * values[0, 0] + dx * values[0, 1])
                 + dy * ((1 - dx) * values[1, 0] + dx * values[1, 1]))


def compare(dem_manifest, folder, out):
    dem_manifest, folder, out = map(Path, (dem_manifest, folder, out))
    if out.exists():
        raise FileExistsError("Preserve earlier saved DEM diagnostics")
    conversion = json.loads(dem_manifest.read_text())
    dem_path = Path(conversion["output"]["path"])
    if not dem_path.is_absolute():
        dem_path = dem_manifest.parent / dem_path
    if conversion["output"]["vertical_datum"] != "WGS84 ellipsoidal height" or digest(dem_path.read_bytes()) != conversion["output"]["sha256"]:
        raise ValueError("Require checksum-pinned explicitly ellipsoidal source DEM")
    run_path = folder / "run-metadata.json"
    run = json.loads(run_path.read_text())
    if "completed_at" not in run:
        raise ValueError("Terrain processing is incomplete")
    if run["dem"]["sha256"] != conversion["output"]["sha256"]:
        raise ValueError("Processing DEM differs from independently sampled DEM")
    scenes = {}
    with rasterio.open(dem_path) as dem:
        if dem.crs != rasterio.crs.CRS.from_epsg(4326) or dem.count != 1:
            raise ValueError("Expected one WGS84 geographic DEM band")
        for scene in ("before", "after"):
            path = folder / (scene + "-terrain.data/elevation.img")
            with rasterio.open(path) as target:
                if target.crs != rasterio.crs.CRS.from_epsg(32645) or (target.transform.a, target.transform.e, target.transform.b, target.transform.d) != (10, -10, 0, 0):
                    raise ValueError("Expected north-up 10 m UTM target")
                transformer = Transformer.from_crs(target.crs, dem.crs, always_xy=True)
                samples = []
                for row in np.linspace(0, target.height - 1, 25).round().astype(int):
                    for col in np.linspace(0, target.width - 1, 25).round().astype(int):
                        actual = float(target.read(1, window=Window(int(col), int(row), 1, 1))[0, 0])
                        if not np.isfinite(actual) or actual <= 0:
                            continue  # Rasuwa land is positive; absent output is excluded.
                        east, north = target.xy(int(row), int(col))
                        lon, lat = transformer.transform(east, north)
                        expected = bilinear(dem, lon, lat)
                        samples.append({"row": int(row), "column": int(col), "longitude": lon,
                                        "latitude": lat, "snap_m": actual,
                                        "independent_nominal_center_bilinear_m": expected,
                                        "error_m": actual - expected})
                if not samples:
                    raise ValueError("No positive saved elevation samples")
                errors = np.array([s["error_m"] for s in samples])
                scenes[scene] = {"source": record(path), "shape": list(target.shape),
                                 "transform": list(target.transform), "sample_count": len(samples),
                                 "max_abs_error_m": float(abs(errors).max()),
                                 "median_abs_error_m": float(np.median(abs(errors))),
                                 "error_percentiles_m": np.percentile(errors, [0, 1, 50, 99, 100]).tolist(),
                                 "samples": samples}
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "status": "nominal-center DEM samples agree" if all(v["max_abs_error_m"] < .001 for v in scenes.values()) else "nominal-center DEM samples disagree",
              "tolerance_m": .001, "sample_protocol": "25-by-25 equally spaced target indices; positive finite saved land elevations",
              "interpolation": "Independent bilinear source DEM sampling at rasterio/PROJ-transformed nominal target pixel centers; no fitted spatial shift",
              "dem": record(dem_path), "dem_manifest": record(dem_manifest),
              "processing_record": record(run_path), "checker": record(Path(__file__)),
              "reference_labels_loaded": False, "scenes": scenes,
              "limitations": ["Checks terrain-height sampling against declared source only",
                              "Does not establish absolute satellite geolocation or hazard-label accuracy",
                              "Fixed positive sample selection is specific to mountainous Rasuwa land"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem-manifest", type=Path, required=True)
    parser.add_argument("--terrain", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.dem_manifest, args.terrain, args.out)
    print(json.dumps({"status": result["status"], "scenes": {k: {n: v[n] for n in ["sample_count", "max_abs_error_m", "median_abs_error_m"]} for k, v in result["scenes"].items()}}, indent=2))
    if result["status"] != "nominal-center DEM samples agree":
        raise SystemExit(1)
