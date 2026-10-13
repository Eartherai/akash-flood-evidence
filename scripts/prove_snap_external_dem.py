"""Check the actual SNAP GeoTIFF reader and external DEM model in isolated JVMs.

Run with: uv run --with jpype1==1.6.0 python scripts/prove_snap_external_dem.py
Uses synthetic analytic fixtures only; no hazard reference labels are loaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin

from prepare_snap_external_dem import ROOT, VENDOR_CLASS_SHA, VENDOR_JAR_SHA, record, digest

RUNTIME = ROOT / "data/snap-runtime/portable"


def expected_value(values, row, col):
    """Independent analytic-array bilinear interpolation, no SNAP calls."""
    r0, c0 = int(np.floor(row)), int(np.floor(col))
    dy, dx = row - r0, col - c0
    return float((1 - dy) * ((1 - dx) * values[r0, c0] + dx * values[r0, c0 + 1])
                 + dy * ((1 - dx) * values[r0 + 1, c0] + dx * values[r0 + 1, c0 + 1]))


def make_fixtures(folder):
    folder = Path(folder); folder.mkdir()
    rows, cols = np.meshgrid(np.arange(7), np.arange(7), indexing="ij")
    for shape, values in [("plane", 1000 + rows * 100 + cols * 10),
                          ("nonlinear", 1000 + rows ** 2 * 23 + cols ** 2 * 11 + rows * cols * 7)]:
        for tag in ("Point", "Area"):
            path = folder / (shape + "-" + tag.lower() + ".tif")
            with rasterio.open(path, "w", driver="GTiff", width=7, height=7,
                               count=1, dtype="float32", crs="EPSG:4326",
                               transform=from_origin(85, 28.1, .001, .001), nodata=-32767) as ds:
                ds.write(values.astype("float32"), 1)
                ds.update_tags(AREA_OR_POINT=tag, VERTICAL_DATUM="WGS84 ellipsoidal height")


def worker(mode, fixtures, overlay_manifest, json_out):
    import jpype
    manifest = json.loads(Path(overlay_manifest).read_text())
    overlay = Path(manifest["overlay_jar"]["path"])
    if digest(overlay.read_bytes()) != manifest["overlay_jar"]["sha256"]:
        raise ValueError("Overlay differs from recorded correction")
    vendor = Path(manifest["vendor_jar"]["path"])
    if digest(vendor.read_bytes()) != VENDOR_JAR_SHA or manifest["vendor_class_sha256"] != VENDOR_CLASS_SHA:
        raise ValueError("Original vendor runtime changed")
    classpath = [str(p) for p in sorted(RUNTIME.rglob("*.jar"))]
    if mode == "corrected":
        classpath.insert(0, str(overlay))
    jpype.startJVM(str(RUNTIME / "jre/Contents/Home/lib/server/libjvm.dylib"), "-Xmx1G",
                   "-Djava.awt.headless=true", "-Dsnap.home=" + str(RUNTIME),
                   "-Dsnap.userdir=" + str(RUNTIME.parent / "user-cache"), classpath=classpath)
    File = jpype.JClass("java.io.File")
    CommonReaders = jpype.JClass("org.esa.snap.engine_utilities.gpf.CommonReaders")
    PixelPos = jpype.JClass("org.esa.snap.core.datamodel.PixelPos")
    GeoPos = jpype.JClass("org.esa.snap.core.datamodel.GeoPos")
    Model = jpype.JClass("org.esa.snap.dem.dataio.FileElevationModel")
    loaded_from = str(Model.class_.getProtectionDomain().getCodeSource().getLocation())
    if mode == "corrected" and not loaded_from.endswith("/external-dem-center-overlay.jar"):
        raise ValueError("Corrected class was not loaded from overlay")
    if mode == "vendor" and "overlay" in loaded_from:
        raise ValueError("Vendor proof accidentally loaded overlay")
    results = []
    # Predeclared centers and noninteger locations away from image boundaries.
    points = [(float(row), float(col)) for row in range(1, 6) for col in range(1, 6)]
    points += [(1.25, 2.75), (2.75, 1.25), (3.125, 4.375), (4.375, 3.125)]
    for path in sorted(Path(fixtures).glob("*.tif")):
        product = CommonReaders.readProduct(File(str(path.resolve())))
        geocoding = product.getSceneGeoCoding()
        model = Model(File(str(path.resolve())), "BILINEAR_INTERPOLATION", jpype.JDouble(-32767))
        model.applyEarthGravitionalModel(False)
        with rasterio.open(path) as ds:
            values = ds.read(1).astype("float64")
            for row, col in points:
                lon, lat = ds.transform * (col + .5, row + .5)
                position = geocoding.getPixelPos(GeoPos(lat, lon), None)
                geo = geocoding.getGeoPos(PixelPos(col + .5, row + .5), None)
                if max(abs(position.x - col - .5), abs(position.y - row - .5)) > 1e-8:
                    raise ValueError("Actual reader does not return expected center coordinates")
                if max(abs(geo.lon - lon), abs(geo.lat - lat)) > 1e-10:
                    raise ValueError("Actual reader changed nominal geographic centers")
                expected = expected_value(values, row, col)
                observed = float(model.getElevation(GeoPos(lat, lon)))
                results.append({"fixture": path.name, "row": row, "column": col,
                                "longitude": lon, "latitude": lat,
                                "reader_pixel_x": float(position.x), "reader_pixel_y": float(position.y),
                                "expected_bilinear_m": expected, "actual_model_m": observed,
                                "error_m": observed - expected})
        model.dispose(); product.dispose()
    report = {"mode": mode, "model_loaded_from": loaded_from,
              "reader_class": "org.esa.snap.dataio.geotiff.GeoTiffProductReader",
              "geocoding_class": "org.esa.snap.core.datamodel.CrsGeoCoding",
              "externalDEMApplyEGM": False, "sample_count": len(results),
              "max_abs_error_m": max(abs(p["error_m"]) for p in results),
              "median_abs_error_m": float(np.median([abs(p["error_m"]) for p in results])),
              "fixture_files": [record(p) for p in sorted(Path(fixtures).glob("*.tif"))],
              "samples": results}
    Path(json_out).write_text(json.dumps(report, indent=2) + "\n")


def run(out, overlay_manifest):
    out, overlay_manifest = Path(out), Path(overlay_manifest)
    if out.exists():
        raise FileExistsError("Preserve prior diagnostic; use a fresh proof directory")
    out.mkdir(parents=True)
    make_fixtures(out / "fixtures")
    reports = {}
    for mode in ("vendor", "corrected"):
        target, log = out / (mode + ".json"), out / (mode + ".log")
        argv = [sys.executable, str(Path(__file__).resolve()), "--worker", mode,
                "--fixtures", str((out / "fixtures").resolve()),
                "--overlay-manifest", str(overlay_manifest.resolve()), "--json-out", str(target.resolve())]
        with log.open("w") as stream:
            process = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
        if process.returncode != 0:
            raise RuntimeError("Independent JVM diagnostic failed; preserved " + str(log))
        reports[mode] = json.loads(target.read_text())
    if reports["vendor"]["sample_count"] != 116 or reports["corrected"]["sample_count"] != 116:
        raise ValueError("Fixture/sample coverage changed")
    vendor_plane = [s["error_m"] for s in reports["vendor"]["samples"] if s["fixture"].startswith("plane-")]
    if max(abs(v - 55) for v in vendor_plane) > 1e-6:
        raise ValueError("Vendor half-cell analytic failure was not reproduced")
    if reports["corrected"]["max_abs_error_m"] > 1e-6:
        raise ValueError("Overlay did not reproduce independent analytic samples")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "verified on analytic DEM fixtures",
              "scope": "Actual installed reader and external DEM model, with Point/Area centers and noninteger analytic samples",
              "reference_labels_loaded": False, "gpu_used": False,
              "vendor_plane_error_m": [min(vendor_plane), max(vendor_plane)],
              "corrected_max_abs_error_m": reports["corrected"]["max_abs_error_m"],
              "corrected_median_abs_error_m": reports["corrected"]["median_abs_error_m"],
              "samples_per_jvm": reports["corrected"]["sample_count"],
              "original_runtime_modified": False, "geographic_centers_preserved": True,
              "overlay_manifest": record(overlay_manifest), "proof_script": record(Path(__file__)),
              "artifacts": [record(p) for p in sorted(out.rglob("*")) if p.is_file()],
              "limitations": ["Analytic proof does not establish absolute satellite geolocation accuracy",
                              "Fresh corrected terrain output DEM must separately agree at nominal output centers",
                              "Local correction is not ESA-validated or endorsed"]}
    (out / "proof.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-manifest", type=Path, default=ROOT / "data/snap-external-dem-pixel-correction/manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/snap-dem-center-proof")
    parser.add_argument("--worker", choices=["vendor", "corrected"])
    parser.add_argument("--fixtures", type=Path); parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.fixtures, args.overlay_manifest, args.json_out)
    else:
        print(json.dumps(run(args.out, args.overlay_manifest), indent=2))
