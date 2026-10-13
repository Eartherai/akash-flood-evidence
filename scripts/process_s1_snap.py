"""Reproduce the native Sentinel-1 sigma0/geometric correction pipeline.

Uses the locally verified derived COG SAFE wrappers and cached SRTM/EGM96.
Outputs retain un-clipped negative noise-corrected powers. They are not a
flood/debris classifier, precise-orbit product, or radiometrically flattened RTC.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from pyproj import CRS
import rasterio

from snap_gpt import command

ROOT = Path(__file__).resolve().parents[1]
BANDS = (
    "Sigma0_VV", "Sigma0_VH", "elevation", "localIncidenceAngle",
    "projectedLocalIncidenceAngle", "incidenceAngleFromEllipsoid", "layoverShadowMask",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def validate_bbox(bbox) -> tuple[float, float, float, float]:
    west, south, east, north = map(float, bbox)
    if not all(np.isfinite([west, south, east, north])) or not (84 <= west < east <= 90 and 0 <= south < north < 84):
        raise ValueError("Require an increasing finite bbox inside UTM zone 45N")
    return west, south, east, north


def graph(scene: str, manifest: Path, target: Path, bbox) -> ET.ElementTree:
    west, south, east, north = validate_bbox(bbox)
    doc = ET.Element("graph", id=f"native-s1-geometric-{scene}")
    ET.SubElement(doc, "version").text = "1.0"

    def node(name, operator, source=None, source_alias="sourceProduct", **parameters):
        elem = ET.SubElement(doc, "node", id=name)
        ET.SubElement(elem, "operator").text = operator
        if source:
            ET.SubElement(ET.SubElement(elem, "sources"), source_alias, refid=source)
        params = ET.SubElement(elem, "parameters")
        for key, value in parameters.items():
            ET.SubElement(params, key).text = str(value).lower() if isinstance(value, bool) else str(value)

    node("read", "Read", file=manifest.resolve())
    node("subset", "Subset", "read", geoRegion=f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))", copyMetadata=True)
    node("noise", "ThermalNoiseRemoval", "subset", selectedPolarisations="VV,VH", removeThermalNoise=True, clipNegativeValues=False)
    node("calibrate", "Calibration", "noise", source_alias="source", selectedPolarisations="VV,VH", outputSigmaBand=True, outputBetaBand=False, outputImageScaleInDb=False, auxFile="Product Auxiliary File")
    node("terrain", "Terrain-Correction", "calibrate", source_alias="source", sourceBands="Sigma0_VV,Sigma0_VH", demName="SRTM 1Sec HGT", demResamplingMethod="BILINEAR_INTERPOLATION", imgResamplingMethod="BILINEAR_INTERPOLATION", pixelSpacingInMeter=10, mapProjection=CRS.from_epsg(32645).to_wkt(version="WKT1_GDAL"), alignToStandardGrid=True, standardGridOriginX=0, standardGridOriginY=0, saveDEM=True, saveLocalIncidenceAngle=True, saveProjectedLocalIncidenceAngle=True, saveIncidenceAngleFromEllipsoid=True, saveLayoverShadowMask=True, nodataValueAtSea=True, applyRadiometricNormalization=False, saveSelectedSourceBand=True)
    node("write", "Write", "terrain", file=target.resolve(), formatName="BEAM-DIMAP")
    ET.indent(doc)
    return ET.ElementTree(doc)


def wrapper_manifest(index_path: Path, scene: str) -> tuple[Path, dict]:
    entries = json.loads(index_path.read_text())
    entry = next(item for item in entries if item["scene"] == scene)
    manifest = ROOT / entry["derived_safe"] / "manifest.safe"
    if sha256(manifest) != entry["derived_manifest_sha256"]:
        raise ValueError(f"{scene} derived manifest changed")
    for item in entry["files"]:
        file = manifest.parent / item["alias"]
        if not file.is_file() or file.stat().st_size != item["bytes"] or sha256(file) != item["sha256"]:
            raise ValueError(f"{scene} source object differs from derived provenance: {item['alias']}")
    return manifest, entry


def cached_auxdata(bbox) -> list[dict]:
    west, south, east, north = validate_bbox(bbox)
    cache = ROOT / "data/snap-runtime/user-cache/auxdata/dem"
    paths = [cache / "egm96/ww15mgh_b.zip"]
    for lat in range(int(np.floor(south)), int(np.floor(north)) + 1):
        for lon in range(int(np.floor(west)), int(np.floor(east)) + 1):
            paths.append(cache / f"SRTM 1Sec HGT/N{lat:02d}E{lon:03d}.SRTMGL1.hgt.zip")
    result = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Cached-only run requires {path}; this runner does not fetch source data")
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Incomplete/corrupt auxiliary archive {path}")
        result.append(record(path))
    return result


def validate_output(folder: Path, scene: str) -> dict:
    stats = {}
    arrays = {}
    transform = None
    shape = None
    for name in BANDS:
        with rasterio.open(folder / f"{scene}-terrain.data/{name}.img") as dataset:
            if dataset.crs != CRS.from_epsg(32645) or not np.allclose([dataset.transform.a, dataset.transform.e, dataset.transform.b, dataset.transform.d], [10, -10, 0, 0], atol=1e-9):
                raise ValueError("Unexpected terrain projection or spacing")
            if transform is not None and (dataset.transform != transform or dataset.shape != shape):
                raise ValueError("Bands have differing output grids")
            transform, shape = dataset.transform, dataset.shape
            a = dataset.read(1)
            arrays[name] = a
            stats[name] = {"finite": int(np.isfinite(a).sum()), "zero": int((a == 0).sum()), "negative": int((a < 0).sum()), "min": float(np.nanmin(a)), "max": float(np.nanmax(a)), "rasterio_nodata": dataset.nodata}
    valid = np.ones(shape, dtype=bool)
    for name in BANDS[:-1]:
        valid &= np.isfinite(arrays[name]) & (arrays[name] > 0)
    valid &= (arrays["localIncidenceAngle"] < 90) & (arrays["projectedLocalIncidenceAngle"] < 90)
    coverage = valid.copy()
    valid &= arrays["layoverShadowMask"] == 0
    values, counts = np.unique(arrays["layoverShadowMask"], return_counts=True)
    if not set(values.tolist()).issubset({0, 1, 2, 3}):
        raise ValueError("Unknown layover/shadow encoding")
    if not coverage.any() or not valid.any():
        raise ValueError("Terrain output has no valid imagery/DEM/angles")
    # Rasuwa is mountainous: an all-zero mask is suspicious, not proof of clear geometry.
    if not np.any(arrays["layoverShadowMask"][coverage] != 0):
        raise ValueError("Mountainous scene has an empty terrain mask; refuse silent missing-mask output")
    return {"shape": list(shape), "crs": "EPSG:32645", "transform": list(transform)[:6], "bands": stats, "mask_values": {str(int(v)): int(n) for v, n in zip(values, counts)}, "geometrically_usable_pixels": int(valid.sum()), "positive_coverage_before_mask": int(coverage.sum()), "mask_zero_is_not_validity": True, "validation_scope": "Output consistency and nonempty masks; not independent absolute geolocation or complete shadow coverage"}


def run(bbox, scenes, output: Path, index_path: Path) -> dict:
    bbox = validate_bbox(bbox)
    if output.exists():
        raise FileExistsError("Refuse to reuse an output directory; select a fresh path")
    aux = cached_auxdata(bbox)
    sources = {scene: wrapper_manifest(index_path, scene) for scene in scenes}
    output.mkdir(parents=True)
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "bbox": list(bbox), "scenes": {}, "source_index": record(index_path), "auxiliary_inputs": aux, "tool_sources": [record(Path(__file__)), record(ROOT / "scripts/snap_gpt.py"), record(ROOT / "uv.lock")], "runtime": json.loads((ROOT / "data/snap-runtime/runtime-download-manifest.json").read_text()), "processing": {"thermal_noise_removed": True, "negative_noise_corrected_powers_clipped": False, "calibration": "sigma0 linear power", "terrain_correction": "Range-Doppler geometric correction, SRTM1SecHGT + EGM96", "projection": "EPSG:32645 WKT1_GDAL", "spacing_m": 10, "standard_grid_pixel_center_origin": [0, 0], "orbit": "Embedded source product state vectors; no applied precise orbit", "radiometric_terrain_flattening": False, "gpt_parallelism": 4, "cache": "512M", "heap": "6G"}, "classification_or_hazard_claim": False}
    report_path = output / "run-metadata.json"
    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    save()
    for scene in scenes:
        manifest, source = sources[scene]
        xml = output / f"{scene}-terrain.xml"
        graph(scene, manifest, output / f"{scene}-terrain.dim", bbox).write(xml, encoding="utf-8", xml_declaration=True)
        argv = command([str(xml.resolve()), "-q", "4", "-c", "512M"])
        entry = {"source_item_id": source["source_item_id"], "source_manifest": record(manifest), "source_wrapper_provenance": record(manifest.parent / "derived-wrapper-provenance.json"), "graph": record(xml), "command": argv, "started_at": datetime.now(timezone.utc).isoformat()}
        report["scenes"][scene] = entry
        save()
        start = time.perf_counter()
        log = output / f"{scene}-terrain.log"
        with log.open("w") as stream:
            result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
        entry.update({"exit_code": result.returncode, "wall_seconds": time.perf_counter() - start, "finished_at": datetime.now(timezone.utc).isoformat(), "log": record(log)})
        save()
        if result.returncode != 0 or "done." not in log.read_text():
            raise RuntimeError(f"SNAP {scene} did not finish successfully; retained log and metadata")
        entry["validation"] = validate_output(output, scene)
        entry["artifacts"] = [record(p) for p in sorted(output.glob(f"{scene}-terrain.*")) if p.is_file()] + [record(p) for p in sorted((output / f"{scene}-terrain.data").rglob("*")) if p.is_file()]
        save()
        print(scene, "complete", entry["validation"]["geometrically_usable_pixels"], "usable pixels", flush=True)
    # Pin the auxiliary inputs actually present after both runs as well.
    report["auxiliary_inputs_after"] = cached_auxdata(bbox)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--scene", choices=["before", "after", "both"], default="both")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, default=ROOT / "data/s1-snap-native/derived-wrapper-index.json")
    args = parser.parse_args()
    run(args.bbox, ["before", "after"] if args.scene == "both" else [args.scene], args.out.resolve(), args.source_index.resolve())


if __name__ == "__main__":
    main()
