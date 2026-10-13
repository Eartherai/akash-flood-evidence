"""Compare SNAP context runs on exact pixel windows, without resampling.

This examines processing sensitivity, not flood accuracy. It never reads
evaluation labels. The center bbox is supplied independently of any reference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from affine import Affine
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window

from process_s1_snap import BANDS, record, validate_bbox


def exact_window(source_transform, source_shape, target_transform, target_shape):
    """Require equal grids and integer shifts; never hide a shift by resampling."""
    if not np.allclose([source_transform.a, source_transform.b, source_transform.d, source_transform.e], [target_transform.a, target_transform.b, target_transform.d, target_transform.e], rtol=0, atol=1e-10):
        raise ValueError("Raster pixel basis differs")
    col, row = (~source_transform) * (target_transform.c, target_transform.f)
    if abs(col - round(col)) > 1e-7 or abs(row - round(row)) > 1e-7:
        raise ValueError("Rasters differ by a fractional pixel")
    col, row = int(round(col)), int(round(row))
    height, width = target_shape
    if col < 0 or row < 0 or col + width > source_shape[1] or row + height > source_shape[0]:
        raise ValueError("Requested comparison window exceeds source coverage")
    return Window(col, row, width, height)


def image_grid(folder, scene):
    with rasterio.open(folder / f"{scene}-terrain.data/Sigma0_VV.img") as dataset:
        return {"transform": dataset.transform, "shape": dataset.shape, "bounds": dataset.bounds, "crs": dataset.crs}


def common_grid(grids):
    if any(grid["crs"] != grids[0]["crs"] for grid in grids):
        raise ValueError("Comparison coordinate systems differ")
    left = max(grid["bounds"].left for grid in grids)
    right = min(grid["bounds"].right for grid in grids)
    bottom = max(grid["bounds"].bottom for grid in grids)
    top = min(grid["bounds"].top for grid in grids)
    transform = grids[0]["transform"]
    col, row = (~transform) * (left, top)
    width = (right - left) / transform.a
    height = (top - bottom) / -transform.e
    if width <= 0 or height <= 0:
        raise ValueError("Comparison has no overlap")
    if max(abs(v - round(v)) for v in [col, row, width, height]) > 1e-7:
        raise ValueError("Overlap edges are not on an exact common pixel grid")
    target = transform * Affine.translation(round(col), round(row))
    shape = (round(height), round(width))
    windows = [exact_window(g["transform"], g["shape"], target, shape) for g in grids]
    return target, shape, windows


def bbox_mask(bbox, transform, shape, crs):
    west, south, east, north = validate_bbox(bbox)
    ring = []
    corners = [(west, south), (east, south), (east, north), (west, north), (west, south)]
    for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:]):
        ring.extend((x0 + f * (x1 - x0), y0 + f * (y1 - y0)) for f in np.linspace(0, 1, 41, endpoint=False))
    ring.append(ring[0])
    tx = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xy = [list(tx.transform(x, y)) for x, y in ring]
    mask = geometry_mask([{"type": "Polygon", "coordinates": [xy]}], out_shape=shape, transform=transform, invert=True, all_touched=False)
    if not mask.any():
        raise ValueError("Center bbox has no comparison pixels")
    return mask


def load(folder, scene, window, shape):
    bands = {}
    for band in BANDS:
        with rasterio.open(folder / f"{scene}-terrain.data/{band}.img") as dataset:
            a = dataset.read(1, window=window)
            if a.shape != shape:
                raise ValueError("Incomplete comparison window")
            bands[band] = a
    return bands


def validity(bands):
    valid = np.ones(bands["elevation"].shape, bool)
    for band in BANDS[:-1]:
        valid &= np.isfinite(bands[band]) & (bands[band] > 0)
    valid &= (bands["localIncidenceAngle"] < 90) & (bands["projectedLocalIncidenceAngle"] < 90)
    return valid, valid & (bands["layoverShadowMask"] == 0)


def compare(old, new, region):
    covered_old, usable_old = validity(old)
    covered_new, usable_new = validity(new)
    covered_both = region & covered_old & covered_new
    usable_both = region & usable_old & usable_new
    mask_diff = old["layoverShadowMask"] != new["layoverShadowMask"]
    results = {
        "region_pixels": int(region.sum()),
        "covered_original": int((region & covered_old).sum()),
        "covered_expanded": int((region & covered_new).sum()),
        "covered_both": int(covered_both.sum()),
        "usable_original": int((region & usable_old).sum()),
        "usable_expanded": int((region & usable_new).sum()),
        "usable_both": int(usable_both.sum()),
        "usable_only_original": int((region & usable_old & ~usable_new).sum()),
        "usable_only_expanded": int((region & ~usable_old & usable_new).sum()),
        "mask_disagreements_on_both_covered": int((covered_both & mask_diff).sum()),
        "mask_disagreements_all_region": int((region & mask_diff).sum()),
        "mask_counts_original": {str(v): int((region & (old["layoverShadowMask"] == v)).sum()) for v in range(4)},
        "mask_counts_expanded": {str(v): int((region & (new["layoverShadowMask"] == v)).sum()) for v in range(4)},
        "bands": {},
    }
    for band in BANDS[:-1]:
        common = usable_both if band.startswith("Sigma") else covered_both
        if not common.any():
            results["bands"][band] = {"common_pixels": 0}
            continue
        a = old[band][common].astype(np.float64)
        b = new[band][common].astype(np.float64)
        d = b - a
        stats = {"common_pixels": int(common.sum()), "exactly_equal_pixels": int((a == b).sum()), "absolute_difference_quantiles": np.quantile(np.abs(d), [0, .5, .9, .99, 1]).tolist(), "mean_signed_difference": float(d.mean()), "root_mean_squared_difference": float(np.sqrt(np.mean(d * d)))}
        if band.startswith("Sigma"):
            dd = 10 * np.log10(b) - 10 * np.log10(a)
            stats["absolute_difference_db_quantiles"] = np.quantile(np.abs(dd), [0, .5, .9, .99, 1]).tolist()
        results["bands"][band] = stats
    return results


def context_bbox(folder, scene):
    p = ET.parse(folder / f"{scene}-terrain.xml").find("./node[@id='subset']/parameters/geoRegion")
    return p.text if p is not None else None


def run(original: Path, expanded: Path, center_bbox, out: Path):
    if out.exists():
        raise FileExistsError("Refuse to overwrite a context report")
    labels = [(original, "before"), (original, "after"), (expanded, "before"), (expanded, "after")]
    grids = [image_grid(folder, scene) for folder, scene in labels]
    target, shape, windows = common_grid(grids)
    region = np.ones(shape, bool)
    center = bbox_mask(center_bbox, target, shape, grids[0]["crs"])
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "scope": "Context sensitivity and grid consistency; no evaluation labels, classifier tuning, or geolocation accuracy claim", "comparison_code": record(Path(__file__)), "no_resampling": True, "common_grid": {"crs": str(grids[0]["crs"]), "shape": list(shape), "transform": list(target)[:6]}, "center_bbox": list(center_bbox), "inputs": [], "scenes": {}}
    for folder, scene in labels:
        artifacts = [record(folder / f"{scene}-terrain.dim"), record(folder / f"{scene}-terrain.xml"), record(folder / f"{scene}-terrain.log")]
        artifacts += [record(p) for p in sorted((folder / f"{scene}-terrain.data").rglob("*")) if p.is_file()]
        report["inputs"].append({"folder": str(folder), "scene": scene, "geo_subset": context_bbox(folder, scene), "artifacts": artifacts})
    for i, scene in enumerate(["before", "after"]):
        old = load(original, scene, windows[i], shape)
        new = load(expanded, scene, windows[i + 2], shape)
        report["scenes"][scene] = {"full_original_common_footprint": compare(old, new, region), "central_bbox": compare(old, new, center), "original_window": list(windows[i].flatten()), "expanded_window": list(windows[i + 2].flatten())}
    metadata = expanded / "run-metadata.json"
    if metadata.is_file():
        report["expanded_run_metadata"] = record(metadata)
    report["interpretation"] = "All-zero mask cells were excluded using image/DEM/angle coverage; any mask or sigma differences are reported, not silently discarded. Two context sizes test sensitivity but cannot prove universal complete shadow coverage."
    out.mkdir(parents=True)
    path = out / "context-stability.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    (out / "context-stability.sha256").write_text(hashlib.file_digest(path.open("rb"), "sha256").hexdigest() + "\n")
    print(json.dumps({scene: report["scenes"][scene]["central_bbox"] for scene in ["before", "after"]}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--center-bbox", nargs=4, type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.original.resolve(), args.expanded.resolve(), validate_bbox(args.center_bbox), args.out.resolve())


if __name__ == "__main__":
    main()
