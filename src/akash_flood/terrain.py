"""Small public Copernicus GLO-30 crops and explicit 30 m slope screening."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from scipy.ndimage import correlate

from .catalog import validate_bbox

BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
NODATA = -9999.0


def tile_urls(bbox):
    west, south, east, north = validate_bbox(bbox)
    if not (-60 <= south < north <= 80):
        raise ValueError("This UTM terrain prototype supports latitude -60 to 80")
    urls = []
    for lat in range(math.floor(south), math.ceil(north)):
        for lon in range(math.floor(west), math.ceil(east)):
            tile = f"Copernicus_DSM_COG_10_{'N' if lat >= 0 else 'S'}{abs(lat):02}_00_{'E' if lon >= 0 else 'W'}{abs(lon):03}_00_DEM"
            urls.append(f"{BUCKET}/{tile}/{tile}.tif")
    if len(urls) > 4:
        raise ValueError("Pilot is limited to four DEM tiles; split a larger area")
    return urls


def slope_degrees(elevation, dx, dy):
    """Horn 3x3 derivative; every neighbor must be valid, including at edges."""
    if dx <= 0 or dy <= 0 or elevation.ndim != 2:
        raise ValueError("Slope needs a 2D elevation array and positive metre spacing")
    valid = np.isfinite(elevation) & (elevation != NODATA)
    complete = correlate(valid.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant", cval=0) == 9
    filled = np.where(valid, elevation, 0).astype(np.float64)
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    dzdx = correlate(filled, kernel, mode="constant") / (8 * dx)
    dzdy = correlate(filled, kernel.T, mode="constant") / (8 * dy)
    result = np.full(elevation.shape, NODATA, dtype=np.float32)
    result[complete] = np.degrees(np.arctan(np.hypot(dzdx[complete], dzdy[complete])))
    return result


def _write(path, array, profile, **tags):
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)
        dst.update_tags(**tags)


def fetch_terrain(bbox, out_dir):
    west, south, east, north = validate_bbox(bbox)
    # Halo for derivatives and reprojection at the requested AOI boundary.
    padded = [west-.003, south-.003, east+.003, north+.003]
    urls = tile_urls(padded)
    zone = min(60, math.floor(((west+east)/2+180)/6)+1)
    epsg = (32600 if (south+north)/2 >= 0 else 32700)+zone
    bounds = transform_bounds(4326, epsg, *bbox, densify_pts=21)
    left, bottom = [math.floor(v/30)*30-90 for v in bounds[:2]]
    right, top = [math.ceil(v/30)*30+90 for v in bounds[2:]]
    width, height = round((right-left)/30), round((top-bottom)/30)
    if width*height > 4_000_000:
        raise ValueError("Terrain crop exceeds 4 million cells; split the pilot area")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Terrain output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", height=height, width=width, count=1, dtype="float32",
                   crs=f"EPSG:{epsg}", transform=from_origin(left, top, 30, 30), nodata=NODATA,
                   compress="deflate")
    merged = np.full((height, width), NODATA, dtype=np.float32)
    sources = []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        for i, url in enumerate(urls):
            with rasterio.open(url) as ds:
                window = from_bounds(*padded, ds.transform)
                c0, r0 = math.floor(window.col_off), math.floor(window.row_off)
                c1, r1 = math.ceil(window.col_off+window.width), math.ceil(window.row_off+window.height)
                window = Window(c0, r0, c1-c0, r1-r0).intersection(Window(0, 0, ds.width, ds.height))
                crop = ds.read(1, window=window, masked=True).astype(np.float32).filled(NODATA)
                source_transform = ds.window_transform(window)
                source_profile = dict(profile, width=crop.shape[1], height=crop.shape[0],
                                      crs=ds.crs, transform=source_transform)
                crop_path = out/f"source-crop-{i}.tif"
                _write(crop_path, crop, source_profile, source_url=url)
                warped = np.full_like(merged, NODATA)
                reproject(crop, warped, src_transform=source_transform, src_crs=ds.crs, src_nodata=NODATA,
                          dst_transform=profile['transform'], dst_crs=profile['crs'], dst_nodata=NODATA,
                          resampling=Resampling.bilinear)
                use = (merged == NODATA) & (warped != NODATA) & np.isfinite(warped)
                merged[use] = warped[use]
                sources.append({"url": url, "local_crop": crop_path.name,
                                "crop_sha256": hashlib.sha256(crop_path.read_bytes()).hexdigest()})
    if not np.any(merged != NODATA):
        raise ValueError("DEM crop has no valid elevations")
    slope = slope_degrees(merged, 30, 30)
    _write(out/"dem-30m.tif", merged, profile, units="metres", source="Copernicus GLO-30 DSM")
    _write(out/"slope-30m.tif", slope, profile, units="degrees", algorithm="Horn 3x3; complete neighborhoods only")
    valid = slope != NODATA
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "bbox": list(bbox),
                "sources": sources, "crs": profile['crs'], "resolution_m": 30,
                "shape": [height, width], "valid_slope_pixels": int(valid.sum()),
                "slope_percentiles_deg": dict(zip(["p05", "p50", "p95"], np.percentile(slope[valid], [5,50,95]).tolist())),
                "grid_fraction_slope_at_most_5_deg": float(np.mean(slope[valid] <= 5)),
                "threshold_sensitivity": {str(t): {"eligible_pixels": int(np.sum(slope[valid] <= t)),
                                                   "fraction_of_valid_terrain": float(np.mean(slope[valid] <= t))}
                                          for t in [5, 10, 15, 20]},
                "statistic_extent": "30 m projected grid with 90 m halo; not a flood result",
                "limitations": ["DSM includes buildings and vegetation; not bare-earth terrain.",
                                "Slope threshold is not a radar layover/shadow mask.",
                                "Thirty-metre derivatives do not resolve ten-metre terrain details."],
                "attribution": "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."}
    metadata['outputs'] = {name: hashlib.sha256((out/name).read_bytes()).hexdigest()
                           for name in ['dem-30m.tif', 'slope-30m.tif']}
    (out/"terrain.provenance.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def align_slope(slope_path, reference_path, out_path):
    """Nearest-neighbor sampling onto a radar grid; creates no finer terrain detail."""
    if Path(out_path).exists():
        raise ValueError("Refusing to overwrite aligned slope")
    with rasterio.open(slope_path) as src, rasterio.open(reference_path) as reference:
        if src.tags().get('units') != 'degrees' or not src.crs or not reference.crs:
            raise ValueError("Source must be a georeferenced slope raster tagged units=degrees")
        profile = reference.profile.copy()
        profile.update(dtype='float32', count=1, nodata=NODATA, compress='deflate')
        output = np.full(reference.shape, NODATA, dtype=np.float32)
        reproject(rasterio.band(src, 1), output, src_nodata=src.nodata,
                  dst_transform=reference.transform, dst_crs=reference.crs,
                  dst_nodata=NODATA, resampling=Resampling.nearest)
        _write(out_path, output, profile, units='degrees', resampling='nearest',
               source_sha256=hashlib.sha256(Path(slope_path).read_bytes()).hexdigest(),
               reference_sha256=hashlib.sha256(Path(reference_path).read_bytes()).hexdigest(),
               limitation='Resampling does not increase terrain information resolution')


def preview_layer(slope_path, bbox, width=600, height=800):
    """WGS84 RGBA view of the measured slope; transparent where unassessed."""
    west, south, east, north = bbox
    with rasterio.open(slope_path) as ds:
        target = np.full((height,width), NODATA, dtype=np.float32)
        reproject(rasterio.band(ds,1), target, src_nodata=ds.nodata,
                  dst_transform=from_origin(west,north,(east-west)/width,(north-south)/height),
                  dst_crs='EPSG:4326', dst_nodata=NODATA, resampling=Resampling.nearest)
    valid = target != NODATA
    # Cream-to-slate ramp: darker means steeper, not flood or damage.
    scale = np.clip(target/60, 0, 1)[...,None]
    colors = np.array([245,237,212])*(1-scale)+np.array([55,77,89])*scale
    return np.dstack((colors.astype(np.uint8), np.where(valid,210,0).astype(np.uint8)))
