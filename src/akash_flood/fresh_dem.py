"""Bounded, declared geographic CopDEM mosaics for fresh SAR cases."""
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window

from .catalog import validate_bbox

NODATA = -32767.0


def mosaic_orthometric(paths, bbox, target, *, maximum_pixels=4_000_000):
    """Copy native centers if grids align; declare bilinear otherwise.

    Coarsest source spacing is used when source longitude spacings differ. This
    resampling does not create finer terrain information. No hole filling.
    """
    west, south, east, north = validate_bbox(bbox)
    if not paths or len(paths) > 4 or Path(target).exists():
        raise ValueError('Require 1–4 source tiles and a fresh output path')
    sources = []
    for path in paths:
        with rasterio.open(path) as ds:
            t = ds.transform
            if ds.count != 1 or ds.crs != rasterio.crs.CRS.from_epsg(4326) or t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
                raise ValueError('Require north-up geographic CopDEM source tiles')
            sources.append({'path': Path(path), 'transform': t, 'shape': ds.shape, 'bounds': ds.bounds})
    dx = max(s['transform'].a for s in sources); dy = max(-s['transform'].e for s in sources)
    anchor = sources[0]['transform']
    c0 = math.floor((west - anchor.c) / dx) - 2
    c1 = math.ceil((east - anchor.c) / dx) + 2
    r0 = math.floor((anchor.f - north) / dy) - 2
    r1 = math.ceil((anchor.f - south) / dy) + 2
    width, height = c1 - c0, r1 - r0
    if width < 1 or height < 1 or width * height > maximum_pixels:
        raise ValueError('DEM mosaic exceeds the declared pixel budget')
    grid = Affine(dx, 0, anchor.c + c0 * dx, 0, -dy, anchor.f - r0 * dy)
    merged = np.full((height, width), NODATA, dtype='float32')
    operations = []
    for source in sources:
        t = source['transform']
        offset_x = (t.c - grid.c) / dx; offset_y = (grid.f - t.f) / dy
        aligned = (np.isclose(t.a, dx, atol=1e-14, rtol=0) and np.isclose(-t.e, dy, atol=1e-14, rtol=0)
                   and abs(offset_x - round(offset_x)) < 1e-7 and abs(offset_y - round(offset_y)) < 1e-7)
        with rasterio.open(source['path']) as ds:
            if aligned:
                ox, oy = round(offset_x), round(offset_y)
                col, row = max(0, ox), max(0, oy)
                stopcol, stoprow = min(width, ox + ds.width), min(height, oy + ds.height)
                if stopcol <= col or stoprow <= row:
                    continue
                window = Window(col - ox, row - oy, stopcol - col, stoprow - row)
                data = ds.read(1, window=window, masked=True).astype('float32')
                good = ~np.ma.getmaskarray(data) & np.isfinite(data.data)
                view = merged[row:stoprow, col:stopcol]
                conflict = good & (view != NODATA) & ~np.isclose(view, data.data, atol=1e-4, rtol=0)
                if conflict.any():
                    raise ValueError('Overlapping source tiles disagree at native centers')
                use = good & (view == NODATA); view[use] = data.data[use]
                method = 'integer-window copy; source centers unchanged'
            else:
                # Read only the relevant source window plus interpolation halo.
                bounds = rasterio.transform.array_bounds(height, width, grid)
                fractional = rasterio.windows.from_bounds(*bounds, transform=t)
                left = max(0, math.floor(fractional.col_off) - 2)
                top = max(0, math.floor(fractional.row_off) - 2)
                right = min(ds.width, math.ceil(fractional.col_off + fractional.width) + 2)
                bottom = min(ds.height, math.ceil(fractional.row_off + fractional.height) + 2)
                if right <= left or bottom <= top:
                    continue
                window = Window(left, top, right - left, bottom - top)
                crop = ds.read(1, window=window, masked=True).astype('float32').filled(NODATA)
                warped = np.full_like(merged, NODATA)
                reproject(crop, warped, src_transform=ds.window_transform(window), src_crs=ds.crs,
                          src_nodata=NODATA, dst_transform=grid, dst_crs=ds.crs, dst_nodata=NODATA,
                          resampling=Resampling.bilinear)
                use = np.isfinite(warped) & (warped != NODATA) & (merged == NODATA)
                merged[use] = warped[use]
                method = 'bilinear harmonization to coarsest source angular spacing'
        operations.append({'path': str(source['path'].resolve()), 'method': method,
                           'source_transform': list(t), 'source_shape': list(source['shape'])})
    # Request itself must be covered, including data beneath sea level and zero.
    yy, xx = np.indices(merged.shape)
    lon = grid.c + (xx + .5) * dx; lat = grid.f - (yy + .5) * dy
    requested = (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    valid = np.isfinite(merged) & (merged != NODATA)
    if not requested.any() or not valid[requested].all():
        raise ValueError('CopDEM has missing cells inside the requested processing extent')
    with rasterio.open(target, 'w', driver='GTiff', height=height, width=width, count=1,
                       dtype='float32', crs='EPSG:4326', transform=grid, nodata=NODATA) as ds:
        ds.write(merged, 1)
        ds.update_tags(AREA_OR_POINT='Area', VERTICAL_DATUM='EGM2008',
                       source='Copernicus GLO30 public DSM', horizontal_resampling_declared='true')
        ds.set_band_unit(1, 'metre')
    return {'requested_bbox': list(bbox), 'shape': [height, width], 'transform': list(grid),
            'angular_spacing_deg': [dx, dy], 'operations': operations,
            'horizontal_resampling_used': any('bilinear' in op['method'] for op in operations),
            'requested_cells': int(requested.sum()), 'requested_valid_cells': int((valid & requested).sum()),
            'no_hole_filling': True, 'nodata': NODATA}
