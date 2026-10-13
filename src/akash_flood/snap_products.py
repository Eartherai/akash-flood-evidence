"""Read local SNAP sigma0 products without assuming a fixed UTM zone.

This checks product/grid consistency. It does not independently establish SAR
geolocation, complete terrain masks, radiometric flattening or hazard accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.windows import Window

BANDS = (
    "Sigma0_VV", "Sigma0_VH", "elevation", "localIncidenceAngle",
    "projectedLocalIncidenceAngle", "incidenceAngleFromEllipsoid", "layoverShadowMask",
)
ANGLES = ("localIncidenceAngle", "projectedLocalIncidenceAngle", "incidenceAngleFromEllipsoid")


@dataclass
class SnapProduct:
    arrays: dict
    grid: tuple
    attributes: dict
    band_metadata: dict


def validate_grid(grid, expected_epsg=None):
    crs, transform, width, height = grid
    epsg = crs.to_epsg() if crs else None
    if epsg is None or not (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760):
        raise ValueError("Require a WGS84 UTM SNAP grid")
    if expected_epsg is not None and epsg != expected_epsg:
        raise ValueError("SNAP grid differs from the planned UTM projection")
    if (width <= 0 or height <= 0
            or not np.allclose([transform.a, transform.e, transform.b, transform.d],
                               [10, -10, 0, 0], atol=1e-9, rtol=0)):
        raise ValueError("Require a north-up 10 metre SNAP grid")
    if not np.isfinite(list(transform)[:6]).all():
        raise ValueError("Nonfinite SNAP grid")
    # SNAP's standardGridOrigin specifies pixel centres. Rasterio reports corners.
    centers = ((transform.c + 5) / 10, (transform.f - 5) / 10)
    if not all(np.isclose(value, round(value), atol=1e-8, rtol=0) for value in centers):
        raise ValueError("SNAP grid is not aligned to the declared 0 metre centre origin")


def read_snap_product(path, expected_epsg=None):
    path = Path(path)
    doc = ET.parse(path)
    info = {}
    for element in doc.findall(".//Spectral_Band_Info"):
        name = element.findtext("BAND_NAME")
        if name in info:
            raise ValueError("Duplicate SNAP band metadata")
        if name in BANDS:
            info[name] = element
    if set(BANDS) - set(info):
        raise ValueError("SNAP output lacks required observation/terrain bands")
    arrays, grid, band_metadata = {}, None, {}
    for name in BANDS:
        element = info[name]
        used_text = element.findtext("NO_DATA_VALUE_USED")
        if used_text not in ("true", "false"):
            raise ValueError("SNAP band has no explicit nodata convention")
        nodata = float(element.findtext("NO_DATA_VALUE")) if used_text == "true" else None
        if nodata is not None and not np.isfinite(nodata):
            raise ValueError("SNAP nodata must be finite and explicit")
        if (element.findtext("LOG10_SCALED") != "false"
                or float(element.findtext("SCALING_FACTOR")) != 1
                or float(element.findtext("SCALING_OFFSET")) != 0):
            raise ValueError("Require unscaled linear SNAP bands")
        unit = element.findtext("PHYSICAL_UNIT")
        expected_unit = "intensity" if name.startswith("Sigma0_") else "meters" if name == "elevation" else "deg" if name in ANGLES else None
        if expected_unit is not None and unit != expected_unit:
            raise ValueError("Unexpected SNAP band units")
        with rasterio.open(path.with_suffix(".data") / (name + ".img")) as dataset:
            current = (dataset.crs, dataset.transform, dataset.width, dataset.height)
            validate_grid(current, expected_epsg)
            if dataset.count != 1 or (grid is not None and current != grid):
                raise ValueError("SNAP band grids or counts differ")
            if dataset.nodata is not None and dataset.nodata != nodata:
                raise ValueError("SNAP XML and raster nodata differ")
            arrays[name] = dataset.read(1)
            grid = current
            band_metadata[name] = {"unit": unit, "nodata": nodata,
                                   "rasterio_nodata": dataset.nodata,
                                   "dtype": dataset.dtypes[0]}
    attributes = {element.attrib["name"]: element.text for element in
                  doc.findall('.//MDElem[@name="Abstracted_Metadata"]/MDATTR')}
    if (not (attributes.get("REL_ORBIT") or "").isdigit()
            or attributes.get("PASS") not in ("ASCENDING", "DESCENDING")
            or attributes.get("abs_calibration_flag") != "1"):
        raise ValueError("Expected calibrated SNAP orbit/pass metadata")
    return SnapProduct(arrays, grid, attributes, band_metadata)


def read_snap(path, expected_epsg=None):
    """Compatibility adapter; product API additionally exposes XML nodata."""
    product = read_snap_product(path, expected_epsg)
    return product.arrays, product.grid, product.attributes


def integer_overlap(left, right):
    """Find shared native cells, refusing reprojection or fractional shifts."""
    validate_grid(left)
    validate_grid(right)
    crs, transform, width, height = left
    other_crs, other_transform, other_width, other_height = right
    if crs != other_crs:
        raise ValueError("SNAP scene projections differ")
    dx = (other_transform.c - transform.c) / 10
    dy = (transform.f - other_transform.f) / 10
    if not all(np.isclose(value, round(value), atol=1e-8, rtol=0) for value in (dx, dy)):
        raise ValueError("SNAP scene origins have a fractional pixel shift")
    dx, dy = round(dx), round(dy)
    col, row = max(0, dx), max(0, dy)
    stop_col, stop_row = min(width, dx+other_width), min(height, dy+other_height)
    if stop_col <= col or stop_row <= row:
        raise ValueError("SNAP scenes have no common native grid cells")
    first = Window(col, row, stop_col-col, stop_row-row)
    second = Window(col-dx, row-dy, first.width, first.height)
    common = (crs, rasterio.windows.transform(first, transform), int(first.width), int(first.height))
    return first, second, common


def geometry_support(product, *, positive_power=False):
    """Require observations and DEM/angles as well as a zero terrain-mask code.

    Signed ellipsoidal DEM heights are valid. Negative noise-corrected power is
    retained unless positive_power=True requests inputs suitable for logarithms.
    """
    arrays = product.arrays
    shape = arrays["Sigma0_VV"].shape
    if any(arrays[name].shape != shape for name in BANDS):
        raise ValueError("SNAP band array shapes differ")
    mask = arrays["layoverShadowMask"]
    if not np.isin(mask, (0, 1, 2, 3)).all():
        raise ValueError("Unexpected SNAP layover/shadow encoding")
    valid = mask == 0
    for name in BANDS[:-1]:
        values = arrays[name]
        valid &= np.isfinite(values)
        nodata = product.band_metadata[name]["nodata"]
        if nodata is not None:
            valid &= values != nodata
        if name in ANGLES:
            valid &= (values > 0) & (values < 90)
        elif name.startswith("Sigma0_") and positive_power:
            valid &= values > 0
    return valid


def product_summary(product):
    geometry = geometry_support(product)
    positive = geometry_support(product, positive_power=True)
    if not positive.any():
        raise ValueError("SNAP output has no positive geometry-supported imagery")
    crs, transform, width, height = product.grid
    stats = {}
    for name, values in product.arrays.items():
        finite = np.isfinite(values)
        stats[name] = {"finite": int(finite.sum()), "zero": int((values == 0).sum()),
                       "negative": int((values < 0).sum()),
                       "min": float(values[finite].min()) if finite.any() else None,
                       "max": float(values[finite].max()) if finite.any() else None,
                       **product.band_metadata[name]}
    values, counts = np.unique(product.arrays["layoverShadowMask"], return_counts=True)
    return {"shape": [height, width], "crs": crs.to_string(),
            "transform": list(transform)[:6], "standard_grid_pixel_center_origin": [0, 0],
            "bands": stats, "mask_values": {str(int(v)): int(n) for v, n in zip(values, counts)},
            "geometry_supported_pixels": int(geometry.sum()),
            "positive_geometry_supported_pixels": int(positive.sum()),
            "signed_ellipsoidal_heights_allowed": True,
            "mask_zero_is_not_validity": True,
            "validation_scope": "Output consistency only; no independent absolute geolocation, terrain-mask completeness or hazard validation"}
