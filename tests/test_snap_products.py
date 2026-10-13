from pathlib import Path
import xml.etree.ElementTree as ET

from affine import Affine
import numpy as np
import pytest
import rasterio

from akash_flood.snap_products import (BANDS, ANGLES, SnapProduct, geometry_support,
                                       integer_overlap, product_summary, read_snap_product, validate_grid)


def make_product(tmp_path, epsg=32719):
    path = tmp_path / "fresh.dim"
    folder = path.with_suffix(".data")
    folder.mkdir()
    doc = ET.Element("Dimap_Document")
    metadata = ET.SubElement(doc, "MDElem", name="Abstracted_Metadata")
    for key, value in {"REL_ORBIT": "24", "PASS": "DESCENDING", "abs_calibration_flag": "1"}.items():
        ET.SubElement(metadata, "MDATTR", name=key).text = value
    for name in BANDS:
        band = ET.SubElement(doc, "Spectral_Band_Info")
        unit = "intensity" if name.startswith("Sigma0") else "meters" if name == "elevation" else "deg" if name in ANGLES else ""
        nodata = -32767 if name == "elevation" else 0
        for key, value in {"BAND_NAME": name, "NO_DATA_VALUE_USED": "false" if name == "layoverShadowMask" else "true",
                           "NO_DATA_VALUE": str(nodata), "LOG10_SCALED": "false", "SCALING_FACTOR": "1.0",
                           "SCALING_OFFSET": "0.0", "PHYSICAL_UNIT": unit}.items():
            ET.SubElement(band, key).text = value
        values = np.full((2, 3), 30 if name in ANGLES else -4 if name == "elevation" else .1, dtype="float32")
        if name == "layoverShadowMask":
            values[:] = 0
        with rasterio.open(folder / (name + ".img"), "w", driver="GTiff", width=3, height=2, count=1,
                           dtype="float32", crs=f"EPSG:{epsg}", transform=Affine(10, 0, 500005, 0, -10, 7000005)) as ds:
            ds.write(values, 1)
    ET.ElementTree(doc).write(path)
    return path


def test_reads_southern_utm_and_accepts_flat_mask_with_signed_dem(tmp_path):
    path = make_product(tmp_path)
    product = read_snap_product(path, 32719)
    assert product.grid[0].to_epsg() == 32719
    assert product.band_metadata["elevation"]["nodata"] == -32767
    assert geometry_support(product).all()
    summary = product_summary(product)
    assert summary["geometry_supported_pixels"] == 6
    assert summary["positive_geometry_supported_pixels"] == 6
    assert summary["mask_values"] == {"0": 6}


def test_nodata_angles_and_negative_power_are_distinct(tmp_path):
    product = read_snap_product(make_product(tmp_path))
    product.arrays["elevation"][0] = [0, -32767, -100]
    product.arrays["Sigma0_VV"][1] = [-.1, 0, .1]
    product.arrays["localIncidenceAngle"][1, 2] = 95
    assert geometry_support(product).tolist() == [[True, False, True], [True, False, False]]
    assert geometry_support(product, positive_power=True).tolist() == [[True, False, True], [False, False, False]]


def test_reader_refuses_wrong_zone_and_non_utm(tmp_path):
    path = make_product(tmp_path)
    with pytest.raises(ValueError, match="planned UTM"):
        read_snap_product(path, 32645)
    with pytest.raises(ValueError, match="WGS84 UTM"):
        validate_grid((rasterio.crs.CRS.from_epsg(4326), Affine(10, 0, 5, 0, -10, 5), 3, 2))


def test_reader_refuses_fractional_standard_grid_and_scaling(tmp_path):
    with pytest.raises(ValueError, match="centre origin"):
        validate_grid((rasterio.crs.CRS.from_epsg(32719), Affine(10, 0, 500000, 0, -10, 7000005), 3, 2))
    path = make_product(tmp_path)
    doc = ET.parse(path)
    doc.find(".//SCALING_FACTOR").text = "2"
    doc.write(path)
    with pytest.raises(ValueError, match="unscaled linear"):
        read_snap_product(path)


def test_reader_checks_band_grid_consistency_and_terrain_encoding(tmp_path):
    path = make_product(tmp_path)
    with rasterio.open(path.with_suffix(".data") / "Sigma0_VH.img", "r+") as ds:
        ds.transform = ds.transform * Affine.translation(1, 0)
    with pytest.raises(ValueError, match="band grids"):
        read_snap_product(path)
    product = SnapProduct({name: np.ones((1, 1)) for name in BANDS}, (), {},
                          {name: {"nodata": None} for name in BANDS})
    product.arrays["layoverShadowMask"][:] = 4
    with pytest.raises(ValueError, match="encoding"):
        geometry_support(product)


def test_exact_southern_overlap_does_not_resample_and_rejects_zone_mismatch():
    crs = rasterio.crs.CRS.from_epsg(32719)
    transform = Affine(10, 0, 500005, 0, -10, 7000005)
    left = (crs, transform, 10, 10)
    right = (crs, transform * Affine.translation(2, 3), 10, 10)
    a, b, grid = integer_overlap(left, right)
    assert (a.col_off, a.row_off, a.width, a.height) == (2, 3, 8, 7)
    assert (b.col_off, b.row_off) == (0, 0)
    assert grid[2:] == (8, 7)
    with pytest.raises(ValueError, match="projections differ"):
        integer_overlap(left, (rasterio.crs.CRS.from_epsg(32619), right[1], 10, 10))
