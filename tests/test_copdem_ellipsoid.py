"""Verify geodetic sign, mandatory grids, pixel centers, and nodata handling."""
from pathlib import Path
import sys

from affine import Affine
import numpy as np
import pytest
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_copdem_ellipsoid as converter


def write_raster(path, values, transform, *, geoid=False, crs="EPSG:4326"):
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0], count=1, dtype="float32", crs=crs, transform=transform, nodata=-32767.0) as raster:
        raster.write(values.astype(np.float32), 1)
        raster.update_tags(AREA_OR_POINT="Point")
        if geoid:
            raster.update_tags(TYPE="VERTICAL_OFFSET_GEOGRAPHIC_TO_VERTICAL", TARGET_CRS_EPSG_CODE="3855", SOURCE_CRS_EPSG_CODE="4979", VERTICAL_DATUM="EGM2008")
            raster.set_band_description(1, "geoid_undulation")
            raster.set_band_unit(1, "metre")


@pytest.mark.parametrize("undulation", [20.0, -30.0])
@pytest.mark.parametrize('geoid_crs', ['EPSG:4326','EPSG:4979'])
def test_explicit_positive_multiplier_adds_signed_geoid_and_preserves_nodata(tmp_path, undulation, geoid_crs):
    folder = tmp_path / "grid with spaces"
    folder.mkdir()
    geoid = folder / "fake_egm08.tif"
    write_raster(geoid, np.full((20, 20), undulation), Affine(.1, 0, 84.5, 0, -.1, 29.5), geoid=True,crs=geoid_crs)
    dem = folder / "orthometric.tif"
    values = np.full((5, 5), 100.0)
    values[0, 0] = -32767.0
    values[1, 1] = 0.0  # Zero elevation is not automatically missing data.
    transform = Affine(.01, 0, 85.0, 0, -.01, 29.0)
    write_raster(dem, values, transform)
    output = folder / "ellipsoid.tif"
    report = converter.convert_dem(dem, geoid, output)
    with rasterio.open(output) as raster:
        actual = raster.read(1)
        assert raster.transform == transform
        assert raster.crs == rasterio.crs.CRS.from_epsg(4326)
        assert raster.compression is None
        assert actual[0, 0] == -32767.0
        assert actual[1, 1] == undulation
        assert actual[2, 2] == 100.0 + undulation
        assert raster.tags()["VERTICAL_DATUM"] == "WGS84 ellipsoidal height"
    assert report["multiplier"] == 1
    assert report["nodata_pixels"] == 1
    assert report["max_proj_vs_independent_error_m"] < 1e-8


def test_independent_bilinear_interpolation_uses_grid_pixel_centers(tmp_path):
    path = tmp_path / "fake_egm08.tif"
    transform = Affine(.1, 0, 84.5, 0, -.1, 29.5)
    rows, cols = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
    lon = 84.5 + (cols + .5) * .1
    lat = 29.5 - (rows + .5) * .1
    values = 3 * lon + 2 * lat - 230
    write_raster(path, values, transform, geoid=True)
    longitude, latitude = np.array([85.0, 85.07, 85.12]), np.array([29.0, 28.96, 28.83])
    with rasterio.open(path) as dataset:
        actual = converter.bilinear_geoid(dataset, longitude, latitude)
    np.testing.assert_allclose(actual, 3 * longitude + 2 * latitude - 230, atol=2e-6)


def test_wrong_datum_or_missing_grid_cannot_silently_fall_back(tmp_path):
    with pytest.raises(ValueError, match="EGM2008"):
        converter.convert_dem(tmp_path / "dem.tif", tmp_path / "absent.tif", tmp_path / "out.tif", source_vertical_datum="EGM96")
    with pytest.raises(FileNotFoundError, match="Mandatory local"):
        converter.mandatory_geoid_transform(tmp_path / "absent.tif")


def test_independent_interpolation_rejects_nodata_or_out_of_coverage(tmp_path):
    path = tmp_path / "fake_egm08.tif"
    values = np.ones((20, 20))
    values[5, 5] = -32767
    write_raster(path, values, Affine(.1, 0, 84.5, 0, -.1, 29.5), geoid=True)
    with rasterio.open(path) as dataset:
        with pytest.raises(ValueError, match="nodata"):
            converter.bilinear_geoid(dataset, 85.05, 28.95)
        with pytest.raises(ValueError, match="coverage"):
            converter.bilinear_geoid(dataset, 90.0, 20.0)


def test_non_geographic_dem_is_rejected(tmp_path):
    geoid = tmp_path / "fake_egm08.tif"
    write_raster(geoid, np.ones((20, 20)), Affine(.1, 0, 84.5, 0, -.1, 29.5), geoid=True)
    dem = tmp_path / "projected.tif"
    write_raster(dem, np.ones((5, 5)), Affine(10, 0, 331000, 0, -10, 3100000), crs="EPSG:32645")
    with pytest.raises(ValueError, match="geographic EPSG"):
        converter.convert_dem(dem, geoid, tmp_path / "out.tif")
