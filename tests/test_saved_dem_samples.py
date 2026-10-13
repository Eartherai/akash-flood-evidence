from pathlib import Path
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_saved_dem_samples as checker


def make_raster(tmp_path, nodata_at=None):
    p = tmp_path / "dem.tif"
    rows, cols = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    values = (1000 + rows * 100 + cols * 10).astype("float32")
    if nodata_at:
        values[nodata_at] = -32767
    with rasterio.open(p, "w", driver="GTiff", width=5, height=5, count=1,
                       dtype="float32", crs="EPSG:4326", nodata=-32767,
                       transform=from_origin(85, 28, .01, .01)) as ds:
        ds.write(values, 1); ds.update_tags(AREA_OR_POINT="Point")
    return p


def test_independent_sampler_uses_nominal_centers_without_an_extra_half_cell(tmp_path):
    with rasterio.open(make_raster(tmp_path)) as ds:
        lon, lat = ds.xy(2, 3)
        assert checker.bilinear(ds, lon, lat) == pytest.approx(1230, abs=1e-8)


def test_independent_sampler_matches_analytic_plane_at_fractional_coordinates(tmp_path):
    with rasterio.open(make_raster(tmp_path)) as ds:
        lon, lat = ds.transform * (1.75 + .5, 2.25 + .5)
        assert checker.bilinear(ds, lon, lat) == pytest.approx(1242.5, abs=1e-8)


def test_missing_source_cell_refused_instead_of_using_neighbor_or_zero(tmp_path):
    with rasterio.open(make_raster(tmp_path, (2, 2))) as ds:
        lon, lat = ds.transform * (1.5 + .5, 1.5 + .5)
        with pytest.raises(ValueError, match="source nodata"):
            checker.bilinear(ds, lon, lat)


def test_sampling_beyond_bilinear_support_refused(tmp_path):
    with rasterio.open(make_raster(tmp_path)) as ds:
        with pytest.raises(ValueError, match="interpolation support"):
            checker.bilinear(ds, 84, 29)


def test_saved_dem_checker_preserves_prior_diagnostic(tmp_path):
    out = tmp_path / "prior.json"; out.write_text("old evidence")
    with pytest.raises(FileExistsError):
        checker.compare(tmp_path / "absent.json", tmp_path / "absent-terrain", out)
    assert out.read_text() == "old evidence"
