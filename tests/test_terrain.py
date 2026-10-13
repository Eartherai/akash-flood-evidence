import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from akash_flood.terrain import NODATA, align_slope, slope_degrees, tile_urls


def test_planar_slope_uses_metre_spacing_and_excludes_edges():
    y,x = np.mgrid[:10,:12]
    z = (x*30*.5 + y*30*.25).astype(float)
    slope = slope_degrees(z, 30, 30)
    assert np.allclose(slope[1:-1,1:-1], np.degrees(np.arctan(np.hypot(.5,.25))))
    assert np.all(slope[0] == NODATA)
    assert np.all(slope[:,-1] == NODATA)


def test_missing_neighbor_invalidates_derivative():
    z = np.zeros((9,9))
    z[4,4] = np.nan
    slope = slope_degrees(z, 30, 30)
    assert np.all(slope[3:6,3:6] == NODATA)
    assert slope[2,2] == 0


def test_tile_boundaries_and_hemisphere_names():
    urls = tile_urls([-1.1,-1.1,-1,-1])
    assert len(urls) == 1 and 'S02_00_W002' in urls[0]
    assert len(tile_urls([85,28,86,29])) == 1


def test_slope_alignment_uses_reference_grid_and_preserves_nodata(tmp_path):
    profile = dict(driver='GTiff',height=4,width=4,count=1,dtype='float32',
                   crs='EPSG:32645',transform=from_origin(340000,3120000,30,30),nodata=NODATA)
    slope = np.full((4,4), 12, dtype=np.float32); slope[1,1] = NODATA
    with rasterio.open(tmp_path/'slope.tif','w',**profile) as ds:
        ds.write(slope,1); ds.update_tags(units='degrees')
    profile.update(width=12,height=12,transform=from_origin(340000,3120000,10,10))
    with rasterio.open(tmp_path/'ref.tif','w',**profile) as ds:
        ds.write(np.ones((12,12),dtype=np.float32),1)
    align_slope(tmp_path/'slope.tif',tmp_path/'ref.tif',tmp_path/'aligned.tif')
    with rasterio.open(tmp_path/'aligned.tif') as ds:
        result=ds.read(1)
        assert ds.shape == (12,12) and ds.transform == profile['transform']
        assert np.all(result[3:6,3:6] == NODATA)
        assert result[0,0] == 12
    with pytest.raises(ValueError, match='overwrite'):
        align_slope(tmp_path/'slope.tif',tmp_path/'ref.tif',tmp_path/'aligned.tif')
