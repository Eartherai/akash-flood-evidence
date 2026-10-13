import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from akash_flood.fresh_dem import mosaic_orthometric


def tile(path, values, transform, nodata=-32767):
    with rasterio.open(path,'w',driver='GTiff',height=values.shape[0],width=values.shape[1],count=1,
                       dtype='float32',crs='EPSG:4326',transform=transform,nodata=nodata) as ds:
        ds.write(values.astype('float32'),1)
    return path


def test_adjacent_native_grids_copy_centers_and_signed_heights(tmp_path):
    a=tile(tmp_path/'a.tif',np.full((10,10),-25),Affine(.1,0,0,0,-.1,1))
    b=tile(tmp_path/'b.tif',np.full((10,10),0),Affine(.1,0,1,0,-.1,1))
    out=tmp_path/'m.tif';r=mosaic_orthometric([a,b],[.3,.3,1.7,.7],out)
    assert not r['horizontal_resampling_used']
    assert r['requested_valid_cells']==r['requested_cells']
    with rasterio.open(out) as ds:
        assert set(np.unique(ds.read(1)))=={-25,0}
        assert ds.transform.c/.1==pytest.approx(round(ds.transform.c/.1))
        assert ds.transform.f/.1==pytest.approx(round(ds.transform.f/.1))
        assert ds.tags()['VERTICAL_DATUM']=='EGM2008'


def test_missing_requested_cells_fail_instead_of_filling(tmp_path):
    v=np.ones((10,10));v[5,5]=-32767
    p=tile(tmp_path/'a.tif',v,Affine(.1,0,0,0,-.1,1))
    with pytest.raises(ValueError,match='missing cells'):
        mosaic_orthometric([p],[.3,.3,.7,.7],tmp_path/'out.tif')


def test_conflicting_native_overlap_fails(tmp_path):
    a=tile(tmp_path/'a.tif',np.ones((10,10)),Affine(.1,0,0,0,-.1,1))
    b=tile(tmp_path/'b.tif',np.full((10,10),2),Affine(.1,0,0,0,-.1,1))
    with pytest.raises(ValueError,match='disagree'):
        mosaic_orthometric([a,b],[.3,.3,.7,.7],tmp_path/'out.tif')


def test_mixed_resolution_harmonization_is_explicit_and_coarsest(tmp_path):
    a=tile(tmp_path/'a.tif',np.ones((10,10)),Affine(.1,0,0,0,-.1,1))
    b=tile(tmp_path/'b.tif',np.ones((10,20)),Affine(.05,0,1,0,-.1,1))
    r=mosaic_orthometric([a,b],[.3,.3,1.7,.7],tmp_path/'out.tif')
    assert r['horizontal_resampling_used']
    assert r['angular_spacing_deg']==[.1,.1]
    assert r['operations'][1]['method'].startswith('bilinear')


def test_large_crop_refused_before_allocation(tmp_path):
    a=tile(tmp_path/'a.tif',np.ones((10,10)),Affine(.1,0,0,0,-.1,1))
    with pytest.raises(ValueError,match='pixel budget'):
        mosaic_orthometric([a],[.1,.1,.9,.9],tmp_path/'out.tif',maximum_pixels=10)
