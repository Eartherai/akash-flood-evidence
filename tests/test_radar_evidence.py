import numpy as np
import pytest
from affine import Affine
from akash_flood.radar_evidence import BANDS, geometry_support, integer_overlap, screen

def scene(value=.1,shape=(9,9)):
    d={name:np.full(shape,40.0) for name in BANDS}
    d.update(Sigma0_VV=np.full(shape,value),Sigma0_VH=np.full(shape,value/2),
             elevation=np.full(shape,2500.0),layoverShadowMask=np.zeros(shape,dtype=np.int8))
    return d

def test_same_lattice_shift_is_copied_exactly_and_half_pixel_is_rejected():
    left=('utm',Affine(10,0,105,0,-10,505),4,5)
    right=('utm',Affine(10,0,105,0,-10,515),5,6)
    a,b,grid=integer_overlap(left,right)
    assert (a.row_off,a.col_off,a.width,a.height)==(0,0,4,5)
    assert (b.row_off,b.col_off,b.width,b.height)==(1,0,4,5)
    assert grid==left
    with pytest.raises(ValueError,match='Subpixel'):integer_overlap(left,('utm',Affine(10,0,110,0,-10,515),5,6))

def test_mask_zero_does_not_make_nodata_or_back_facing_geometry_valid():
    s=scene();s['elevation'][0,0]=0;s['Sigma0_VV'][0,1]=0
    s['localIncidenceAngle'][0,2]=90;s['projectedLocalIncidenceAngle'][0,3]=91
    s['layoverShadowMask'][0,4]=1;s['Sigma0_VH'][0,5]=np.nan
    result=geometry_support(s)
    assert not result[0,:6].any() and result.sum()==75

def test_unknown_mask_codes_rejected():
    s=scene();s['layoverShadowMask'][0,0]=4
    with pytest.raises(ValueError,match='encoding'):geometry_support(s)

def test_unclipped_negative_sample_remains_in_mean_but_negative_mean_cannot_be_logged():
    s=scene();s['Sigma0_VV'][4,4]=-.1
    r=screen(s,s,window=3,minimum_samples=9,edge_buffer_pixels=0)
    assert np.isclose(r['means']['before_VV'][4,4],.7/9)
    s['Sigma0_VV'][:]=-.1
    r=screen(s,s,window=3,minimum_samples=9,edge_buffer_pixels=0)
    assert not r['valid'].any() and np.isnan(r['delta_vv']).all()

def test_darkening_screen_does_not_relabel_preexisting_dark_water():
    bright=scene(.1);dark=scene(.001)
    r=screen(bright,dark,window=3,minimum_samples=9,edge_buffer_pixels=0)
    assert r['valid'].sum()==49 and r['water'].sum()==49 and r['change'].sum()==49
    r=screen(dark,dark,window=3,minimum_samples=9,edge_buffer_pixels=0)
    assert not r['water'].any() and not r['change'].any()

def test_polarization_disagreement_is_not_change_candidate():
    a,b=scene(),scene(.001);b['Sigma0_VH'][:]=1
    r=screen(a,b,window=3,minimum_samples=9,edge_buffer_pixels=0)
    assert r['valid'].sum()==49 and not r['change'].any()

def test_unassessed_aoi_is_counted_and_geometry_holes_do_not_become_zero_change():
    a,b=scene(),scene();a['elevation'][4,4]=0
    r=screen(a,b,window=3,minimum_samples=9,edge_buffer_pixels=0)
    assert r['counts']['assessed_pixels']==40 and r['counts']['unassessed_pixels']==41
    assert np.isnan(r['delta_vv'][4,4])
