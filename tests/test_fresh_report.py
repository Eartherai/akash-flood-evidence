"""Report rejects altered counts, grids, or physical-closure substitutions."""
import importlib.util
import json
from pathlib import Path
import struct
import zlib

import numpy as np
import pytest
import rasterio
from affine import Affine

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fresh_report', ROOT/'scripts/build_fresh_report.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def save(path,value):
    path.write_text(json.dumps(value))


@pytest.fixture
def water(tmp_path):
    aoi=np.ones((2,3),dtype='uint8');before=np.array([[1,0,255],[0,1,0]],dtype='uint8')
    after=np.array([[0,1,255],[0,1,1]],dtype='uint8')
    new,lost,counts=mod.temporal_layers(before,after,aoi==1)
    transform=Affine(10,0,300005,0,-10,100005)
    files={}
    arrays={'aoi-mask.tif':aoi,'valid-mask.tif':(before!=255).astype('uint8'),
            'before-water.tif':before,'after-water.tif':after,
            'new-water-candidates.tif':new,'lost-water-candidates.tif':lost}
    for name,array in arrays.items():
        path=tmp_path/name
        with rasterio.open(path,'w',driver='GTiff',height=2,width=3,count=1,dtype='uint8',
                           crs='EPSG:32745',transform=transform,nodata=255 if 'water' in name else 0) as ds:
            ds.write(array,1)
        files[name]=mod.record(path)
    policy=tmp_path/'policy.json';save(policy,{})
    processing=tmp_path/'run-metadata.json';save(processing,{'status':'completed'})
    path=tmp_path/'water-manifest.json'
    save(path,{'schema':'akash-fresh-water-v1','available':True,'reference_labels_loaded':False,'gpu_used':False,
               'grid':{'epsg':32745,'shape':[2,3],'transform':list(transform),'resolution_m':10,
                       'resampling_used_for_date_alignment':False},'counts':counts,'outputs':files,
               'policy':mod.record(policy),'processing_record':mod.record(processing)})
    return path


def test_report_water_recounts_joint_domain_in_southern_utm(water):
    manifest,_,grid=mod.read_water(water)
    assert grid[0].to_epsg()==32745
    assert manifest['counts']['unknown_pixels']==1
    assert manifest['counts']['new_water_candidate_pixels']==2


def test_report_rejects_checksum_tampering_before_decoding(water):
    m=json.loads(water.read_text());Path(m['outputs']['before-water.tif']['path']).write_bytes(b'altered')
    with pytest.raises(ValueError,match='checksum'):
        mod.read_water(water)


def test_report_rejects_wrong_temporal_total(water):
    m=json.loads(water.read_text());m['counts']['new_water_candidate_pixels']=3;save(water,m)
    with pytest.raises(ValueError,match='recounted'):
        mod.read_water(water)


def test_report_rejects_changed_grid_even_with_new_file_hash(water):
    m=json.loads(water.read_text());file=Path(m['outputs']['after-water.tif']['path'])
    with rasterio.open(file,'r+') as ds:ds.transform=Affine(10,0,300010,0,-10,100005)
    m['outputs']['after-water.tif']=mod.record(file);save(water,m)
    with pytest.raises(ValueError,match='matching'):
        mod.read_water(water)


def test_report_rejects_unknown_replaced_by_zero_on_one_date(water):
    m=json.loads(water.read_text());file=Path(m['outputs']['after-water.tif']['path'])
    with rasterio.open(file,'r+') as ds:
        a=ds.read(1);a[0,2]=0;ds.write(a,1)
    m['outputs']['after-water.tif']=mod.record(file);save(water,m)
    with pytest.raises(ValueError,match='joint assessment'):
        mod.read_water(water)


def test_standard_library_png_retains_native_rgb_cells(tmp_path):
    rgb=np.array([[[0,128,255],[255,128,0]],[[12,34,56],[78,90,12]]],dtype='uint8')
    path=tmp_path/'map.png';mod.write_png(path,rgb)
    b=path.read_bytes();assert b[:8]==b'\x89PNG\r\n\x1a\n'
    length=struct.unpack('>I',b[8:12])[0];assert b[12:16]==b'IHDR'
    assert struct.unpack('>II',b[16:24])==(2,2)
    start=8+12+length;size=struct.unpack('>I',b[start:start+4])[0]
    assert zlib.decompress(b[start+8:start+8+size])==b'\0'+rgb[0].tobytes()+b'\0'+rgb[1].tobytes()
