"""A resumed acquisition must preserve exact scene and native-crop identity."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio.windows import Window

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('optical_crop_builder',ROOT/'scripts/build_s2_debris_candidates.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


@pytest.fixture
def saved(tmp_path):
    asset={'href':'https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/sentinel-2-c1-l2a/45/R/UM/2026/8/a/B04.tif',
           'proj:transform':[10,0,300000,0,-10,3200040,0,0,1],'proj:shape':[100,100],
           'gsd':10,'file:size':12345,'raster:bands':[{'nodata':0,'scale':.0001,'offset':-.1}]}
    scl=copy.deepcopy(asset);scl.update({'proj:transform':[20,0,300000,0,-20,3200040,0,0,1],'gsd':20})
    pair={name:{'id':name,'properties':{'proj:epsg':32645},'assets':{'red':asset,'scl':scl}} for name in ('before','after')}
    items={}
    for name,item in pair.items():
        path=tmp_path/(name+'.stac.json');mod.dump(path,item);items[name]=mod.record(path)
    folder=tmp_path/'before';folder.mkdir();crop=folder/'red-dn.tif'
    with rasterio.open(crop,'w',driver='GTiff',width=4,height=4,count=1,dtype='uint16',
                       crs='EPSG:32645',transform=Affine(*asset['proj:transform'][:6]),nodata=0) as ds:
        ds.write(np.full((4,4),5000,dtype='uint16'),1)
    mod.dump(tmp_path/'protocol.json',{'schema':'akash-s2-surface-change-protocol-v1','execute_requested':True,'source_items':items})
    mod.dump(tmp_path/'failed.json',{'status':'failed','received_body_bytes':4,'requests':[
        {'url':asset['href'],'file_bytes':12345,'etag':'fixed','status':206,'sha256':'a'*64}]})
    return tmp_path,pair,Window(0,0,2,2),(CRS.from_epsg(32645),Affine(*scl['proj:transform'][:6]),2,2)


def test_reuse_is_explicitly_later_hash_capture_and_exact_radiometric_replay(saved,tmp_path):
    folder,pair,window,grid=saved;audit=mod.prepare_reuse(folder,pair)
    assert 'after the failed run' in audit['hash_capture_scope']
    out=tmp_path/'new';out.mkdir()
    x,receipt=mod.import_crop(pair['before'],'red',window,grid,audit['crops']['before']['red'],out)
    assert x.shape==(2,2) and np.allclose(x,.4)
    assert receipt['source_object_full_checksum_verified'] is False
    assert receipt['crop']['sha256']==audit['crops']['before']['red']['crop']['sha256']


def test_changed_selected_scene_cannot_import_an_old_crop(saved):
    folder,pair,_,_=saved;pair=copy.deepcopy(pair);pair['before']['id']='different'
    with pytest.raises(ValueError,match='source item'):mod.prepare_reuse(folder,pair)


def test_changed_crop_bytes_are_rejected_after_replay_freeze(saved,tmp_path):
    folder,pair,window,grid=saved;audit=mod.prepare_reuse(folder,pair)
    entry=audit['crops']['before']['red'];Path(entry['crop']['path']).write_bytes(b'altered')
    with pytest.raises(ValueError,match='checksum changed'):
        mod.import_crop(pair['before'],'red',window,grid,entry,tmp_path)


def test_half_cell_crop_shift_is_rejected_even_with_new_hash(saved,tmp_path):
    folder,pair,window,grid=saved;path=folder/'before/red-dn.tif'
    with rasterio.open(path,'r+') as ds:ds.transform=ds.transform*Affine.translation(.5,0)
    audit=mod.prepare_reuse(folder,pair)
    with pytest.raises(ValueError,match='crop grid'):
        mod.import_crop(pair['before'],'red',window,grid,audit['crops']['before']['red'],tmp_path)


def test_inconsistent_remote_object_identity_refuses_import(saved):
    folder,pair,_,_=saved;failure=json.loads((folder/'failed.json').read_text())
    other=copy.deepcopy(failure['requests'][0]);other['etag']='changed';failure['requests'].append(other)
    mod.dump(folder/'failed.json',failure)
    with pytest.raises(ValueError,match='consistent saved source'):mod.prepare_reuse(folder,pair)
