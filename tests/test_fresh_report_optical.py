"""The optional optical view must not hide unknowns or claim debris accuracy."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fresh_report_optical',ROOT/'scripts/build_fresh_report.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def save(path,value):path.write_text(json.dumps(value))


@pytest.fixture
def optical(tmp_path):
    transform=Affine(20,0,300000,0,-20,100000)
    labels=np.array([[1,0,255],[255,255,255]],dtype='uint8')
    aoi=np.ones((2,3),dtype='uint8');valid=(labels!=255).astype('uint8')
    scl=np.where(valid==1,4,9).astype('uint8')
    files={}
    for name,array in {'surface-change-candidates.tif':labels,'aoi-mask.tif':aoi,'valid-mask.tif':valid,
                       'before-scl.tif':scl,'after-scl.tif':scl}.items():
        file=tmp_path/name
        with rasterio.open(file,'w',driver='GTiff',height=2,width=3,count=1,dtype='uint8',
                           crs='EPSG:32745',transform=transform,nodata=255 if name.startswith('surface') else 0) as ds:ds.write(array,1)
        files[name]=mod.record(file)
    catalog=tmp_path/'catalog.json';save(catalog,{})
    protocol=tmp_path/'protocol.json';save(protocol,{'catalog':mod.record(catalog),'source_items':{}})
    transport=tmp_path/'transport.json';save(transport,{'requests':[]})
    path=tmp_path/'optical-manifest.json'
    case={'bbox':[1,2,3,4],'event_date':'2026-08-26'}
    manifest={'schema':'akash-s2-surface-change-v1','status':'completed',**case,
              'reference_labels_loaded':False,'gpu_used':False,'local_debris_accuracy_established':False,
              'partial_exploration_is_original_gate_pass':False,'scene_date_alignment_resampling_used':False,
              'outputs':files,'protocol':mod.record(protocol),'transport':mod.record(transport),
              'source_crops':{n:{'scl':{'crop':files[n+'-scl.tif']}} for n in ('before','after')},
              'grid':{'shape':[2,3],'transform':list(transform),'crs':'EPSG:32745'},
              'counts':{'aoi_pixels':6,'joint_assessed_pixels':2,'unknown_pixels':4,'candidate_pixels':1},
              'candidate_area_m2':400,'joint_assessed_fraction':2/6,'network_received_body_bytes':0,
              'raw_scl_single_scene_clear':{n:{'raw_scl_clear_pixels':2,'fraction':2/6,'original70percentgate_passes':False} for n in ('before','after')},
              'source_dates':{'before':'2026-08-24','after':'2026-08-27'},'scope':'partial surface-change candidates','limitations':[]}
    save(path,manifest)
    return path,{'case':case,'grid':{'epsg':32745}}


def test_optical_view_recounts_unknowns_and_generates_map_from_native_labels(optical):
    path,water=optical;facts,rgb=mod.read_optical(path,water)
    assert facts['counts']['unknown_pixels']==4 and facts['candidate_area_m2']==400
    assert rgb[0,0].tolist()==[147,78,35] and rgb[1,2].tolist()==[193,200,207]


def test_other_case_is_rejected(optical):
    path,water=optical;water['case']['event_date']='2026-08-28'
    with pytest.raises(ValueError,match='different case'):mod.read_optical(path,water)


def test_unknown_cells_cannot_be_replaced_by_negative_classifications(optical):
    path,water=optical;m=json.loads(path.read_text());file=Path(m['outputs']['surface-change-candidates.tif']['path'])
    with rasterio.open(file,'r+') as ds:
        x=ds.read(1);x[x==255]=0;ds.write(x,1)
    m['outputs']['surface-change-candidates.tif']=mod.record(file);save(path,m)
    with pytest.raises(ValueError,match='unknown/assessed'):mod.read_optical(path,water)


def test_claimed_validated_debris_is_rejected(optical):
    path,water=optical;m=json.loads(path.read_text());m['local_debris_accuracy_established']=True;save(path,m)
    with pytest.raises(ValueError,match='explicitly partial'):mod.read_optical(path,water)


def test_clear_scl_coverage_cannot_be_inflated_to_pass_original_gate(optical):
    path,water=optical;m=json.loads(path.read_text());m['raw_scl_single_scene_clear']['after']={'raw_scl_clear_pixels':6,'fraction':1.,'original70percentgate_passes':True};save(path,m)
    with pytest.raises(ValueError,match='source SCL coverage'):mod.read_optical(path,water)
