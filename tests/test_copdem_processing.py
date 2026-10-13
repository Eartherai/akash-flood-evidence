import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import process_s1_copdem as pipeline

def make_dem(tmp_path):
    p=tmp_path/'ellipsoid.tif'
    with rasterio.open(p,'w',driver='GTiff',width=20,height=20,count=1,dtype='float32',
                       crs='EPSG:4326',transform=from_origin(85.2,28.4,.02,.02),nodata=-32767) as ds:
        ds.write(np.full((20,20),2500,dtype='float32'),1)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'output':{'path':str(p),'sha256':pipeline.sha256(p),
                                             'vertical_datum':'WGS84 ellipsoidal height'}}))
    return p,manifest

def test_processor_rejects_wrong_height_datum_and_changed_dem_bytes(tmp_path):
    dem,manifest=make_dem(tmp_path)
    p,nodata,_=pipeline.ellipsoid_dem(manifest,(85.3,28.12,85.43,28.32))
    assert p==dem and nodata==-32767
    data=json.loads(manifest.read_text());data['output']['vertical_datum']='EGM2008 orthometric height'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='ellipsoidal'):pipeline.ellipsoid_dem(manifest,(85.3,28.12,85.43,28.32))
    data['output']['vertical_datum']='WGS84 ellipsoidal height';data['output']['sha256']='0'*64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='checksum'):pipeline.ellipsoid_dem(manifest,(85.3,28.12,85.43,28.32))

def test_context_outside_dem_is_refused_instead_of_silent_holes(tmp_path):
    dem,manifest=make_dem(tmp_path)
    with pytest.raises(ValueError,match='does not cover'):pipeline.ellipsoid_dem(manifest,(85.1,28.12,85.43,28.32))

def test_existing_output_refused_before_any_processing(tmp_path):
    dem,manifest=make_dem(tmp_path)
    with pytest.raises(FileExistsError):pipeline.run((85.3,28.12,85.43,28.32),['before'],tmp_path,tmp_path/'not-present.json',manifest)
