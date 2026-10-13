import importlib.util
from pathlib import Path
import sys

import numpy as np
from affine import Affine

from akash_flood.snap_products import SnapProduct,BANDS

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('fresh_water',ROOT/'scripts/infer_fresh_water.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Model:
    classes_=np.array([0,1])
    def predict_proba(self,x):return np.tile([.1,.9],(len(x),1))


def product():
    arrays={name:np.full((101,101),35,dtype='float32') for name in BANDS}
    arrays['Sigma0_VV'].fill(.01);arrays['Sigma0_VH'].fill(.001)
    arrays['elevation'].fill(-25);arrays['layoverShadowMask'].fill(0)
    metadata={name:{'nodata':-32767 if name=='elevation' else 0,'unit':None} for name in BANDS}
    return SnapProduct(arrays,('EPSG:32721',Affine(10,0,500005,0,-10,7000005),101,101),{},metadata)


def test_signed_elevation_is_assessed_and_declared_missing_remains_unknown():
    b,a=product(),product();a.arrays['elevation'][50,50]=-32767
    r=module.infer(Model(),b,a,np.ones((101,101),dtype=bool))
    assert r['before_labels'][50,50]==255
    assert r['before_labels'][50,20]==1
    assert r['counts']['assessed_pixels']>0
    assert r['counts']['new_water_candidate_pixels']==0


def test_negative_raw_power_not_turn_into_model_water():
    b,a=product(),product();a.arrays['Sigma0_VV'][25,25]=-.001
    r=module.infer(Model(),b,a,np.ones((101,101),dtype=bool))
    assert r['after_labels'][25,25]==255 and np.isnan(r['after_score'][25,25])
    assert r['lost'][25,25]==255 and r['new'][25,25]==255
