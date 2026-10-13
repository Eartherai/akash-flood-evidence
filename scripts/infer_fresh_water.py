"""Apply the unchanged Sen1 CPU checkpoint to a bound generic SNAP case.

This is water-state transfer, not confirmed flood/debris or validated local
accuracy. Uses signed DEM validity declared by the generic SNAP reader.
"""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.features import geometry_mask
from scipy.ndimage import uniform_filter,distance_transform_edt
from shapely.geometry import Polygon,mapping

from akash_flood.flood_model import predict_water,FEATURE_NAMES
from akash_flood.radar_evidence import crop_arrays
from akash_flood.snap_products import SnapProduct,read_snap_product,geometry_support,integer_overlap
from akash_flood.water_inference import positive_db,temporal_layers
from infer_trained_water import require_record,record,sha,write_raster,TRAINING_PROTOCOL_SHA

ROOT=Path(__file__).resolve().parents[1]
POLICY={
    'threshold':.5,'minimum_joint_positive_geometry_samples_7x7':40,'coverage_edge_buffer_m':200,
    'features':'Unchanged frozen Sen1 clipped-dB feature function from individual calibrated powers; no mean-power screening input',
    'height_validity':'Finite signed ellipsoidal DEM excluding declared nodata; no positive-height assumption',
    'unknown':'Unsupported pixels255; no epsilon substitutions or predicted flood absence',
    'context':'Full matched processing grid; AOI restriction after features; no interpolation for date alignment',
    'changes':'Gained and lost water-state predictions are candidates, never confirmed event flooding',
}


def footprint(product):
    valid=np.ones(product.arrays['elevation'].shape,dtype=bool)
    for name in ('Sigma0_VV','Sigma0_VH','elevation'):
        x=product.arrays[name];valid &= np.isfinite(x)
        nodata=product.band_metadata[name]['nodata']
        if nodata is not None:valid &= x!=nodata
    return valid


def infer(model,before,after,aoi):
    support=geometry_support(before,positive_power=True)&geometry_support(after,positive_power=True)
    if aoi.shape!=support.shape or aoi.dtype!=bool:
        raise ValueError('Require matching boolean AOI')
    count=uniform_filter(support.astype('float64'),size=7,mode='constant',cval=0)*49
    coverage=footprint(before)&footprint(after)
    distance=distance_transform_edt(np.pad(coverage,1,constant_values=False))[1:-1,1:-1]
    assessed=aoi&support&(count>=40-1e-8)&(distance>=20)
    result={}
    for date,p in [('before',before),('after',after)]:
        db,_=positive_db(p.arrays)
        score,labels=predict_water(model,db,support)
        labels[~assessed]=255;score[~assessed]=np.nan
        result[date+'_labels']=labels;result[date+'_score']=score
    result['new'],result['lost'],result['counts']=temporal_layers(result['before_labels'],result['after_labels'],aoi)
    result['assessed']=assessed
    return result


def cropped(product,window,grid):
    return SnapProduct(crop_arrays(product.arrays,window),grid,product.attributes,product.band_metadata)


def main():
    import joblib,sklearn
    from threadpoolctl import threadpool_limits
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processing-record',type=Path,required=True)
    p.add_argument('--model-freeze',type=Path,default=ROOT/'outputs/sen1-cpu-v1/model-evaluation-freeze.json')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Require fresh model-output directory')
    start=time.perf_counter()
    run=json.loads(a.processing_record.read_text());freeze=json.loads(a.model_freeze.read_text())
    if run.get('schema')!='akash-fresh-s1-processing-v1' or run.get('status')!='completed' or not run.get('completed_at'):
        raise ValueError('Generic raw processing has not completed')
    if run['processing'].get('SRTM_used') is not False or run['processing'].get('externalDEMApplyEGM') is not False or not run.get('dem_reader_patch'):
        raise ValueError('Require declared corrected Copernicus DEM processing')
    for item in run['tool_sources']:
        require_record(Path(item['path']),item)
    require_record(Path(run['source_plan']['path']),run['source_plan'])
    for item in freeze['code']:
        require_record(Path(item['path']),item)
    require_record(Path(freeze['protocol']['path']),freeze['protocol'])
    protocol=json.loads(Path(freeze['protocol']['path']).read_text())
    if freeze['protocol']['sha256']!=TRAINING_PROTOCOL_SHA or protocol['features']!=list(FEATURE_NAMES) or freeze['threshold']!=.5 or sklearn.__version__!='1.7.2':
        raise ValueError('Frozen checkpoint feature/library contract differs')
    model_path=Path(freeze['model']['path']);require_record(model_path,freeze['model'])
    products={};epsg=run['projection']['epsg']
    for date in ('before','after'):
        scene=run['scenes'][date]
        if scene.get('status')!='completed':raise ValueError('Missing completed scene')
        artifacts=scene['artifacts'];known={Path(i['path']).resolve():i for i in artifacts}
        dim=Path(a.processing_record).parent/(date+'-terrain.dim')
        for path in [dim]+[x for x in dim.with_suffix('.data').glob('*') if x.suffix in ('.img','.hdr')]:
            require_record(path,known[path.resolve()])
        products[date]=read_snap_product(dim,expected_epsg=epsg)
    for key in ('REL_ORBIT','PASS','MISSION','ACQUISITION_MODE'):
        if products['before'].attributes.get(key)!=products['after'].attributes.get(key):
            raise ValueError('Processed date metadata differs')
    w,v,grid=integer_overlap(products['before'].grid,products['after'].grid)
    before=cropped(products['before'],w,grid);after=cropped(products['after'],v,grid)
    west,south,east,north=run['case']['bbox'];tx=Transformer.from_crs(4326,grid[0],always_xy=True)
    perimeter=[]
    for x,y in [((west,south),(east,south)),((east,south),(east,north)),((east,north),(west,north)),((west,north),(west,south))]:
        for f in np.linspace(0,1,41,endpoint=False):perimeter.append(tx.transform(x[0]+f*(y[0]-x[0]),x[1]+f*(y[1]-x[1])))
    aoi=geometry_mask([mapping(Polygon(perimeter))],out_shape=before.arrays['elevation'].shape,transform=grid[1],invert=True,all_touched=False)
    # Save policy before executing inference; this is a new engineering transfer
    # policy. Original Rasuwa and all observed holdout results are preserved.
    a.out.mkdir(parents=True)
    policy_record={**POLICY,'recorded_before_inference_at':datetime.now(timezone.utc).isoformat(),
                   'training_protocol_sha256':TRAINING_PROTOCOL_SHA,'model_sha256':freeze['model']['sha256'],
                   'case_id':run['case']['case_id'],'source_processing_sha256':sha(a.processing_record),
                   'known_research_context':'Sen1 final baseline metrics and prior Rasuwa screening outcomes are already known; no local validation claim',
                   'reference_labels_loaded':False,'retuning':False}
    (a.out/'policy.json').write_text(json.dumps(policy_record,indent=2)+'\n')
    model=joblib.load(model_path)
    with threadpool_limits(limits=4):result=infer(model,before,after,aoi)
    for date in ('before','after'):
        write_raster(a.out/(date+'-water.tif'),result[date+'_labels'],grid,255)
        write_raster(a.out/(date+'-water-score.tif'),np.where(result['assessed'],result[date+'_score'],-9999).astype('float32'),grid,-9999)
    for name in ('new','lost'):write_raster(a.out/(name+'-water-candidates.tif'),result[name],grid,255)
    write_raster(a.out/'aoi-mask.tif',aoi.astype('uint8'),grid,0)
    write_raster(a.out/'valid-mask.tif',result['assessed'].astype('uint8'),grid,0)
    report={
        'schema':'akash-fresh-water-v1','available':True,'scope':'Experimental water-state transfer; permanent and flood water inseparable; local hazard accuracy unvalidated',
        'case':run['case'],'source_dates':{date:run['scenes'][date]['source_datetime'] for date in ('before','after')},
        'counts':result['counts'],'grid':{'crs':str(grid[0]),'epsg':epsg,'transform':list(grid[1]),'shape':[grid[3],grid[2]],'resolution_m':10,'resampling_used_for_date_alignment':False},
        'model':{**freeze['model'],'threshold':.5,'freeze':record(a.model_freeze),'scikit_learn_version':sklearn.__version__,'classifier_score_calibrated':False},
        'policy':record(a.out/'policy.json'),'processing_record':record(a.processing_record),
        'code':[record(Path(__file__)),record(ROOT/'src/akash_flood/snap_products.py'),record(ROOT/'src/akash_flood/flood_model.py'),record(ROOT/'src/akash_flood/water_inference.py')],
        'outputs':{x.name:record(x) for x in a.out.glob('*.tif')},
        'label_encoding':{'0':'predicted nonwater','1':'predicted water','255':'unknown'},
        'reference_labels_loaded':False,'gpu_used':False,'wall_seconds':time.perf_counter()-start,
        'limitations':['Model transfer and absolute geolocation unvalidated; all candidate changes are provisional.',
                       'Signed DEM values use declared nodata; ocean nodata remains enabled in the processing graph.',
                       'No debris detection, road damage, observed closures or operational safety assertion.',
                       '10m grid spacing with 3x3/7x7 model context; no calibrated uncertainty percentage.',
                       'This is generic inference infrastructure, not proof of an unseen arbitrary-case raw-to-report run.']}
    (a.out/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'counts':result['counts'],'wall_seconds':report['wall_seconds'],'manifest':record(a.out/'manifest.json')},indent=2))


if __name__=='__main__':main()
