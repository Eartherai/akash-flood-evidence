"""Save actual radar observations, candidate screens, coverage and provenance.

No official reference geometry or label is loaded by this program.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Polygon, mapping

from akash_flood.radar_evidence import RECIPE, read_snap, integer_overlap, crop_arrays, screen

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def write(path,values,grid,nodata):
    crs,transform,width,height=grid
    with rasterio.open(path,'w',driver='GTiff',width=width,height=height,count=1,dtype=values.dtype,
                       crs=crs,transform=transform,nodata=nodata,compress='deflate',tiled=True) as ds:
        ds.write(values,1)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,required=True);parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--bbox',nargs=4,type=float,default=RECIPE['aoi_bbox'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--processing-record',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Use a fresh output directory; preserve earlier evidence')
    west,south,east,north=args.bbox
    if not -180<west<east<180 or not -90<south<north<90:raise ValueError('Invalid AOI bounds')
    start=time.perf_counter();args.out.mkdir(parents=True)
    processing_record=json.loads(args.processing_record.read_text())
    if 'completed_at' not in processing_record:raise ValueError('Preprocessing run has not completed')
    for scene,dim in [('before',args.before),('after',args.after)]:
        artifacts=processing_record['scenes'][scene]['artifacts']
        records={Path(item['path']).resolve():item for item in artifacts}
        required=[dim]+[p for p in dim.with_suffix('.data').glob('*') if p.suffix in ('.img','.hdr')]
        for path in required:
            item=records.get(path.resolve())
            if item is None or item['sha256']!=sha(path):
                raise ValueError('Radar input is not bound to the declared completed processing run: '+str(path))
    recipe=dict(RECIPE,aoi_bbox=args.bbox)
    recipe_path=args.out/'recipe.json';recipe_path.write_text(json.dumps(recipe,indent=2)+'\n')
    before,left,bmeta=read_snap(args.before);after,right,ameta=read_snap(args.after)
    for key in ('REL_ORBIT','PASS','MISSION','ACQUISITION_MODE'):
        if bmeta.get(key)!=ameta.get(key):raise ValueError('Before/after metadata differs: '+key)
    w,v,grid=integer_overlap(left,right);before=crop_arrays(before,w);after=crop_arrays(after,v)
    transformer=Transformer.from_crs(4326,grid[0],always_xy=True)
    perimeter=[]
    for edge in [[(west,south),(east,south)],[(east,south),(east,north)],[(east,north),(west,north)],[(west,north),(west,south)]]:
        for fraction in np.linspace(0,1,41,endpoint=False):
            lon=edge[0][0]+fraction*(edge[1][0]-edge[0][0]);lat=edge[0][1]+fraction*(edge[1][1]-edge[0][1])
            perimeter.append(transformer.transform(lon,lat))
    aoi=geometry_mask([mapping(Polygon(perimeter))],out_shape=before['Sigma0_VV'].shape,
                      transform=grid[1],invert=True,all_touched=False)
    result=screen(before,after,aoi=aoi,window=recipe['smooth_window_pixels'],
                  minimum_samples=recipe['minimum_joint_geometry_samples'],
                  edge_buffer_pixels=recipe['coverage_edge_buffer_m']//10)
    write(args.out/'aoi-mask.tif',aoi.astype('uint8'),grid,0)
    write(args.out/'valid-mask.tif',result['valid'].astype('uint8'),grid,0)
    write(args.out/'geometry-support.tif',(result['geometry_support']&aoi).astype('uint8'),grid,0)
    for key,array in result['db'].items():
        write(args.out/(key.lower()+'-mean-db.tif'),np.where(result['valid'],array,-9999).astype('float32'),grid,-9999)
    for name in ('delta_vv','delta_vh'):
        write(args.out/(name.replace('_','-')+'-db.tif'),np.where(result['valid'],result[name],-9999).astype('float32'),grid,-9999)
    for date,scene in [('before',before),('after',after)]:
        # Raw sigma0 observations for the independently defined registration diagnostic.
        values=np.where(result['valid'],scene['Sigma0_VV'],-9999).astype('float32')
        write(args.out/(date+'-vv-power.tif'),values,grid,-9999)
    for name in ('change','water'):
        values=np.where(result['valid'],result[name],255).astype('uint8')
        write(args.out/(name+'-candidates.tif'),values,grid,255)
    shared_dem=(before['elevation']>0)&(after['elevation']>0)
    max_dem_difference=float(np.max(np.abs(before['elevation'][shared_dem]-after['elevation'][shared_dem]))) if shared_dem.any() else None
    wrapper=json.loads((ROOT/'data/s1-snap-native/derived-wrapper-index.json').read_text())
    runtime=json.loads((ROOT/'data/snap-runtime/runtime-download-manifest.json').read_text())
    sources={item['scene']:{'item_id':item['source_item_id'],'derived_manifest_sha256':item['derived_manifest_sha256'],
                          'original_manifest_sha256':item['original_manifest_sha256']} for item in wrapper}
    auxiliary=[]
    for p in (ROOT/'data/snap-runtime/user-cache/auxdata/dem').rglob('*.zip'):
        auxiliary.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
    source_dates={}
    for scene in ('before','after'):
        stac=json.loads((ROOT/'data/s1-snap-native/metadata'/(scene+'.stac.json')).read_text())
        source_dates[scene]=stac['properties']['datetime']
    report={'status':'actual observations and conservative candidate screens; not confirmed hazard labels',
            'created_at':datetime.now(timezone.utc).isoformat(),'available':True,'requested_bbox':args.bbox,
            'source_dates':source_dates,'source_items':sources,'units':'Sigma0 linear power; 10 log10(mean power) and after-minus-before dB',
            'recipe':recipe,'recipe_sha256':sha(recipe_path),'code_sha256':sha(ROOT/'src/akash_flood/radar_evidence.py'),
            'builder_sha256':sha(Path(__file__)),'counts':result['counts'],
            'grid':{'crs':str(grid[0]),'transform':list(grid[1]),'shape':[grid[3],grid[2]],
                    'resolution_m':10,'corner_origin_modulo_resolution_m':5,'resampling_used_for_alignment':False,
                    'before_window':list(w.flatten()),'after_window':list(v.flatten()),
                    'shared_positive_dem_pixels':int(shared_dem.sum()),'maximum_shared_dem_difference_m':max_dem_difference},
            'preprocessing':{'engine':'ESA SNAP 14.0.0 native ARM Java CPU' + ('; local external-DEM sampling correction overlay' if processing_record.get('dem_reader_patch') else ''),
                             'thermal_noise_removed':True,'negative_values_clipped':False,'calibrated':'sigma0',
                             'geometric_terrain_correction':True,'dem':processing_record['processing'].get('dem',processing_record['processing']['terrain_correction']),
                             'orbit':'source embedded state vectors; no external precise-orbit update',
                             'radiometric_terrain_flattening':False,'runtime_manifest':runtime,
                             'auxiliary_inputs':auxiliary if processing_record['processing'].get('SRTM_used',True) else [],
                             'processing_record_path':str(args.processing_record),
                             'processing_record_sha256':sha(args.processing_record),
                             'dem_sampling_correction':processing_record.get('dem_reader_patch'),
                             'processing_verification':'External DEM sampler center correction applied; review independent numerical diagnostic' if processing_record.get('dem_reader_patch') else 'External DEM sampling verification incomplete; engineering/provisional evidence',
                             'track_input_eligibility':'Copernicus DEM' if processing_record['processing'].get('SRTM_used') is False else 'SRTM engineering run; not a conforming Track B entry'},
            'inputs':[{'path':str(path),'sha256':sha(path),'bands':{
                p.name:{'sha256':sha(p),'bytes':p.stat().st_size}
                for p in path.with_suffix('.data').glob('*') if p.suffix in ('.img','.hdr')}}
                for path in (args.before,args.after)],
            'outputs':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in args.out.glob('*.tif')},
            'reference_labels_loaded':False,'gpu_used':False,'account_used':False,
            'wall_seconds':time.perf_counter()-start,
            'limitations':['A radar change is not a landslide/flood diagnosis. Darkening is not confirmed water.',
                           'No road closures are assigned from these candidate pixels.',
                           'Seven-pixel spatial averaging reduces independent spatial detail; 10 m is the grid spacing.',
                           'Masks exclude absent coverage, modeled layover/shadow and invalid viewing angles; remaining validity is provisional.',
                           'Noise-subtracted negative samples remain in averages; nonpositive averages are not logged.',
                           'Terrain-mask context and image-content registration require separate recorded checks.',
                           'Same-grid/DEM agreement does not establish absolute geolocation accuracy.']}
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'counts':report['counts'],'wall_seconds':report['wall_seconds'],'output':str(args.out)},indent=2))

if __name__=='__main__':main()
