"""Geometric radar processing with explicitly ellipsoidal Copernicus DEM.

Uses the same Sentinel-1 noise/calibration/geometric operators as the engineering
SRTM run. No SRTM elevation input, external orbit or terrain-flattening claim.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from process_s1_snap import ROOT, graph, record, sha256, validate_bbox, validate_output, wrapper_manifest
from snap_gpt import command

def ellipsoid_dem(path,bbox):
    manifest=json.loads(path.read_text());output=manifest['output']
    dem=Path(output['path'])
    if not dem.is_absolute():dem=path.parent/dem
    if output['vertical_datum']!='WGS84 ellipsoidal height' or sha256(dem)!=output['sha256']:
        raise ValueError('Require checksum-pinned ellipsoidal Copernicus DEM')
    with rasterio.open(dem) as ds:
        if ds.crs!=rasterio.crs.CRS.from_epsg(4326) or ds.count!=1:
            raise ValueError('External DEM must be one geographic WGS84 band')
        west,south,east,north=bbox
        if not (ds.bounds.left<west<east<ds.bounds.right and ds.bounds.bottom<south<north<ds.bounds.top):
            raise ValueError('DEM does not cover requested processing context')
        nodata=ds.nodata
        if nodata is None or not np.isfinite(nodata):raise ValueError('Explicit finite DEM nodata required')
    return dem,nodata,manifest

def verified_reader_patch(manifest_path,proof_path):
    if not proof_path:
        raise ValueError('A DEM sampler overlay requires its independent JVM proof')
    manifest=json.loads(manifest_path.read_text());proof=json.loads(proof_path.read_text())
    expected={'class':'org/esa/snap/dem/dataio/FileElevationModel.class',
              'old_method':'computeCornerBasedIndex','new_method':'computeIndex','changed_constant_count':1,
              'method_bodies_unchanged':True}
    if any(manifest['scope'].get(key)!=value for key,value in expected.items()):
        raise ValueError('Unexpected DEM sampler correction scope')
    for key in ('vendor_jar','overlay_jar'):
        item=manifest[key]
        if sha256(Path(item['path']))!=item['sha256']:
            raise ValueError('DEM sampler runtime checksum mismatch')
    if (proof['overlay_manifest']['sha256']!=sha256(manifest_path)
            or proof.get('status')!='verified on analytic DEM fixtures'
            or proof.get('geographic_centers_preserved') is not True
            or proof.get('original_runtime_modified') is not False
            or not 0<=proof.get('corrected_max_abs_error_m',float('inf'))<1e-6
            or proof.get('samples_per_jvm',0)<100):
        raise ValueError('Missing or incompatible independent DEM sampler proof')
    return Path(manifest['overlay_jar']['path']),{
        'manifest':record(manifest_path),'overlay':manifest['overlay_jar'],
        'independent_analytic_proof':record(proof_path),'scope':expected,
        'runtime_is_locally_modified':True,'original_runtime_jars_modified':False,
        'ESA_validated_or_endorsed':False,
        'limitations':['Numerical sampling correction only; absolute satellite geolocation and hazard accuracy are not established.']}


def run(bbox,scenes,output,index_path,dem_manifest,patch_manifest=None,patch_proof=None):
    bbox=validate_bbox(bbox)
    if output.exists():raise FileExistsError('Use a fresh output directory; preserve earlier run evidence')
    dem,nodata,conversion=ellipsoid_dem(dem_manifest,bbox)
    overlay,patch_record=verified_reader_patch(patch_manifest,patch_proof) if patch_manifest else (None,None)
    if patch_proof and not patch_manifest:raise ValueError('DEM reader proof requires its overlay manifest')
    sources={scene:wrapper_manifest(index_path,scene) for scene in scenes}
    output.mkdir(parents=True)
    report={'started_at':datetime.now(timezone.utc).isoformat(),'bbox':list(bbox),'scenes':{},
            'source_index':record(index_path),'dem_conversion_manifest':record(dem_manifest),
            'dem':record(dem),'dem_conversion':conversion,
            'tool_sources':[record(Path(__file__)),record(ROOT/'scripts/process_s1_snap.py'),record(ROOT/'scripts/snap_gpt.py'),record(ROOT/'uv.lock')],
            'processing':{'thermal_noise_removed':True,'negative_noise_corrected_powers_clipped':False,
                          'calibration':'sigma0 linear power','terrain_correction':'Range-Doppler geometric correction',
                          'dem':'Copernicus GLO-30 converted from EGM2008 orthometric to WGS84 ellipsoidal height',
                          'externalDEMApplyEGM':False,'SRTM_used':False,
                          'projection':'EPSG:32645 WKT1_GDAL','spacing_m':10,'standard_grid_pixel_center_origin':[0,0],
                          'orbit':'Embedded source state vectors; no applied precise orbit',
                          'radiometric_terrain_flattening':False,'gpt_parallelism':4,'cache':'512M','heap':'6G'},
            'dem_reader_patch':patch_record,
            'classification_or_hazard_claim':False}
    destination=output/'run-metadata.json'
    def save():destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    for scene in scenes:
        manifest,source=sources[scene];xml=output/(scene+'-terrain.xml')
        doc=graph(scene,manifest,output/(scene+'-terrain.dim'),bbox)
        parameters=doc.find('.//node[@id="terrain"]/parameters')
        parameters.find('demName').text='External DEM'
        import xml.etree.ElementTree as ET
        for key,value in {'externalDEMFile':str(dem.resolve()),'externalDEMNoDataValue':str(nodata),'externalDEMApplyEGM':'false'}.items():
            ET.SubElement(parameters,key).text=value
        ET.indent(doc);doc.write(xml,encoding='utf-8',xml_declaration=True)
        argv=command([str(xml.resolve()),'-q','4','-c','512M']);log=output/(scene+'-terrain.log')
        if overlay:
            cp_index=argv.index('-cp')+1
            argv[cp_index]=str(overlay.resolve())+':'+argv[cp_index]
        entry={'source_item_id':source['source_item_id'],'source_manifest':record(manifest),
               'source_wrapper_provenance':record(manifest.parent/'derived-wrapper-provenance.json'),
               'graph':record(xml),'started_at':datetime.now(timezone.utc).isoformat()}
        report['scenes'][scene]=entry;save();start=time.perf_counter()
        with log.open('w') as stream:result=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
        entry.update(exit_code=result.returncode,wall_seconds=time.perf_counter()-start,log=record(log))
        save()
        if result.returncode!=0 or 'done.' not in log.read_text():
            raise RuntimeError('SNAP did not complete successfully; preserved logs and partial output')
        if 'Cannot create layover/shadow' in log.read_text():
            raise RuntimeError('Terrain-mask generation reported missing DEM')
        entry['validation']=validate_output(output,scene)
        entry['artifacts']=[record(p) for p in sorted(output.glob(scene+'-terrain.*')) if p.is_file()]+[
            record(p) for p in sorted((output/(scene+'-terrain.data')).rglob('*')) if p.is_file()]
        entry['finished_at']=datetime.now(timezone.utc).isoformat();save()
        print(scene,'complete',entry['validation']['geometrically_usable_pixels'],'geometry-usable pixels',flush=True)
    report['completed_at']=datetime.now(timezone.utc).isoformat();save()
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bbox',nargs=4,type=float,required=True);parser.add_argument('--scene',choices=['before','after','both'],default='both')
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--dem-manifest',type=Path,required=True)
    parser.add_argument('--dem-reader-patch',type=Path,help='Explicit local external-DEM sampler overlay manifest; original runtime remains untouched')
    parser.add_argument('--dem-reader-proof',type=Path,help='Checksum-bound independent analytic JVM proof for the overlay')
    parser.add_argument('--source-index',type=Path,default=ROOT/'data/s1-snap-native/derived-wrapper-index.json')
    args=parser.parse_args();run(args.bbox,['before','after'] if args.scene=='both' else [args.scene],args.out.resolve(),args.source_index.resolve(),args.dem_manifest.resolve(),args.dem_reader_patch,args.dem_reader_proof)
