"""Local interactive access proofs and a grounded CPU assistant.

Road observations entered here are explicitly hypothetical. This does not turn
the raw calibration crop or official evaluation products into damage evidence.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import math
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlparse

from shapely.geometry import mapping
import numpy as np
import rasterio

from akash_flood.access import build_road_graph
from akash_flood.access_bounds import analyze_access_bounds
from akash_flood.copilot import FactStore, ask_local_model
from akash_flood.radar_copilot import RadarFactStore, generate_situation_report
from akash_flood.osm import named_points
from prove_access_bounds import verify_certificates

ROOT=Path(__file__).resolve().parents[1]
WEB=ROOT/'web'
SOURCE=ROOT/'data/rasuwa-pilot/osm.json'
SOURCE_PROVENANCE=ROOT/'data/rasuwa-pilot/osm.provenance.json'
osm=json.loads(SOURCE.read_text())
source_provenance=json.loads(SOURCE_PROVENANCE.read_text())
source_sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
graph=build_road_graph(osm)
settlements,destinations=named_points(osm)
groups=defaultdict(list)
for u,v,key,data in graph.edges(keys=True,data=True):
    if data['bridge']:
        groups[f"way-{data['way_id']}-nodes-{min(u,v)}-{max(u,v)}"].append(data['edge_id'])
groups={key:sorted(edges) for key,edges in sorted(groups.items())}
edge_data={data['edge_id']:data for u,v,key,data in graph.edges(keys=True,data=True)}
point_data={point['id']:point for point in settlements+destinations}
model_lock=threading.Lock()
cache={}
report_artifacts={}
report_artifact_lock=threading.Lock()
REPORT_ARTIFACT_LIMIT=16
RADAR_PRODUCTION=ROOT/'outputs/radar-evidence-copdem-pixel-corrected/manifest.json'
RADAR_PROVISIONAL=ROOT/'outputs/radar-evidence-copdem/manifest.json'
RADAR_ENGINEERING=ROOT/'outputs/radar-evidence/manifest.json'
RADAR_OVERRIDE=None
TRAINED_WATER=ROOT/'outputs/sen1-water-rasuwa/manifest.json'
TRAINED_WATER_SCHEMA='akash-trained-water-v1'
TRAINED_WATER_SCOPE='Supervised water-state candidates; permanent and flood water inseparable; Rasuwa accuracy not established'
trained_water_cache=None
trained_water_lock=threading.Lock()


class RadarUnavailable(RuntimeError):
    """The source was missing or failed its radar-only manifest checks."""


def radar_source_path():
    # Prefer the conforming DEM run once its manifest exists. Never silently
    # switch back to an engineering source when that production file is invalid.
    if RADAR_OVERRIDE is not None:return RADAR_OVERRIDE
    return next((path for path in [RADAR_PRODUCTION,RADAR_PROVISIONAL,RADAR_ENGINEERING] if path.is_file()),RADAR_PRODUCTION)


def radar_exposure_paths(path):
    if not path.parent.name.startswith('radar-evidence'):return None
    folder=path.parent.with_name(path.parent.name.replace('radar-evidence','radar-exposure',1))
    summary=folder/'summary.json';binding=folder/'source-and-output-manifest.json'
    return (summary,binding) if summary.is_file() and binding.is_file() else None


def radar_store():
    path=radar_source_path()
    try:
        exposure=radar_exposure_paths(path)
        return RadarFactStore.from_path(path,exposure_path=exposure[0],exposure_manifest_path=exposure[1]) if exposure else RadarFactStore.from_path(path)
    except (OSError,ValueError,KeyError,TypeError) as error:
        raise RadarUnavailable('Actual radar facts are unavailable or failed source checks.') from error


def access(observations):
    if not isinstance(observations,dict) or set(observations)-set(groups):
        raise ValueError('Only known source bridge-segment IDs may be observed')
    if any(value not in ['open','closed','unknown'] for value in observations.values()):
        raise ValueError('Observation states must be open, closed or unknown')
    observations={key:value for key,value in observations.items() if value!='unknown'}
    signature=tuple(sorted(observations.items()))
    if signature in cache:return cache[signature]
    if len(observations)>5:raise ValueError('This research comparison permits at most five hypothetical inspections')
    closed={edge for key,value in observations.items() if value=='closed' for edge in groups[key]}
    uncertain={key:edges for key,edges in groups.items() if key not in observations}
    start=time.perf_counter()
    proof=analyze_access_bounds(graph,settlements,destinations,closed,uncertain,max_snap_m=500)
    certificate_checks=verify_certificates(graph,proof)
    proof['provenance']={
        'created_at':datetime.now(timezone.utc).isoformat(),'osm_sha256':source_sha,
        'requested_historical_snapshot':source_provenance['snapshot'],
        'scenario_rule':'Uninspected bridge-tagged physical segments remain uncertain; user observations are hypothetical.',
        'scenario_is_synthetic':True,'hazard_inputs_used':False,'n_graph_nodes':len(graph),
        'n_directed_edges':graph.number_of_edges(),'n_settlements':len(settlements),
        'n_destinations':len(destinations),'n_uncertain_physical_segments':len(uncertain),
        'analysis_seconds':round(time.perf_counter()-start,6),
        'certificate_checks':certificate_checks,
        'limitations':['Synthetic inspection states; no physical passability or flood inference.',
                       'Any mapped destination, not a verified nearest or operative hospital.',
                       'Independent segment states may miss correlated bridge/hazard constraints.',
                       'Nearest-node snapping, missing facilities, network boundary and restrictions remain limitations.']}
    raw=json.dumps(proof,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()
    store=FactStore(proof,source_sha256=hashlib.sha256(raw).hexdigest(),source_path='local computed access proof')
    cache[signature]=(proof,store)
    if len(cache)>32:cache.pop(next(iter(cache)))
    return proof,store


def geometry():
    seen=set();roads=[]
    for u,v,key,data in graph.edges(keys=True,data=True):
        physical=(data['way_id'],min(u,v),max(u,v))
        if physical in seen:continue
        seen.add(physical)
        group=f"way-{data['way_id']}-nodes-{min(u,v)}-{max(u,v)}"
        roads.append({'type':'Feature','geometry':mapping(data['geometry']),
                      'properties':{'way_id':data['way_id'],'bridge':data['bridge'],
                                    'link_id':group if group in groups else None}})
    return {'roads':{'type':'FeatureCollection','features':roads},'settlements':settlements,'destinations':destinations,
            'bbox':source_provenance['bbox'],'snapshot':source_provenance['snapshot'],'source_sha256':source_sha}


def summary(proof):
    rows=[]
    for row in proof['settlements']:
        value={key:row[key] for key in ['id','name','status','reason','snap_node_id','snap_distance_m','baseline_reachable','optimistic_reachable','pessimistic_reachable']}
        value.update(lon=point_data[row['id']]['lon'],lat=point_data[row['id']]['lat'])
        for key in ['path_witness','optimistic_path_witness']:
            witness=row[key]
            if witness:
                value[key]={k:witness[k] for k in ['scenario','destination_id','distance_m','route_objective','path_edge_ids']}
                value[key]['geometry']={'type':'MultiLineString','coordinates':[list(edge_data[e]['geometry'].coords) for e in witness['path_edge_ids']]}
        for key in ['unreachable_certificate','pessimistic_cut_certificate']:
            cut=row[key]
            if cut:
                value[key]={'scenario':cut['scenario'],'source_node_id':cut['source_node_id'],
                            'reachable_nodes_count':len(cut['reachable_node_ids']),
                            'cut_edge_ids':cut['removed_edge_ids'],
                            'geometry':{'type':'MultiLineString','coordinates':[list(edge_data[e]['geometry'].coords) for e in cut['removed_edge_ids']]}}
        rows.append(value)
    candidates=[]
    for item in proof['inspection_outcomes']:
        if not (item['if_open']['resolved_count'] or item['if_closed']['resolved_count']):continue
        candidates.append({**item,'minimum_resolved':min(item['if_open']['resolved_count'],item['if_closed']['resolved_count'])})
    candidates.sort(key=lambda item:(-item['minimum_resolved'],-max(item['if_open']['resolved_count'],item['if_closed']['resolved_count']),item['link_id']))
    return {'status_counts':proof['status_counts'],'rows':rows,'inspection_outcomes':candidates,
            'remaining_uncertain_links':len(proof['uncertain_links']),'provenance':proof['provenance'],
            'scope':'Hypothetical graph experiment. Inspection states are user assumptions, not observations of damage.'}


def source_status():
    result={'road_snapshot':source_provenance['snapshot'],'satellite_stage':'pending','copilot_endpoint':'http://127.0.0.1:8782'}
    path=ROOT/'outputs/s1-raw-crop-proof/provenance.json'
    if path.exists():
        r=json.loads(path.read_text());result['satellite_stage']='raw crop calibrated; noise/terrain/alignment not yet validated'
        result['satellite']={key:r[key] for key in ['status','requested_bbox','relative_orbit','pass','network_bytes_read','wall_seconds','gpu_used','account_used','not_established']}
    radar=radar_evidence()
    if radar['available']:
        result['satellite_stage']=radar['run_role']+'; hazard validation incomplete'
        result['radar']={key:radar[key] for key in ['run_role','source_sha256','source_dates','preprocessing']}
    result['benchmark']={}
    for region in ['rasuwa-development','nepal-west-holdout','nepal-east-holdout']:
        path=ROOT/'outputs/inspection-benchmark'/region/'results.json'
        if path.exists():
            r=json.loads(path.read_text());result['benchmark'][region]={'initial_ambiguous':r['initial_ambiguous'],
                'budget3_unresolved':{name:metrics['3']['mean_unresolved_count'] for name,metrics in r['policies'].items()},
                'scope':r['scope']}
    return result

def radar_evidence():
    path=radar_source_path()
    if not path.is_file():
        return {'available':False,'status':'Actual radar evidence is being processed and checked locally.'}
    try:
        store=radar_store()
        path=Path(store.source_path)
        report=store.manifest
    except (RadarUnavailable,OSError,ValueError):
        return {'available':False,'status':'The radar source record failed checks; no unverified numbers are displayed.'}
    production=path==RADAR_PRODUCTION and store.dem_is_copernicus
    provisional=path==RADAR_PROVISIONAL and store.dem_is_copernicus
    role=('Copernicus DEM candidate screen with corrected source-pixel alignment' if production else
          'Provisional Copernicus DEM engineering screen; source-pixel offset correction pending' if provisional else 'Engineering check only')
    eligibility=('Copernicus DEM matches the listed DEM family. Full submission eligibility and hazard accuracy are not established.' if production else
                 'Provisional engineering evidence: Copernicus DEM matches the listed input family, but a source-pixel alignment correction is pending. Do not treat this run as the final terrain-corrected case study.' if provisional else
                 'Engineering evidence only. This run does not provide a conforming Track B case study with the listed Copernicus DEM input.')
    return {**report,'figure_url':f'/radar-evidence/{store.source_sha256}/overview.png' if path.with_name('overview.png').is_file() else None,
            'source_sha256':store.source_sha256,'run_role':role,'dem_family_matches_track':store.dem_is_copernicus,
            'production_source_active':production,'eligibility_note':eligibility}


def cache_radar_report(answer,store):
    """Keep only completed source-resolved reports, never client questions."""
    if answer.get('status')=='decline':return answer
    ids=('radar_acquisition','radar_coverage','radar_candidates','radar_limits')
    language=answer.get('language')
    claims=answer.get('claims')
    if (answer.get('status')!='answer' or answer.get('report_complete') is not True
        or answer.get('report_type')!='source_resolved_radar_situation_report'
        or answer.get('claim_scope')!='real_radar_candidate_screen_only'
        or answer.get('source_sha256')!=store.source_sha256 or language not in ['en','ne']
        or not isinstance(claims,list) or len(claims)!=4
        or any(not isinstance(claim,dict) for claim in claims)
        or {claim.get('fact_id') for claim in claims}!=set(ids)):
        raise ValueError('Only a complete source-resolved radar report may be exported')
    for claim in claims:
        fact=store.facts[claim['fact_id']]
        if (claim.get('text')!=getattr(fact,language) or claim.get('source_pointer')!=fact.source_pointer
            or claim.get('source_sha256')!=fact.source_sha256 or claim.get('source_path')!=fact.source_path
            or claim.get('scope')!='real_radar_candidate_screen_only'):
            raise ValueError('Report export must retain the exact source clauses and hashes')
    expected=store.resolve({'status':'answer','language':language,'fact_ids':list(ids)},language=language,candidate_ids=ids)
    if answer.get('answer')!=expected['answer']:
        raise ValueError('Report export cannot introduce prose beyond resolved source clauses')
    generated_at=datetime.now(timezone.utc).isoformat()
    token=secrets.token_hex(16)
    url=f'/radar-report/{token}.json'
    source_path=Path(store.source_path)
    run_role=('Copernicus DEM candidate screen with corrected source-pixel alignment' if source_path==RADAR_PRODUCTION and store.dem_is_copernicus else
              'Provisional Copernicus DEM engineering screen; source-pixel offset correction pending' if source_path==RADAR_PROVISIONAL and store.dem_is_copernicus else 'Engineering check only')
    artifact={'generated_at':generated_at,'source_run_role':run_role,
              'source_manifest_counts':store.manifest['counts'],'source_dates':store.manifest['source_dates'],
              'requested_bbox':store.manifest['requested_bbox'],'model_report':answer}
    raw=(json.dumps(artifact,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    entry={'bytes':raw,'filename':f'akash-radar-report-{language}-{store.source_sha256[:12]}.json',
           'source_path':store.source_path,'source_sha256':store.source_sha256,
           'exposure_binding':store.exposure_binding}
    with report_artifact_lock:
        report_artifacts[token]=entry
        while len(report_artifacts)>REPORT_ARTIFACT_LIMIT:report_artifacts.pop(next(iter(report_artifacts)))
    return {**answer,'download_url':url,'report_generated_at':generated_at}


def radar_report_artifact(token):
    if len(token)!=32 or any(char not in '0123456789abcdef' for char in token):return None
    with report_artifact_lock:entry=report_artifacts.get(token)
    if entry is None:return None
    try:store=radar_store()
    except RadarUnavailable:store=None
    if (store is None or entry['source_path']!=store.source_path or entry['source_sha256']!=store.source_sha256
        or entry['exposure_binding']!=store.exposure_binding):
        with report_artifact_lock:report_artifacts.pop(token,None)
        return None
    return entry


def _trained_digest(value):
    return isinstance(value,str) and len(value)==64 and all(char in '0123456789abcdef' for char in value)


def _trained_integer(value):
    return isinstance(value,int) and not isinstance(value,bool) and value>=0


def trained_water_record():
    """Validate saved model outputs without fitting, evaluation or copilot use."""
    global trained_water_cache
    if not TRAINED_WATER.is_file():return None
    raw=TRAINED_WATER.read_bytes();digest=hashlib.sha256(raw).hexdigest();value=json.loads(raw)
    if (not isinstance(value,dict) or value.get('schema')!=TRAINED_WATER_SCHEMA or value.get('available') is not True
        or value.get('scope')!=TRAINED_WATER_SCOPE or value.get('reference_labels_loaded') is not False
        or value.get('gpu_used') is not False):raise ValueError('Unsupported trained-water schema or claim scope')
    bbox=value['requested_bbox'];dates=value['source_dates'];counts=value['counts'];model=value['model'];grid=value['grid']
    if (not isinstance(bbox,list) or len(bbox)!=4 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in bbox)
        or not -180<=bbox[0]<bbox[2]<=180 or not -90<=bbox[1]<bbox[3]<=90):raise ValueError('Invalid trained-water footprint')
    times=[datetime.fromisoformat(dates[key].replace('Z','+00:00')) for key in ['before','after']]
    if any(date.tzinfo is None for date in times) or times[0]>=times[1]:raise ValueError('Invalid trained-water acquisition dates')
    fields=['aoi_pixels','assessed_pixels','unknown_pixels','before_water_pixels','after_water_pixels',
            'new_water_candidate_pixels','lost_water_candidate_pixels','unchanged_water_pixels']
    if (any(not _trained_integer(counts[key]) for key in fields) or counts['aoi_pixels']==0
        or counts['assessed_pixels']+counts['unknown_pixels']!=counts['aoi_pixels']
        or counts['before_water_pixels']!=counts['lost_water_candidate_pixels']+counts['unchanged_water_pixels']
        or counts['after_water_pixels']!=counts['new_water_candidate_pixels']+counts['unchanged_water_pixels']
        or counts['new_water_candidate_pixels']+counts['lost_water_candidate_pixels']+counts['unchanged_water_pixels']>counts['assessed_pixels']):
        raise ValueError('Inconsistent trained-water accounting')
    if (any(not _trained_digest(model[key]) for key in ['sha256','freeze_sha256','protocol_sha256'])
        or not _trained_integer(model['bytes']) or model['bytes']==0 or model.get('threshold')!=0.5
        or model.get('scikit_learn_version')!='1.7.2'):raise ValueError('Unsupported trained-water model binding')
    transform=grid['transform'];shape=grid['shape']
    if (grid.get('crs')!='EPSG:32645' or grid.get('resolution_m')!=10 or not isinstance(shape,list) or len(shape)!=2
        or any(not _trained_integer(x) or x==0 for x in shape) or not isinstance(transform,list) or len(transform)!=9
        or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in transform)
        or transform[0]!=10 or transform[1]!=0 or transform[3]!=0 or transform[4]!=-10 or transform[6:]!=[0,0,1]):
        raise ValueError('Unsupported trained-water grid')
    outputs=value['outputs'];required=['before-water.tif','after-water.tif','new-water-candidates.tif',
                                      'lost-water-candidates.tif','valid-mask.tif','aoi-mask.tif','overview.png']
    if not isinstance(outputs,dict) or any(name not in outputs for name in required):raise ValueError('Missing trained-water outputs')
    files=[];folder=TRAINED_WATER.parent.resolve()
    for name,record in outputs.items():
        if (not isinstance(name,str) or Path(name).name!=name or not isinstance(record,dict)
            or not _trained_digest(record.get('sha256')) or not _trained_integer(record.get('bytes'))):raise ValueError('Invalid trained-water output record')
        path=folder/name
        if not path.is_file() or path.resolve().parent!=folder or path.stat().st_size!=record['bytes']:raise ValueError('Trained-water output missing or changed')
        stat=path.stat();files.append((name,path,stat.st_size,stat.st_mtime_ns,stat.st_ino))
    key=(str(TRAINED_WATER),digest,tuple((name,size,modified,inode) for name,_,size,modified,inode in files))
    with trained_water_lock:
        if trained_water_cache is not None and trained_water_cache[0]==key:return trained_water_cache[1]
        for name,path,_,_,_ in files:
            with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=outputs[name]['sha256']:raise ValueError('Trained-water output SHA256 mismatch')
        arrays={}
        for name in required[:-1]:
            with rasterio.open(folder/name) as dataset:
                if (dataset.count!=1 or dataset.dtypes[0]!='uint8' or list(dataset.shape)!=shape
                    or dataset.crs!=rasterio.crs.CRS.from_epsg(32645) or list(dataset.transform)!=transform):
                    raise ValueError('Trained-water raster grid differs from its record')
                arrays[name]=dataset.read(1)
        aoi=arrays['aoi-mask.tif'];valid=arrays['valid-mask.tif']
        if not np.isin(aoi,[0,1]).all() or not np.isin(valid,[0,1]).all() or np.any((valid==1)&(aoi==0)):
            raise ValueError('Invalid trained-water coverage masks')
        assessed=valid==1
        for name in required[:4]:
            labels=arrays[name]
            if not np.isin(labels,[0,1,255]).all() or np.any(labels[~assessed]!=255) or np.any(labels[assessed]==255):
                raise ValueError('Trained-water labels must preserve the joint valid domain')
        before=arrays['before-water.tif']==1;after=arrays['after-water.tif']==1
        new=assessed&after&~before;lost=assessed&before&~after;unchanged=assessed&before&after
        if not np.array_equal(arrays['new-water-candidates.tif'][assessed],new[assessed].astype('uint8')) or not np.array_equal(arrays['lost-water-candidates.tif'][assessed],lost[assessed].astype('uint8')):
            raise ValueError('Trained-water temporal labels disagree with before/after states')
        actual_counts={'aoi_pixels':int((aoi==1).sum()),'assessed_pixels':int(assessed.sum()),
                       'unknown_pixels':int(((aoi==1)&~assessed).sum()),'before_water_pixels':int((before&assessed).sum()),
                       'after_water_pixels':int((after&assessed).sum()),'new_water_candidate_pixels':int(new.sum()),
                       'lost_water_candidate_pixels':int(lost.sum()),'unchanged_water_pixels':int(unchanged.sum())}
        if any(actual_counts[name]!=counts[name] for name in fields):raise ValueError('Trained-water counts disagree with raster labels')
        result={key:value[key] for key in ['schema','available','scope','source_dates','requested_bbox','model','grid','preprocessing','limitations','reference_labels_loaded','gpu_used']}
        result.update(counts=actual_counts,source_sha256=digest,output_sha256={name:record['sha256'] for name,record in outputs.items()},
                      figure_url=f'/trained-water/{digest}/overview.png',
                      verification='Recorded output SHAs, shared grid, joint masks, label domain, temporal states and pixel counts checked locally')
        trained_water_cache=(key,result)
        return result


def trained_water_evidence():
    try:
        result=trained_water_record()
        return result if result is not None else {'available':False,'status':'Trained water outputs are not available yet.'}
    except (OSError,ValueError,KeyError,TypeError,AttributeError,rasterio.errors.RasterioError):
        return {'available':False,'status':'Trained water artifacts failed source checks; no unverified model counts are displayed.'}


def model_request(body,path):
    """Client fields are separate from trusted source facts and graph states."""
    fields={'language'} if path=='/api/radar-report' else {'question','language'}
    if not isinstance(body,dict) or set(body)!=fields:
        raise ValueError('Unexpected request fields')
    if body['language'] not in ['en','ne']:
        raise ValueError('Use en/ne language')
    if path!='/api/radar-report' and (not isinstance(body['question'],str) or not body['question'].strip() or len(body['question'])>1200):
        raise ValueError('Use a question of at most 1200 characters')
    return body


def radar_exposure():
    """Display only exposure metadata bound to the active radar output hashes."""
    try:
        store=radar_store()
        if store.exposure_binding is None:
            return {'available':False,'status':'Mapped-feature intersections for the active radar source are not available yet.'}
        binding=store.exposure_binding
        value=store.exposure_summary
        fields=['mapped_feature_count','features_with_assessed_radar','features_with_change_candidate_overlap','features_with_water_candidate_overlap']
        fact=store.facts['radar_exposure']
        return {'available':True,'rows':[{'kind':kind,**{key:value[kind][key] for key in fields}} for kind in ['roads','bridges','buildings']],
                'road_buffer':{key:value['road_buffer'][key] for key in ['half_width_m','total_width_m']},
                'source_sha256':binding['summary_sha256'],'radar_manifest_sha256':store.source_sha256,
                'binding_sha256':binding['source_sha256'],'claim':{'en':fact.en,'ne':fact.ne},
                'source_pointer':fact.source_pointer,'claim_scope':'real_radar_candidate_screen_only'}
    except (RadarUnavailable,OSError,ValueError,KeyError,TypeError):
        return {'available':False,'status':'Mapped-feature facts failed source bindings; no intersection counts are displayed.'}


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*_):pass
    def reply(self,value,status=200):
        raw=json.dumps(value,allow_nan=False,ensure_ascii=False).encode();self.send_response(status)
        self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def do_GET(self):
        path=urlparse(self.path).path
        if path=='/api/geometry':self.reply(geometry());return
        if path=='/api/sources':self.reply(source_status());return
        if path=='/api/radar-evidence':self.reply(radar_evidence());return
        if path=='/api/radar-exposure':self.reply(radar_exposure());return
        if path=='/api/trained-water':self.reply(trained_water_evidence());return
        if path=='/api/access':self.reply(summary(access({})[0]));return
        if path.startswith('/trained-water/'):
            components=path.strip('/').split('/');record=trained_water_evidence()
            if len(components)!=3 or components[2]!='overview.png' or record.get('available') is not True or components[1]!=record['source_sha256']:
                self.reply({'error':'Trained water figure unavailable or its source changed'},404);return
            raw=TRAINED_WATER.with_name('overview.png').read_bytes()
            if hashlib.sha256(raw).hexdigest()!=record['output_sha256']['overview.png']:
                self.reply({'error':'Trained water figure changed; refresh the source record'},404);return
            self.send_response(200);self.send_header('Content-Type','image/png');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);return
        if path.startswith('/radar-report/'):
            name=path.removeprefix('/radar-report/')
            artifact=radar_report_artifact(name[:-5]) if name.endswith('.json') else None
            if artifact is None:self.reply({'error':'Report not found or its source changed. Generate a new report.'},404);return
            raw=artifact['bytes'];self.send_response(200)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Disposition',f'attachment; filename="{artifact["filename"]}"')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(raw)))
            self.end_headers();self.wfile.write(raw);return
        if path.startswith('/radar-evidence/'):
            components=path.strip('/').split('/')
            if len(components)!=3 or components[2]!='overview.png':self.reply({'error':'Not found'},404);return
            try:store=radar_store()
            except RadarUnavailable:self.reply({'error':'The radar source is unavailable'},503);return
            if components[1]!=store.source_sha256:self.reply({'error':'The radar source changed; refresh the evidence record'},404);return
            target=Path(store.source_path).with_name('overview.png')
        else:
            target=(WEB/(path.lstrip('/') or 'index.html')).resolve()
            if not target.is_relative_to(WEB.resolve()):self.reply({'error':'Not found'},404);return
        if not target.is_file():self.reply({'error':'Not found'},404);return
        raw=target.read_bytes();self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(target)[0] or 'application/octet-stream')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def do_POST(self):
        if self.path not in ['/api/access','/api/ask','/api/proof','/api/radar-ask','/api/radar-report']:self.reply({'error':'Not found'},404);return
        if self.headers.get('Origin') and self.headers['Origin']!='http://'+self.headers.get('Host',''):
            self.reply({'error':'Cross-origin requests are disabled'},403);return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=32768:raise ValueError('Invalid request size')
            body=json.loads(self.rfile.read(length))
            if self.path in ['/api/radar-ask','/api/radar-report']:
                model_request(body,self.path)
                store=radar_store()
                if not model_lock.acquire(blocking=False):self.reply({'error':'The CPU assistant is answering another question. Try again shortly.'},429);return
                try:
                    if self.path=='/api/radar-report':
                        answer=generate_situation_report(store,language=body['language'],endpoint='http://127.0.0.1:8782')
                        answer=cache_radar_report(answer,store)
                    else:answer=ask_local_model(store,body['question'],language=body['language'],endpoint='http://127.0.0.1:8782')
                except (ValueError,KeyError,TypeError):
                    self.reply({'error':'The local model did not return a valid source selection. No report was generated. Try again.'},503);return
                finally:model_lock.release()
                self.reply(answer);return
            expected={'observations','question','language'} if self.path=='/api/ask' else {'observations'}
            if not isinstance(body,dict) or set(body)!=expected:raise ValueError('Unexpected request fields')
            proof,store=access(body['observations'])
            if self.path=='/api/proof':self.reply(proof);return
            if self.path=='/api/access':self.reply(summary(proof));return
            if not isinstance(body['question'],str) or not 0<len(body['question'])<=1200 or body['language'] not in ['en','ne']:
                raise ValueError('Use a question of at most 1200 characters and en/ne language')
            if not model_lock.acquire(blocking=False):self.reply({'error':'The CPU assistant is answering another question. Try again shortly.'},429);return
            try:answer=ask_local_model(store,body['question'],language=body['language'],endpoint='http://127.0.0.1:8782')
            finally:model_lock.release()
            self.reply(answer)
        except RadarUnavailable as error:self.reply({'error':str(error)},503)
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as error:self.reply({'error':str(error)},400)
        except Exception:self.reply({'error':'The local assistant is unavailable. Start the pinned CPU model server and retry.'},503)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8783)
    parser.add_argument('--radar-source',choices=['auto','corrected','provisional','engineering'],default='auto',
                        help='Select a recorded radar run; auto prefers the corrected Copernicus DEM manifest.')
    args=parser.parse_args()
    RADAR_OVERRIDE={'auto':None,'corrected':RADAR_PRODUCTION,'provisional':RADAR_PROVISIONAL,'engineering':RADAR_ENGINEERING}[args.radar_source]
    print(f'Access proof explorer: http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
