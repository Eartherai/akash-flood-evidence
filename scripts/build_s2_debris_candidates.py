"""Read a bounded public optical pair and screen provisional surface changes.

Despite the filename, this does not establish debris presence. Output is an
explicitly partial exposed-surface/vegetation-loss candidate layer. The protocol
is frozen before acquisition and remains distinct from the original70%gate.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

from affine import Affine
import numpy as np
from pyproj import CRS,Transformer
import rasterio
from rasterio.features import geometry_mask,shapes
from rasterio.windows import Window
from shapely.geometry import Polygon,mapping,shape
from shapely.ops import transform

from akash_flood.catalog import validate_bbox
from akash_flood.fresh_case import reject_source_path
from akash_flood.optical_change import POLICY,StrictRangeHTTP,select_optical_pair,reflectance,average_2x2,screen
from akash_flood.s1_crop import SingleRasterOpener
from build_fresh_report import record,write_png

ROOT=Path(__file__).resolve().parents[1]
SOURCES=[
    {'url':'https://sentiwiki.copernicus.eu/web/s2-products','supports':'SR = DN*scale+offset; PB04andlater BOA_ADD_OFFSET; DN0nodata; band resolutions'},
    {'url':'https://sentiwiki.copernicus.eu/web/s2-processing','supports':'SCL cloud/shadow/snow/land classes; imperfect classifier, not ground truth'},
    {'url':'https://nhess.copernicus.org/articles/23/2625/2023/','supports':'Pre/post NDVI loss is a potential shallow-landslide indicator; false positives and local empirical threshold limitations'},
    {'url':'https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/barren_soil/','supports':'B11/B04/B08/B02 BSI formula; bare ground can reflect agriculture, buildings, deforestation and season changes'},
]


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare_reuse(folder,pair):
    """Freeze completed local crops from a preserved failed acquisition.

    The legacy failed run did not capture per-crop hashes at write time. Hashes
    are explicitly recorded now, never represented as contemporaneous checks.
    Source linkage relies on the original crop writer and saved range ledger;
    it does not independently verify a provider whole-object checksum.
    """
    if folder is None:return None
    folder=folder.resolve();reject_source_path(folder)
    protocol_path=folder/'protocol.json';failure_path=folder/'failed.json'
    protocol=json.loads(protocol_path.read_text());failure=json.loads(failure_path.read_text())
    if (protocol.get('schema')!='akash-s2-surface-change-protocol-v1'
            or protocol.get('execute_requested') is not True or failure.get('status')!='failed'):
        raise ValueError('Reuse requires the preserved failed optical acquisition and protocol')
    audit={'recorded_at':datetime.now(timezone.utc).isoformat(),'source_protocol':record(protocol_path),
           'source_failure':record(failure_path),'prior_received_body_bytes':failure['received_body_bytes'],
           'hash_capture_scope':'Legacy crop hashes captured after the failed run, before this replay; original write-time crop hashes unavailable',
           'source_linkage_scope':'Original crop-writer provenance, saved STAC, TIFF geometry and successful source range identities; not whole-object checksum verification',
           'crops':{}}
    for name,item in pair.items():
        source=folder/(name+'.stac.json');source_record=record(source)
        if source_record!=protocol['source_items'][name] or json.loads(source.read_text())!=item:
            raise ValueError('Reuse source item differs from selected optical scene')
        audit['crops'][name]={}
        for key in POLICY['bands']:
            path=folder/name/(key+'-dn.tif')
            if not path.exists():continue
            asset=item['assets'][key]
            ranges=[r for r in failure['requests'] if r['url']==asset['href']]
            identities={(r['file_bytes'],r['etag']) for r in ranges}
            if (not ranges or len(identities)!=1 or next(iter(identities))[0]!=asset['file:size']
                    or any(r.get('status')!=206 or len(r.get('sha256',''))!=64 for r in ranges)):
                raise ValueError('Reuse crop lacks consistent saved source range identities')
            audit['crops'][name][key]={'crop':record(path),'source_item':source_record,
                 'remote_identity':{'bytes':ranges[0]['file_bytes'],'etag':ranges[0]['etag']},
                 'source_object_full_checksum_verified':False,'successful_range_count':len(ranges)}
    if not any(audit['crops'].values()):raise ValueError('No completed crops to reuse')
    return audit


def import_crop(item,key,window,base_grid,entry,out):
    source=Path(entry['crop']['path'])
    if record(source)!=entry['crop']:raise ValueError('Reused crop checksum changed after freeze')
    asset=item['assets'][key];crs,affine,_,_=asset_grid(item,key)
    factor=1 if key in ('scl','swir16') else 2
    native=Window(window.col_off*factor,window.row_off*factor,window.width*factor,window.height*factor)
    expected=affine*Affine.translation(native.col_off,native.row_off)
    band=asset['raster:bands'][0];dtype='uint8' if key=='scl' else 'uint16'
    if crs!=base_grid[0] or asset_grid(item,'scl')[1].c!=affine.c or asset_grid(item,'scl')[1].f!=affine.f:
        raise ValueError('Reuse grid origins/projections differ')
    with rasterio.open(source) as ds:
        if (ds.count!=1 or ds.crs!=crs or ds.transform!=expected or ds.dtypes[0]!=dtype
                or ds.width!=native.width or ds.height!=native.height or ds.nodata!=band['nodata']):
            raise ValueError('Reused crop grid, type or nodata differs from exact native source window')
        dn=ds.read(1)
    destination=out/(key+'-dn.tif');destination.write_bytes(source.read_bytes())
    receipt={'crop':record(destination),'asset':asset,'imported_from':entry,
         'source_object_full_checksum_verified':False,'remote_identity':entry['remote_identity'],
         'native_window':list(native.flatten()),'horizontal_aggregation':'none' if factor==1 else 'exact2x2allvalid10m-to20maverage'}
    if key=='scl':return dn,receipt
    values,valid=reflectance(dn,band['scale'],band['offset'],band['nodata'])
    if factor==2:values,_=average_2x2(values,valid)
    return values,receipt


def perimeter(bbox,crs):
    west,south,east,north=bbox;tx=Transformer.from_crs(4326,crs,always_xy=True)
    points=[]
    for p,q in [((west,south),(east,south)),((east,south),(east,north)),((east,north),(west,north)),((west,north),(west,south))]:
        for f in np.linspace(0,1,41,endpoint=False):points.append(tx.transform(p[0]+f*(q[0]-p[0]),p[1]+f*(q[1]-p[1])))
    return Polygon(points)


def asset_grid(item,key):
    asset=item['assets'][key];epsg=item['properties']['proj:epsg']
    crs=CRS.from_epsg(epsg);grid=Affine(*asset['proj:transform'][:6]);height,width=asset['proj:shape']
    size=20 if key in ('scl','swir16') else 10
    if (not crs.is_projected or any(a.unit_name!='metre' for a in crs.axis_info)
            or grid.a!=size or grid.e!=-size or grid.b!=0 or grid.d!=0
            or asset.get('gsd')!=size or len(asset.get('raster:bands',[]))!=1):
        raise ValueError('Require native10m/20m single-band metric Collection-1 assets')
    return crs,grid,width,height


def window_for(item,bbox):
    crs,affine,width,height=asset_grid(item,'scl');geometry=perimeter(bbox,crs);inv=~affine
    xy=np.array([inv*xy for xy in geometry.exterior.coords])
    left,top=np.floor(xy.min(axis=0)).astype(int)-4;right,bottom=np.ceil(xy.max(axis=0)).astype(int)+4
    if left<0 or top<0 or right>width or bottom>height or (right-left)*(bottom-top)>4_000_000:
        raise ValueError('Optical AOI/60m halo unsupported or exceeds4million20m cells')
    window=Window(int(left),int(top),int(right-left),int(bottom-top))
    return window,(crs,affine*Affine.translation(left,top),window.width,window.height),geometry


def read_band(item,key,window,base_grid,client,out):
    asset=item['assets'][key];crs,affine,width,height=asset_grid(item,key);url=asset['href']
    scl_affine=asset_grid(item,'scl')[1]
    factor=1 if key in ('scl','swir16') else 2
    if crs!=base_grid[0] or affine.c!=scl_affine.c or affine.f!=scl_affine.f:
        raise ValueError('Band origins/projections differ; no fitted registration or interpolation')
    native=Window(window.col_off*factor,window.row_off*factor,window.width*factor,window.height*factor)
    band=asset['raster:bands'][0];expected_type='uint8' if key=='scl' else 'uint16'
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_PAM_ENABLED='NO'):
        with rasterio.open(url,opener=SingleRasterOpener(url,client)) as ds:
            if (ds.crs!=crs or ds.transform!=affine or ds.width!=width or ds.height!=height
                    or ds.count!=1 or ds.dtypes[0]!=expected_type or ds.nodata!=band['nodata']):
                raise ValueError('Saved TIFF metadata differs from selected STAC asset')
            dn=ds.read(1,window=native)
    if client.identities[url][0]!=asset['file:size']:
        raise ValueError('Remote asset length differs from exact saved catalogue source')
    if dn.shape!=(int(native.height),int(native.width)):
        raise ValueError('Incomplete native optical crop')
    path=out/(key+'-dn.tif')
    with rasterio.open(path,'w',driver='GTiff',count=1,width=dn.shape[1],height=dn.shape[0],dtype=dn.dtype,
                       crs=crs,transform=affine*Affine.translation(native.col_off,native.row_off),nodata=band['nodata'],compress='deflate') as dst:
        dst.write(dn,1)
    receipt={'crop':record(path),'asset':asset,'source_object_full_checksum_verified':False,
             'remote_identity':{'bytes':client.identities[url][0],'etag':client.identities[url][1]},
             'range_checksum_scope':'Each received range hashed in global transport ledger; advertised whole-object hash retained but not asserted verified',
             'native_window':list(native.flatten()),'horizontal_aggregation':'none' if factor==1 else 'exact2x2allvalid10m-to20maverage'}
    if key=='scl':return dn,receipt
    if 'scale' not in band or 'offset' not in band:
        raise ValueError('Declared reflectance scale/offset required; no assumed radiometry')
    values,valid=reflectance(dn,band['scale'],band['offset'],band['nodata'])
    if factor==2:values,_=average_2x2(values,valid)
    return values,receipt


def write_layer(path,array,grid,nodata):
    with rasterio.open(path,'w',driver='GTiff',count=1,width=int(grid[2]),height=int(grid[3]),dtype=array.dtype,
                       crs=grid[0],transform=grid[1],nodata=nodata,compress='deflate') as dst:dst.write(array,1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True);p.add_argument('--bbox',type=float,nargs=4,required=True)
    p.add_argument('--event-date',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reuse-crops',type=Path,help='Explicitly import completed crops from a preserved failed run')
    p.add_argument('--execute',action='store_true');a=p.parse_args();bbox=validate_bbox(a.bbox)
    if a.out.exists():raise FileExistsError('Preserve earlier optical runs; require fresh output')
    reject_source_path(a.catalog);catalog=json.loads(a.catalog.read_text());pair=select_optical_pair(catalog,bbox,a.event_date)
    window,grid,geometry=window_for(pair['before'],bbox)
    after_window,after_grid,_=window_for(pair['after'],bbox)
    if grid!=after_grid or window!=after_window:raise ValueError('Optical date grids differ; no date interpolation allowed')
    reuse=prepare_reuse(a.reuse_crops,pair)
    a.out.mkdir(parents=True);items={}
    for name,item in pair.items():
        path=a.out/(name+'.stac.json');dump(path,item);items[name]=record(path)
    policy={'schema':'akash-s2-surface-change-protocol-v1','recorded_before_pixel_acquisition_at':datetime.now(timezone.utc).isoformat(),
        'catalog':record(a.catalog),'bbox':bbox,'event_date':a.event_date,'policy':POLICY,'source_items':items,
        'code':[record(Path(__file__)),record(ROOT/'src/akash_flood/optical_change.py'),record(ROOT/'src/akash_flood/s1_crop.py')],
        'sources':SOURCES,'known_context':'Prior radar reference overlap and previous SCL coverage outcomes already known; new experimental surface-change rule, not a blind case-study validation',
        'crop_reuse':reuse,
        'reference_labels_loaded':False,'original_gate_unchanged':True,'maximum_new_http_body_bytes':60_000_000,
        'grid':{'crs':str(grid[0]),'transform':list(grid[1]),'shape':[int(grid[3]),int(grid[2])]},'execute_requested':a.execute}
    dump(a.out/'protocol.json',policy)
    if not a.execute:
        print(json.dumps({'status':'protocol_only','network_requests':0,'pixels_read':False,'selected':{k:x['id'] for k,x in pair.items()}},indent=2));return
    start=time.perf_counter();client=StrictRangeHTTP();receipts={};scenes={};scls={}
    try:
        for name,item in pair.items():
            folder=a.out/name;folder.mkdir();scenes[name]={};receipts[name]={}
            for key in POLICY['bands']:
                entry=reuse['crops'][name].get(key) if reuse else None
                values,receipt=import_crop(item,key,window,grid,entry,folder) if entry else read_band(item,key,window,grid,client,folder)
                receipts[name][key]=receipt
                dump(folder/(key+'-crop-receipt.json'),receipt)
                if key=='scl':scls[name]=values
                else:scenes[name][key]=values
        aoi=geometry_mask([mapping(geometry)],out_shape=(int(grid[3]),int(grid[2])),transform=grid[1],invert=True,all_touched=False)
        labels,assessed,indices,counts,raw_clear=screen(scenes['before'],scenes['after'],scls['before'],scls['after'],aoi)
        write_layer(a.out/'surface-change-candidates.tif',labels,grid,255)
        write_layer(a.out/'valid-mask.tif',assessed.astype('uint8'),grid,0);write_layer(a.out/'aoi-mask.tif',aoi.astype('uint8'),grid,0)
        for name in ('before','after'):
            for key,x in indices[name].items():write_layer(a.out/(name+'-'+key+'.tif'),np.where(assessed,x,-9999).astype('float32'),grid,-9999)
        unproject=Transformer.from_crs(grid[0],4326,always_xy=True).transform
        features=[]
        for g,value in shapes(labels,mask=labels==1,transform=grid[1],connectivity=4):
            poly=shape(g)
            if value!=1 or not poly.is_valid:raise ValueError('Invalid native optical candidate polygon')
            features.append({'type':'Feature','geometry':mapping(transform(unproject,poly)),'properties':{
                'native_grid_area_m2':poly.area,'candidate_basis':POLICY['scope'],'confirmed_debris':False,'confirmed_landslide':False,'road_closure_assigned':False}})
        dump(a.out/'surface-change-candidates.geojson',{'type':'FeatureCollection','features':features})
        rgb=np.full((*aoi.shape,3),255,dtype='uint8');rgb[aoi]=[193,200,207];rgb[assessed]=[232,240,230];rgb[labels==1]=[147,78,35]
        write_png(a.out/'overview.png',rgb)
        transport={'received_body_bytes':client.bytes_read,'requests':client.requests,'tls_verified':True,'redirects_followed':False,
                   'maximum_body_bytes':client.max_bytes,'account_used':False,'source_full_objects_downloaded':False}
        dump(a.out/'transport.json',transport)
        clear={name:{'raw_scl_clear_pixels':n,'fraction':n/counts['aoi_pixels'],'original70percentgate_passes':n/counts['aoi_pixels']>=.7} for name,n in raw_clear.items()}
        manifest={'schema':'akash-s2-surface-change-v1','status':'completed','completed_at':datetime.now(timezone.utc).isoformat(),
            'protocol':record(a.out/'protocol.json'),'bbox':bbox,'event_date':a.event_date,
            'source_dates':{name:item['properties']['datetime'] for name,item in pair.items()},'source_crops':receipts,
            'counts':counts,'raw_scl_single_scene_clear':clear,'joint_assessed_fraction':counts['joint_assessed_pixels']/counts['aoi_pixels'],
            'candidate_area_m2':counts['candidate_pixels']*400,'grid':policy['grid'],'scene_date_alignment_resampling_used':False,
            '10m_reflectance_aggregated_to20m':True,'partial_exploration_is_original_gate_pass':False,
            'transport':record(a.out/'transport.json'),'network_received_body_bytes':client.bytes_read,'network_requests':len(client.requests),
            'prior_failed_run_received_body_bytes':reuse['prior_received_body_bytes'] if reuse else 0,
            'recorded_campaign_received_body_bytes':client.bytes_read+(reuse['prior_received_body_bytes'] if reuse else 0),
            'imported_crop_count':sum('imported_from' in r for group in receipts.values() for r in group.values()),
            'outputs':{path.name:record(path) for path in a.out.iterdir() if path.suffix in ('.tif','.geojson','.png')},
            'scope':POLICY['scope'],'local_debris_accuracy_established':False,'reference_labels_loaded':False,'gpu_used':False,
            'wall_seconds':time.perf_counter()-start,'limitations':[
                'Spectral changes can be agriculture, deforestation, riverbank erosion, moisture, season change, shadow or atmospheric/classification errors.',
                'Only previously vegetated jointly supported surfaces are screened; bare-surface debris and small slides may be missed.',
                'Cloud/shadow/snow/water/invalid neighborhoods remain unknown; no generated imagery or temporal union is used.',
                'Numeric thresholds are untuned engineering hypotheses; absolute registration and local debris accuracy unvalidated.',
                'Partial exploration does not overturn the preserved70percent single-scene optical gate or constitute a complete debris map.',
                'Range checksums bind received bytes and ETag/length consistency only; full-object provider digests have not been verified.']}
        dump(a.out/'manifest.json',manifest)
        print(json.dumps({'status':'completed','counts':counts,'clear':clear,'network_body_bytes':client.bytes_read,
                          'network_requests':len(client.requests),'local_debris_accuracy_established':False,'manifest':record(a.out/'manifest.json')},indent=2))
    except Exception as error:
        dump(a.out/'failed.json',{'status':'failed','error_type':type(error).__name__,'message':str(error),
             'received_body_bytes':client.bytes_read,'requests':client.requests,'no_automatic_retry':True})
        raise


if __name__=='__main__':main()
