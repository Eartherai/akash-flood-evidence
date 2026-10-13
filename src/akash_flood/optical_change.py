"""Exploratory exposed-surface/vegetation-loss screen, never confirmed debris.

The fixed thresholds are engineering hypotheses, without local label fitting.
All unsupported observations remain unknown. No cloud reconstruction is used.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import re
from urllib.parse import urlsplit

import numpy as np
import requests
from scipy.ndimage import binary_erosion
from shapely.geometry import box, shape

from .catalog import utc, validate_bbox
from .fresh_case import parse_discovery

POLICY = {
    'same_mgrs_tile': True, 'selector': 'Earliest full-footprint post-event scene; latest compatible pre-event scene; ID ties; whole event day excluded; no cloud/outcome ranking',
    'bands': ['blue','red','nir','swir16','scl'],
    'output_resolution_m': 20, 'reflectance_harmonization': 'Declared STAC DN*scale+offset; verify saved asset geometry/nodata; all four valid positive 10m samples required for each exact2x2 average',
    'land_scl_classes': [4,5], 'invalid_neighborhood_radius_m': 60,
    'pre_ndvi_min': .4, 'post_ndvi_max': .3, 'ndvi_loss_min': .2,
    'post_bsi_min_exclusive': 0, 'bsi_increase_min': .1,
    'reflectance_min_exclusive': 0, 'reflectance_max_inclusive': 1.5,
    'denominator_min_exclusive': 1e-6, 'original_single_scene_clear_gate_fraction': .7,
    'threshold_basis': 'Untuned engineering hypotheses for substantial vegetation loss plus exposed-surface spectrum; cited sources support mechanism/formula, not these numeric thresholds or local accuracy',
    'scope': 'Exposed-surface/vegetation-loss candidates requiring review, not confirmed debris, landslides or damage',
    'no_temporal_union': True, 'no_cloud_reconstruction': True,
}
PUBLIC_HOST='e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com'


def select_optical_pair(document,bbox,event_date,max_days=24):
    validate_bbox(bbox)
    if not 1 <= max_days <= 24:
        raise ValueError('Optical screen is bounded to at most24days')
    start=utc(event_date+'T00:00:00Z');end=start+timedelta(days=1)
    groups,_=parse_discovery(document)
    candidates={'before':[],'after':[]}
    for item in groups['sentinel2']:
        if item.get('collection')!='sentinel-2-c1-l2a' or not shape(item['geometry']).covers(box(*bbox)):
            continue
        when=utc(item['properties']['datetime'])
        branch='before' if start-timedelta(days=max_days)<=when<start else 'after' if end<=when<=end+timedelta(days=max_days) else None
        if branch:
            candidates[branch].append(item)
    tile=lambda i:tuple(i['properties'][k] for k in ('mgrs:utm_zone','mgrs:latitude_band','mgrs:grid_square'))
    options=[]
    for post in candidates['after']:
        pre=[x for x in candidates['before'] if tile(x)==tile(post) and x['properties']['proj:epsg']==post['properties']['proj:epsg']]
        if pre:
            pre=sorted(pre,key=lambda x:(-utc(x['properties']['datetime']).timestamp(),x['id']))[0]
            options.append((utc(post['properties']['datetime']),-utc(pre['properties']['datetime']).timestamp(),post['id'],pre['id'],pre,post))
    if not options:
        raise ValueError('No compatible full-AOI Collection-1 optical pair')
    selected=min(options,key=lambda row:row[:4])
    return {'before':selected[-2],'after':selected[-1]}


def reflectance(dn,scale,offset,nodata):
    if not np.isfinite([scale,offset]).all() or scale<=0 or nodata is None:
        raise ValueError('Explicit finite source scaling, offset and nodata required')
    valid=np.isfinite(dn)&(dn!=nodata)&(dn>0)&(dn<32767)
    values=dn.astype('float32')*scale+offset
    valid &= np.isfinite(values)&(values>0)&(values<=POLICY['reflectance_max_inclusive'])
    return np.where(valid,values,np.nan).astype('float32'),valid


def average_2x2(values,valid):
    if values.shape!=valid.shape or values.ndim!=2 or any(size%2 for size in values.shape):
        raise ValueError('Exact even-sized10m grid and matching validity required')
    h,w=values.shape
    blocks=values.reshape(h//2,2,w//2,2)
    supported=valid.reshape(h//2,2,w//2,2).all(axis=(1,3))
    result=blocks.mean(axis=(1,3))
    return np.where(supported,result,np.nan).astype('float32'),supported


def screen(before,after,scl_before,scl_after,aoi):
    if aoi.dtype!=bool or aoi.shape!=scl_before.shape or aoi.shape!=scl_after.shape:
        raise ValueError('Matching boolean AOI and native20m SCL required')
    support=np.isin(scl_before,POLICY['land_scl_classes'])&np.isin(scl_after,POLICY['land_scl_classes'])
    for scene in (before,after):
        if set(scene)!= {'blue','red','nir','swir16'} or any(x.shape!=aoi.shape for x in scene.values()):
            raise ValueError('Four exact same-grid reflectance bands required')
        for values in scene.values():support &= np.isfinite(values)&(values>0)&(values<=1.5)
    # Euclidean60m exclusion around any unsupported observation; border is also
    # unsupported. This includes cloud/shadow/snow/water/nonland and missing data.
    yy,xx=np.mgrid[-3:4,-3:4];structure=xx*xx+yy*yy<=9
    supported=binary_erosion(support,structure=structure,border_value=0)
    indices={}
    for name,scene in [('before',before),('after',after)]:
        denom=scene['nir']+scene['red'];bsi_denom=sum(scene.values())
        finite_den=np.isfinite(denom)&np.isfinite(bsi_denom)&(denom>1e-6)&(bsi_denom>1e-6)
        ndvi=np.full(aoi.shape,np.nan,dtype='float32');bsi=ndvi.copy()
        np.divide(scene['nir']-scene['red'],denom,out=ndvi,where=finite_den)
        np.divide(scene['swir16']+scene['red']-scene['nir']-scene['blue'],bsi_denom,out=bsi,where=finite_den)
        supported &= finite_den&np.isfinite(ndvi)&np.isfinite(bsi)
        indices[name]={'ndvi':ndvi,'bsi':bsi}
    assessed=aoi&supported
    p,q=indices['before'],indices['after']
    candidate=assessed&(p['ndvi']>=.4)&(q['ndvi']<=.3)&(p['ndvi']-q['ndvi']>=.2)&(q['bsi']>0)&(q['bsi']-p['bsi']>=.1)
    labels=np.where(assessed,candidate,255).astype('uint8')
    for scene in indices.values():
        for key in scene:scene[key][~assessed]=np.nan
    raw_clear={name:int((aoi&np.isin(scl,(4,5,6))).sum()) for name,scl in [('before',scl_before),('after',scl_after)]}
    counts={'aoi_pixels':int(aoi.sum()),'joint_assessed_pixels':int(assessed.sum()),
            'unknown_pixels':int((aoi&~assessed).sum()),'candidate_pixels':int(candidate.sum())}
    return labels,assessed,indices,counts,raw_clear


class StrictRangeHTTP:
    """Credential-free exact public ranges with aggregate body-byte budget."""
    def __init__(self,max_bytes=60_000_000,session=None):
        if not 1<=max_bytes<=60_000_000:raise ValueError('Maximum optical budget is60MB')
        self.max_bytes=max_bytes;self.bytes_read=0;self.requests=[];self.identities={}
        self.client=session or requests.Session();self.client.trust_env=False
        if getattr(self.client,'auth',None):raise ValueError('No credentials accepted')
    def range(self,url,start,stop):
        parsed=urlsplit(url)
        if parsed.scheme!='https' or parsed.netloc!=PUBLIC_HOST or parsed.query or parsed.fragment or not parsed.path.startswith('/sentinel-2-c1-l2a/') or not parsed.path.endswith('.tif'):
            raise ValueError('Require exact public Collection-1 TIFF URL')
        count=stop-start
        if start<0 or not 0<count<=4_000_000 or self.bytes_read+count>self.max_bytes or len(self.requests)>=256:
            raise ValueError('Optical range or aggregate resource limit exceeded')
        headers={'Range':f'bytes={start}-{stop-1}','Accept-Encoding':'identity','User-Agent':'AkashOpticalExploration/1.0'}
        if url in self.identities:headers['If-Match']=self.identities[url][1]
        response=self.client.get(url,headers=headers,stream=True,timeout=(15,30),verify=True,allow_redirects=False)
        try:
            content_range=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
            if response.status_code!=206 or not content_range:raise ValueError('Exact range response required; no full-scene fallback')
            first,last,total=map(int,content_range.groups());etag=response.headers.get('ETag')
            if first!=start or last+1!=min(stop,total) or not etag or response.headers.get('Content-Encoding','identity')!='identity':
                raise ValueError('Source range/ETag/encoding contract differs')
            if url in self.identities and self.identities[url]!=(total,etag):raise ValueError('Remote source identity changed')
            chunks=[];received=0
            for chunk in response.iter_content(65536):
                if chunk:
                    received+=len(chunk);self.bytes_read+=len(chunk)
                    if received>count or self.bytes_read>self.max_bytes:raise ValueError('Optical body-byte budget exceeded')
                    chunks.append(chunk)
            if received!=last-first+1:raise ValueError('Incomplete optical source range')
            payload=b''.join(chunks);self.identities[url]=(total,etag)
            self.requests.append({'url':url,'start':first,'stop':last+1,'file_bytes':total,'etag':etag,'received_body_bytes':received,'sha256':hashlib.sha256(payload).hexdigest(),'status':206})
            return payload,total
        finally:response.close()
