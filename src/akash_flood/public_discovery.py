"""Bounded public Earth Search discovery, preserving complete response bytes."""
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import requests

from .catalog import validate_bbox

BASE = 'https://earth-search.aws.element84.com/v1'
COLLECTIONS = ('sentinel-1-grd', 'sentinel-2-c1-l2a')


def trusted_endpoint(url):
    p = urlsplit(url)
    if p.scheme != 'https' or p.hostname != 'earth-search.aws.element84.com' or p.port not in (None,443) or p.username or p.password or not p.path.startswith('/v1/'):
        raise ValueError('Pagination leaves the official public Earth Search endpoint')


def public_search(collection, bbox, event_date, out, *, max_days=24, session=None, max_pages=20):
    if collection not in COLLECTIONS or not isinstance(max_days,int) or isinstance(max_days,bool) or not 1 <= max_days <= 90:
        raise ValueError('Use the declared Sentinel collection and bounded date window')
    validate_bbox(bbox)
    event = date.fromisoformat(event_date)
    start = datetime.combine(event - timedelta(days=max_days),time(),tzinfo=timezone.utc)
    end = datetime.combine(event + timedelta(days=max_days+1),time(),tzinfo=timezone.utc)
    query = {'collections':[collection],'bbox':list(bbox),'datetime':f'{start.isoformat()}/{end.isoformat()}','limit':100}
    path=Path(out);path.mkdir(parents=True,exist_ok=False)
    client=session or requests.Session()
    url=BASE+'/search';method='POST';body=query
    items=[];seen=set();pages=[];network=0;pagination=set()
    for number in range(max_pages):
        trusted_endpoint(url)
        signature=json.dumps([url,method,body],sort_keys=True)
        if signature in pagination:
            raise ValueError('Catalogue pagination repeats; query incomplete')
        pagination.add(signature)
        response=client.request(method,url,json=body if method=='POST' else None,timeout=(15,45),allow_redirects=False)
        response.raise_for_status()
        if response.status_code!=200:
            raise ValueError('Unexpected catalogue response or redirect')
        raw=response.content;network+=len(raw)
        if len(raw)>10_000_000 or network>40_000_000:
            raise ValueError('Catalogue response byte budget exceeded')
        page=path/f'page-{number:02}.json';page.write_bytes(raw)
        value=json.loads(raw)
        if value.get('type')!='FeatureCollection' or not isinstance(value.get('features'),list):
            raise ValueError('Expected public STAC FeatureCollection')
        for item in value['features']:
            if item.get('collection')!=collection or item.get('type')!='Feature' or not isinstance(item.get('id'),str):
                raise ValueError('Catalogue returned another source family or malformed item')
            if item['id'] in seen:
                raise ValueError('Repeated item across pages; completeness ambiguous')
            seen.add(item['id']);items.append(item)
        pages.append({'path':str(page.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                      'url':url,'method':method,'body':body,'item_count':len(value['features'])})
        nextlinks=[link for link in value.get('links',[]) if link.get('rel')=='next']
        if not nextlinks:
            return items,{'collection':collection,'query':query,'pages':pages,'network_bytes':network,
                          'search_complete':True,'completeness_scope':'No next link in final provider response within bounded query; provider availability and metadata accuracy are separate'}
        if len(nextlinks)!=1:
            raise ValueError('Ambiguous next-page links')
        link=nextlinks[0];url=link['href'];method=link.get('method','GET').upper()
        if method not in ('GET','POST') or link.get('headers'):
            raise ValueError('Unsupported pagination method or requested headers')
        if method=='POST':
            body=link.get('body')
            if not isinstance(body,dict):
                raise ValueError('POST pagination lacks explicit body')
            if link.get('merge'):
                body={**query,**body}
        else:
            body=None
    raise ValueError('Page budget exceeded; no complete discovery claim')
