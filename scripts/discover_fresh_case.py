"""Discover public native SAR and optical metadata for a new area/date."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path

from akash_flood.catalog import validate_bbox
from akash_flood.public_discovery import BASE,COLLECTIONS,public_search


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bbox',nargs=4,type=float,required=True)
    p.add_argument('--event-date',required=True)
    p.add_argument('--context-buffer-deg',type=float,default=.03)
    p.add_argument('--max-days',type=int,default=24)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bbox=validate_bbox(a.bbox)
    if not 0<=a.context_buffer_deg<=.25 or a.out.exists():
        raise ValueError('Use bounded context buffer and fresh output')
    context=validate_bbox([bbox[0]-a.context_buffer_deg,bbox[1]-a.context_buffer_deg,
                           bbox[2]+a.context_buffer_deg,bbox[3]+a.context_buffer_deg])
    a.out.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(public_search,c,context,a.event_date,a.out/c,max_days=a.max_days) for c in COLLECTIONS]
        results=[job.result() for job in jobs]
    output={'source':BASE,'retrieved_at':datetime.now(timezone.utc).isoformat(),
            'event_date':a.event_date,'bbox':bbox,'queried_context_bbox':context,'max_days':a.max_days,
            'sentinel1':results[0][0],'sentinel2':results[1][0],
            'search_complete':True,'search_records':[r[1] for r in results],
            'analysis_status':'not_run','reference_labels_loaded':False,'account_used':False,'gpu_used':False}
    (a.out/'catalog.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'sentinel1_items':len(output['sentinel1']),'sentinel2_items':len(output['sentinel2']),
                      'network_bytes':sum(r[1]['network_bytes'] for r in results),'search_complete':True,
                      'catalog':str((a.out/'catalog.json').resolve()),'pixels_downloaded':False},indent=2))


if __name__=='__main__':main()
