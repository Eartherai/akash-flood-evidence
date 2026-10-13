"""One explicitly selected historical provider; no automatic fallback or retry."""
import argparse
import json
from pathlib import Path
import run_fresh_case as base

ENDPOINTS={'fossgis':'https://overpass-api.de/api/interpreter',
           'private-coffee':'https://overpass.private.coffee/api/interpreter'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bbox',type=float,nargs=4,required=True)
    p.add_argument('--event-date',required=True)
    p.add_argument('--osm-snapshot',required=True)
    p.add_argument('--provider',choices=ENDPOINTS,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    original=base.OSM_ENDPOINT
    base.OSM_ENDPOINT=ENDPOINTS[a.provider]
    # A historical query declares up to90s of server work. The base helper's
    #20s first-byte timeout can end it before that declared work completes.
    # Keep its120s total wall/body bound, while permitting the initial response.
    import requests
    class HistoricalSession(requests.Session):
        def get(self,url,**kwargs):
            kwargs['timeout']=(15,95)
            return super().get(url,**kwargs)
    session=HistoricalSession()
    try:
        result=base.fetch_bounded_historic_osm(a.bbox,a.osm_snapshot,a.event_date,a.out,session=session)
        result['provider_choice']=a.provider
        result['provider_source']='https://wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances'
        result['automatic_fallback_or_retry']=False
        result['connect_read_timeout_seconds']=[15,95]
        base.write_json(a.out/'network-receipt.json',result,replace=True)
    except BaseException as error:
        a.out.mkdir(parents=True,exist_ok=True)
        base.write_json(a.out/'failure.json',{'status':'failed','endpoint':ENDPOINTS[a.provider],
            'provider_choice':a.provider,'error_type':type(error).__name__,'error':str(error),
            'automatic_retry':False,'current_data_fallback':False,'at_most_one_provider_request':True})
        raise
    finally:
        base.OSM_ENDPOINT=original
        session.close()
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
