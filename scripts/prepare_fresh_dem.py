"""Prepare source-bound Copernicus DEM for a new processing extent on CPU.

Full provider tiles are bounded to four and 400MB total source transfer. Cached
Rasuwa bytes may be reused only after their original manifest hashes match.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pyproj

from akash_flood.catalog import validate_bbox
from akash_flood.fresh_dem import mosaic_orthometric
from akash_flood.terrain import tile_urls
from prepare_copdem_ellipsoid import (DownloadBudget, convert_dem, file_record, sha256,
                                     GRID_SHA256, GRID_BYTES, GRID_URL, LICENSE_URL)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bbox', nargs=4, type=float, required=True, help='Processing extent, including SAR terrain context')
    parser.add_argument('--aoi-bbox', nargs=4, type=float, help='Requested map AOI; defaults to processing extent')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--budget-bytes', type=int, default=400_000_000)
    parser.add_argument('--geoid', type=Path, default=ROOT / 'data/copdem-ellipsoid/us_nga_egm08_25.tif')
    args = parser.parse_args()
    bbox = validate_bbox(args.bbox); aoi = validate_bbox(args.aoi_bbox or args.bbox)
    if bbox[0] > aoi[0] or bbox[1] > aoi[1] or bbox[2] < aoi[2] or bbox[3] < aoi[3]:
        raise ValueError('Processing context must cover requested AOI')
    # Two source sample halos are acquired, including across degree boundaries.
    urls = tile_urls([bbox[0]-.001,bbox[1]-.001,bbox[2]+.001,bbox[3]+.001])
    if args.out.exists() or not 1 <= args.budget_bytes <= 500_000_000:
        raise ValueError('Require fresh output and bounded download budget')
    args.out.mkdir(parents=True)
    budget = DownloadBudget(args.budget_bytes)
    manifest = {'schema':'akash-fresh-dem-v1','started_at':datetime.now(timezone.utc).isoformat(),
                'requested_bbox':aoi,'context_bbox':bbox,'code':[file_record(Path(__file__)),file_record(ROOT/'src/akash_flood/fresh_dem.py'),file_record(ROOT/'scripts/prepare_copdem_ellipsoid.py')],
                'pyproj_version':pyproj.__version__,'source_tiles':[],
                'dem_source':'Copernicus GLO30 public provider COG tiles; EGM2008 orthometric heights',
                'download_budget_bytes':budget.maximum,'snap_settings':{'externalDEMApplyEGM':False,'externalDEMNoDataValue':-32767.0},
                'reference_labels_loaded':False,'SRTM_used':False,'gpu_used':False}
    def save():
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    save()
    cached_path = ROOT/'data/copdem-ellipsoid/manifest.json'
    cached = json.loads(cached_path.read_text()) if cached_path.is_file() else None
    sources = []
    for index,url in enumerate(urls):
        path=args.out/f'copdem-source-{index}.tif'
        prior=cached['downloads'].get('dem_orthometric') if cached else None
        if prior and prior['url']==url and Path(prior['path']).is_file() and sha256(prior['path'])==prior['sha256'] and Path(prior['path']).stat().st_size==prior['bytes']:
            shutil.copy2(prior['path'],path)
            item={**file_record(path),'url':url,'reused_verified_source':prior,'network_bytes':0}
        else:
            item=budget.download(url,path,65_000_000)
        manifest['source_tiles'].append(item);sources.append(path);save()
    geoid=args.geoid
    if not geoid.is_file():
        geoid=args.out/'us_nga_egm08_25.tif'
        grid_item=budget.download(GRID_URL,geoid,85_000_000,GRID_SHA256,GRID_BYTES)
    else:
        if geoid.stat().st_size!=GRID_BYTES or sha256(geoid)!=GRID_SHA256:
            raise ValueError('Mandatory NGA EGM2008 grid differs from reviewed official source')
        grid_item={**file_record(geoid),'url':GRID_URL,'reused_verified_source':True,'network_bytes':0}
    manifest['geoid']={**grid_item,'license':'Public Domain; official NGA/PROJ README','purpose':'Mandatory geodetic auxiliary; no hazard labels'}
    orthometric=args.out/'copdem-mosaic-orthometric.tif'
    manifest['horizontal_mosaic']=mosaic_orthometric(sources,bbox,orthometric)
    manifest['orthometric_mosaic']=file_record(orthometric);save()
    ellipsoid=args.out/'copdem-mosaic-ellipsoid.tif'
    diagnostics=convert_dem(orthometric,geoid,ellipsoid)
    diagnostic_path=args.out/'conversion-diagnostics.json'
    diagnostic_path.write_text(json.dumps(diagnostics,indent=2,allow_nan=False)+'\n')
    manifest['diagnostics']={**file_record(diagnostic_path),'independent_sample_count':diagnostics['independent_sample_count'],
                             'max_proj_vs_independent_error_m':diagnostics['max_proj_vs_independent_error_m'],
                             'max_float32_vs_independent_error_m':diagnostics['max_float32_vs_independent_error_m']}
    manifest['output']={**file_record(ellipsoid),'vertical_datum':'WGS84 ellipsoidal height','horizontal_crs':'EPSG:4326',
                        'nodata':-32767.0,'dtype':'float32','compression':None,'formula':diagnostics['formula']}
    manifest['network_downloaded_bytes']=budget.used
    manifest['completed_at']=datetime.now(timezone.utc).isoformat()
    manifest['source_license_url']=LICENSE_URL
    manifest['limitations']=['Copernicus DSM includes vegetation/building heights.',
        'Native centers copied when aligned; mixed angular grids explicitly harmonized with bilinear interpolation.',
        'Datum conversion is numerically checked, not a validation of absolute topography or SAR geolocation.',
        'Only up to4tiles, latitude-60to80 and4million geographic mosaic cells are supported.',
        'SAR must use the documented local external-DEM sampling correction with EGM application disabled.']
    save();print(json.dumps({'output':manifest['output'],'source_tiles':len(sources),'network_downloaded_bytes':budget.used,
                            'horizontal_resampling_used':manifest['horizontal_mosaic']['horizontal_resampling_used']},indent=2))


if __name__=='__main__':main()
