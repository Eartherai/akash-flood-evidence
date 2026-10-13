"""Export the source-bound optical composite and per-cell date audit offline.

No inference, source acquisition, checking labels or scenario changes occur here.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
import numpy as np
import rasterio

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/optical-composite-2026-10-10'
OUT=ROOT/'outputs/optical-composite-viewer-2026-10-10'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rle(array):
    flat=array.ravel();starts=np.r_[0,np.flatnonzero(flat[1:]!=flat[:-1])+1];ends=np.r_[starts[1:],len(flat)]
    return np.column_stack((flat[starts],ends-starts)).astype('int64').ravel().tolist()
def main():
    if OUT.exists():raise FileExistsError('Preserve previous viewer; choose a new version')
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    if manifest['schema']!='akash-optical-composite-v1' or manifest['reference_labels_loaded'] or manifest['local_debris_accuracy_established']:
        raise ValueError('Wrong composite scope')
    for name,binding in manifest['outputs'].items():
        p=SOURCE/name
        if p.stat().st_size!=binding['bytes'] or sha(p)!=binding['sha256']:raise ValueError('Composite output changed: '+name)
    names={'aoi':'aoi-mask.tif','valid':'valid-mask.tif','candidate':'surface-change-candidates.tif',
           'before_source':'before-selected-source-index.tif','after_source':'after-selected-source-index.tif',
           'before_age':'before-selected-age-days.tif','after_age':'after-selected-age-days.tif'}
    arrays={};grid=None
    for key,name in names.items():
        with rasterio.open(SOURCE/name) as d:
            identity=(str(d.crs),tuple(d.transform),d.shape)
            if grid and identity!=grid:raise ValueError('Grid mismatch')
            grid=identity;arrays[key]=d.read(1)
    with rasterio.open(ROOT/'outputs/s2-surface-change-resumed/valid-mask.tif') as d:
        if (str(d.crs),tuple(d.transform),d.shape)!=grid:raise ValueError('Original grid mismatch')
        arrays['original_valid']=d.read(1)
    aoi=arrays['aoi']==1;valid=aoi&(arrays['valid']==1);old=aoi&(arrays['original_valid']==1);candidate=valid&(arrays['candidate']==1)
    counts={'aoi':int(aoi.sum()),'assessed':int(valid.sum()),'unknown':int((aoi&~valid).sum()),'candidates':int(candidate.sum()),
            'original_assessed':int(old.sum()),'added':int((valid&~old).sum()),'lost':int((old&~valid).sum())}
    expected=manifest['counts']
    if [counts[k] for k in ['aoi','assessed','unknown','candidates']]!=[expected[k] for k in ['aoi_pixels','joint_assessed_pixels','unknown_pixels','candidate_pixels']] or counts['lost']:
        raise ValueError('Recount or retained-support mismatch')
    protocol=json.loads((SOURCE/'protocol.json').read_text())
    data={'schema':'akash-optical-cell-audit-v1','counts':counts,'grid':manifest['grid'],
          'sources':[{k:s[k] for k in ['source_index','branch','item_id','datetime']} for s in protocol['ordered_sources']],
          'layers':{k:rle(v) for k,v in arrays.items()},'ages':manifest['assessed_age_days_histograms'],
          'source_manifest_sha256':sha(SOURCE/'manifest.json'),'protocol_sha256':sha(SOURCE/'protocol.json'),
          'runtime':{k:manifest[k] for k in ['new_body_bytes','wall_seconds','cpu_seconds','peak_process_rss_bytes','acquisition_stop_reason']},
          'scope':manifest['scope'],'limitations':manifest['limitations']}
    OUT.mkdir();(OUT/'data.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
    shutil.copyfile(ROOT/'web/optical-composite-audit.html',OUT/'index.html')
    bindings={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE/'manifest.json',SOURCE/'protocol.json',ROOT/'outputs/s2-surface-change-resumed/valid-mask.tif',Path(__file__),ROOT/'web/optical-composite-audit.html']}
    proof={'created_at':datetime.now(timezone.utc).isoformat(),'status':'recounts_and_source_hashes_passed_browser_review_pending',
           'counts':counts,'sources':bindings,'outputs':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.iterdir()},
           'inference_run':False,'reference_labels_loaded':False,'final_test_rescored':False,'visual_review':'pending'}
    (OUT/'verification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'viewer':str(OUT),'counts':counts,'json_bytes':(OUT/'data.json').stat().st_size},indent=2))
if __name__=='__main__':main()
