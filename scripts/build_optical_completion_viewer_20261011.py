"""Render saved acquisition closeout; no inference or checking source is loaded."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import numpy as np
import rasterio
import build_optical_composite_viewer as base

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/optical-completion-2026-10-11'
PRIOR = ROOT / 'outputs/optical-composite-2026-10-10'
OUT = ROOT / 'outputs/optical-completion-viewer-2026-10-11'

def main():
    base.SOURCE = SOURCE
    base.OUT = OUT
    base.main()
    with rasterio.open(PRIOR / 'valid-mask.tif') as d:
        prior = d.read(1)
        grid = (str(d.crs), tuple(d.transform), d.shape)
    with rasterio.open(SOURCE / 'valid-mask.tif') as d:
        if (str(d.crs), tuple(d.transform), d.shape) != grid:
            raise ValueError('Prior composite grid differs')
        valid = d.read(1)
    with rasterio.open(SOURCE / 'aoi-mask.tif') as d:
        aoi = d.read(1) == 1
    old = aoi & (prior == 1)
    new = aoi & (valid == 1)
    if np.any(old & ~new):
        raise ValueError('Original assessed support was lost')
    gained = new & ~old
    data = json.loads((OUT / 'data.json').read_text())
    data['layers']['prior_composite_valid'] = base.rle(prior)
    data['completion'] = {'date':'2026-10-11', 'prior_assessed':int(old.sum()),
        'added':int(gained.sum()), 'lost':int((old & ~new).sum()),
        'gain_percentage_points':100 * int(gained.sum()) / int(aoi.sum()),
        'prior_manifest_sha256':base.sha(PRIOR / 'manifest.json'),
        'scope':'Completion of already selected reflectance sources; no date selection or detector change.'}
    if data['completion']['added'] != 4316 or int(new.sum()) != 269363:
        raise ValueError('Completion recount differs from execution')
    (OUT / 'data.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
    shutil.copy2(ROOT / 'web/optical-completion-audit.html', OUT / 'index.html')
    proof = json.loads((OUT / 'verification.json').read_text())
    proof['created_at'] = datetime.now(timezone.utc).isoformat()
    proof['completion'] = data['completion']
    proof['sources'].update({str(p.relative_to(ROOT)):base.sha(p) for p in [
        Path(__file__), ROOT/'web/optical-completion-audit.html', PRIOR/'manifest.json', PRIOR/'valid-mask.tif']})
    proof['outputs'] = {p.name:{'bytes':p.stat().st_size,'sha256':base.sha(p)}
        for p in OUT.iterdir() if p.is_file() and p.name != 'verification.json'}
    (OUT / 'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'viewer':str(OUT),'completion':data['completion']},indent=2))

if __name__ == '__main__':
    main()
