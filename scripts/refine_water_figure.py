"""Render saved water labels legibly, preserving the original figure/manifest.

No model inference, training, label evaluation or changes to raster values.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    folder = ROOT / 'outputs/sen1-water-rasuwa'
    manifest = folder / 'manifest.json'
    record = json.loads(manifest.read_text())
    for name, item in record['outputs'].items():
        if sha(folder / name) != item['sha256']:
            raise ValueError('Saved outputs changed')
    history = ROOT / 'outputs/sen1-water-render-revisions/original'
    history.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest, history / 'manifest.json')
    shutil.copy2(folder / 'overview.png', history / 'overview.png')
    arrays = {}
    for name in ['before-water.tif', 'after-water.tif', 'new-water-candidates.tif', 'lost-water-candidates.tif', 'aoi-mask.tif', 'valid-mask.tif']:
        with rasterio.open(folder / name) as ds:
            arrays[name] = ds.read(1)
            transform = ds.transform
    aoi = arrays['aoi-mask.tif'] == 1
    rows, cols = np.nonzero(aoi)
    minx = (transform.c + cols.min() * 10) / 1000
    maxx = (transform.c + (cols.max() + 1) * 10) / 1000
    maxy = (transform.f - rows.min() * 10) / 1000
    miny = (transform.f - (rows.max() + 1) * 10) / 1000
    height, width = aoi.shape
    extent = [transform.c / 1000, (transform.c + width * 10) / 1000,
              (transform.f - height * 10) / 1000, transform.f / 1000]
    colors = ListedColormap(['#e9eee9', '#007eaa', '#abb7bb'])
    norm = BoundaryNorm([-.5,.5,1.5,2.5],3)
    fig, axes = plt.subplots(1,3,figsize=(12,7))
    for ax, date in zip(axes[:2],['before','after']):
        labels = arrays[date+'-water.tif']
        display = np.where(labels == 255, 2, labels).astype('float32')
        display[~aoi] = np.nan
        ax.imshow(display,cmap=colors,norm=norm,extent=extent,interpolation='nearest')
        ax.set_title(date.capitalize()+': '+record['source_dates'][date][:10],loc='left',fontsize=11)
    change = np.zeros(aoi.shape,dtype='float32')
    change[arrays['new-water-candidates.tif']==1] = 1
    change[arrays['lost-water-candidates.tif']==1] = 2
    change[aoi & (arrays['valid-mask.tif']==0)] = 3
    change[~aoi] = np.nan
    axes[2].imshow(change,cmap=ListedColormap(['#e9eee9','#007eaa','#e6a455','#abb7bb']),
                   norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4),extent=extent,interpolation='nearest')
    axes[2].set_title('Temporal model changes',loc='left',fontsize=11)
    for ax in axes:
        ax.set_xlim(minx-.2,maxx+.2);ax.set_ylim(miny-.2,maxy+.2)
        ax.set_xlabel('UTM45N easting (km)',fontsize=9);ax.set_ylabel('Northing (km)',fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(5));ax.tick_params(labelsize=8)
    fig.suptitle('Akash | CPU trained water-state candidates',x=.06,ha='left',fontsize=17,fontweight='bold')
    fig.legend(handles=[Patch(color='#e9eee9',label='Nonwater / no temporal change'),
                        Patch(color='#007eaa',label='Water / gained-water candidate'),
                        Patch(color='#e6a455',label='Lost-water candidate'),Patch(color='#abb7bb',label='Unknown')],
               loc='lower center',ncol=2,fontsize=10,bbox_to_anchor=(.5,.15),frameon=False)
    c=record['counts']
    footer=(f"Joint coverage: {c['assessed_pixels']:,}/{c['aoi_pixels']:,} pixels; gained-water candidates: {c['new_water_candidate_pixels']:,}.\n"
            'Permanent and flood water inseparable. Model change is not confirmed flooding, debris or damage.\n'
            '10m grid; 3x3/7x7 context. Local accuracy/geolocation unverified. Copernicus DEM; documented local SNAP correction.')
    fig.text(.06,.045,footer,fontsize=9,linespacing=1.5)
    fig.subplots_adjust(top=.88,bottom=.31,left=.06,right=.98,wspace=.30)
    fig.savefig(folder/'overview.png',dpi=140,facecolor='white');plt.close(fig)
    record['outputs']['overview.png']={'sha256':sha(folder/'overview.png'),'bytes':(folder/'overview.png').stat().st_size}
    record['rendering_revision']={'at':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(Path(__file__)),
        'original_manifest_sha256':sha(history/'manifest.json'),'original_figure_sha256':sha(history/'overview.png'),
        'change':'Display AOI bounds and move legend away from labels; no raster, model or inference change',
        'model_inference_repeated':False}
    manifest.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record['rendering_revision'],indent=2))


if __name__=='__main__':main()
