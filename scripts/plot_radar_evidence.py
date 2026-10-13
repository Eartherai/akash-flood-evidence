"""Scientific overview of saved observations/candidates; no generated imagery."""
import argparse
import json
from pathlib import Path
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args()
    report=json.loads((args.folder/'manifest.json').read_text())
    arrays={}
    for name in ('aoi-mask','valid-mask','before_vv-mean-db','after_vv-mean-db','delta-vv-db','change-candidates','water-candidates'):
        with rasterio.open(args.folder/(name+'.tif')) as ds:
            arrays[name]=ds.read(1);bounds=ds.bounds
    inside=arrays['aoi-mask']>0;rows,cols=np.where(inside)
    row=slice(rows.min(),rows.max()+1);col=slice(cols.min(),cols.max()+1)
    extent=[bounds.left+cols.min()*10,bounds.left+(cols.max()+1)*10,
            bounds.top-(rows.max()+1)*10,bounds.top-rows.min()*10]
    valid=(arrays['valid-mask']>0)[row,col];inside=inside[row,col]
    fig,axes=plt.subplots(2,3,figsize=(13.2,10.2),layout='constrained')
    fig.suptitle('Rasuwa | actual Sentinel-1 observations and candidate screens',fontsize=17,fontweight='bold')
    dates=report['source_dates'];before=dates['before'].split('T')[0];after=dates['after'].split('T')[0]
    specs=[('before_vv-mean-db',f'Before: {before} | mean VV sigma0','gray',-30,0,'dB'),
           ('after_vv-mean-db',f'After: {after} | mean VV sigma0','gray',-30,0,'dB'),
           ('delta-vv-db','After minus before | VV mean power','RdBu_r',-6,6,'dB')]
    for ax,(key,title,cmap,low,high,unit) in zip(axes.flat,specs):
        colors=plt.get_cmap(cmap).copy();colors.set_bad('#bfc3c5')
        arr=np.ma.masked_where(~valid,arrays[key][row,col])
        img=ax.imshow(arr,extent=extent,cmap=colors,vmin=low,vmax=high,interpolation='nearest')
        ax.set_title(title,fontsize=10);fig.colorbar(img,ax=ax,shrink=.65,label=unit)
    state=np.where(valid,1,0).astype(float);state[~inside]=np.nan
    ax=axes[1,0];colors=ListedColormap(['#bfc3c5','#277b72']);colors.set_bad('white')
    ax.imshow(state,extent=extent,cmap=colors,vmin=0,vmax=1,interpolation='nearest')
    count=report['counts'];percentage=100*count['assessed_pixels']/count['aoi_pixels']
    ax.set_title(f'Assessed {percentage:.1f}% | green; unassessed | grey',fontsize=10)
    for ax,name,color,title in [(axes[1,1],'change-candidates','#b94a3e','Same-sign VV/VH change candidates'),
                                 (axes[1,2],'water-candidates','#247abd','New darkening/water-screen candidates')]:
        arr=arrays[name][row,col].astype(float);arr[~inside]=np.nan
        arr[arr==255]=2
        colors=ListedColormap(['#f1f1ee',color,'#bfc3c5']);colors.set_bad('white')
        ax.imshow(arr,extent=extent,cmap=colors,vmin=0,vmax=2,interpolation='nearest')
        ax.set_title(title,fontsize=10)
    for ax in axes.flat:
        ax.set_xlabel('UTM easting (m)',fontsize=8);ax.set_ylabel('UTM northing (m)',fontsize=8)
        ax.ticklabel_format(axis='both',style='plain',useOffset=False);ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    dem = report['preprocessing']['dem']
    dem_label = ('Copernicus GLO-30, explicit EGM2008 → ellipsoid conversion'
                 if 'Copernicus' in dem else 'SRTM + EGM96 (engineering check only)')
    fig.supxlabel('10 m grid, 7 × 7 power averaging. Grey pixels are unassessed. Candidates are not confirmed flood/landslide labels.\n'
                  + dem_label + '; embedded orbit vectors; no terrain flattening or road passability inference.',fontsize=9)
    fig.savefig(args.folder/'overview.png',dpi=160)
    fig.savefig(args.folder/'overview.svg')
    plt.close(fig)
    print(args.folder/'overview.png')

if __name__=='__main__':main()
