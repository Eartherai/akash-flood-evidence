"""Apply our frozen CPU water model to source-bound corrected SNAP powers.

Run with scikit-learn==1.7.2 and matplotlib==3.10.8. No reference labels load.
The joblib checkpoint must be the locally trained artifact bound to its freeze.
"""
from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Polygon, mapping

from akash_flood.flood_model import FEATURE_NAMES
from akash_flood.radar_evidence import read_snap, integer_overlap, crop_arrays
from akash_flood.water_inference import infer_pair

ROOT = Path(__file__).resolve().parents[1]
TRAINING_PROTOCOL_SHA = '57dc4cc412d09ef06889a29d6f3a7956d1c5666dbf57c58750b77575e3b749d2'
INFERENCE_PROTOCOL_SHA = '8f0d02e996d1a750238900d379c9ba0982903337d99903ff5ef15d621a86f977'
SCOPE = 'Supervised water-state candidates; permanent and flood water inseparable; Rasuwa accuracy not established'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(path):
    p = Path(path)
    return {'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def require_record(path, value):
    if Path(value['path']).resolve() != Path(path).resolve() or value['sha256'] != sha(path) or value['bytes'] != Path(path).stat().st_size:
        raise ValueError('Input differs from frozen artifact: ' + str(path))


def write_raster(path, array, grid, nodata):
    crs, transform, width, height = grid
    with rasterio.open(path, 'w', driver='GTiff', width=width, height=height,
                       count=1, dtype=array.dtype, crs=crs, transform=transform,
                       nodata=nodata, compress='deflate', tiled=True) as ds:
        ds.write(array, 1)


def plot_overview(path, result, aoi, grid, dates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    labels = ListedColormap(['#e9eee9', '#007eaa', '#abb7bb'])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5], labels.N)
    changes = ListedColormap(['#e9eee9', '#007eaa', '#e6a455', '#abb7bb'])
    changenorm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], changes.N)
    crs, t, width, height = grid
    extent = [t.c / 1000, (t.c + width * t.a) / 1000,
              (t.f + height * t.e) / 1000, t.f / 1000]
    fig, axes = plt.subplots(1, 3, figsize=(12, 7), constrained_layout=False)
    for ax, date in zip(axes[:2], ['before', 'after']):
        values = result[date + '_labels']
        display = np.where(values == 255, 2, values).astype('float32')
        display[~aoi] = np.nan
        ax.imshow(display, cmap=labels, norm=norm, extent=extent, interpolation='nearest')
        ax.set_title(date.capitalize() + ': ' + dates[date][:10], loc='left', fontsize=11)
    display = np.zeros(aoi.shape, dtype='float32')
    display[result['new'] == 1] = 1
    display[result['lost'] == 1] = 2
    display[aoi & ~result['assessed']] = 3
    display[~aoi] = np.nan
    axes[2].imshow(display, cmap=changes, norm=changenorm, extent=extent, interpolation='nearest')
    axes[2].set_title('Temporal classification changes', loc='left', fontsize=11)
    for ax in axes:
        ax.set_xlabel('UTM45N easting (km)', fontsize=9)
        ax.set_ylabel('Northing (km)', fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=8)
    fig.suptitle('Akash | CPU trained water-state candidates', x=.06, ha='left', fontsize=17, fontweight='bold')
    fig.legend(handles=[Patch(color='#e9eee9', label='Predicted nonwater / no temporal change'),
                        Patch(color='#007eaa', label='Predicted water / gained water candidate'),
                        Patch(color='#e6a455', label='Lost water candidate'),
                        Patch(color='#abb7bb', label='Unknown')], loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(.5, .13), frameon=False)
    c = result['counts']
    footer = (f"Joint assessed coverage: {c['assessed_pixels']:,}/{c['aoi_pixels']:,} pixels; "
              f"new-water candidates: {c['new_water_candidate_pixels']:,}. 10m grid; model uses 3x3/7x7 context.\n"
              'Permanent and flood water inseparable. Temporal model change is not confirmed flooding or damage.\n'
              'Rasuwa accuracy and absolute geolocation not established. Copernicus DEM; local SNAP external-DEM sampling correction.')
    fig.text(.06, .045, footer, ha='left', fontsize=9, linespacing=1.5)
    fig.subplots_adjust(top=.89, bottom=.24, left=.06, right=.98, wspace=.30)
    fig.savefig(path, dpi=140, facecolor='white'); plt.close(fig)


def main():
    import joblib
    import sklearn
    from threadpoolctl import threadpool_limits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'outputs/sen1-cpu-v1/water-model.joblib')
    parser.add_argument('--model-freeze', type=Path, default=ROOT / 'outputs/sen1-cpu-v1/model-evaluation-freeze.json')
    parser.add_argument('--inference-protocol', type=Path, default=ROOT / 'research/sen1-rasuwa-inference-protocol.json')
    parser.add_argument('--processing-record', type=Path, default=ROOT / 'outputs/snap-copdem-pixel-corrected/run-metadata.json')
    parser.add_argument('--before', type=Path, default=ROOT / 'outputs/snap-copdem-pixel-corrected/before-terrain.dim')
    parser.add_argument('--after', type=Path, default=ROOT / 'outputs/snap-copdem-pixel-corrected/after-terrain.dim')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/sen1-water-rasuwa')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve existing model maps; use a new, explicitly documented output')
    start = time.perf_counter()
    freeze = json.loads(args.model_freeze.read_text())
    require_record(args.model, freeze['model'])
    protocol_path = ROOT / 'research/sen1-cpu-training-protocol.json'
    require_record(protocol_path, freeze['protocol'])
    protocol = json.loads(protocol_path.read_text())
    if sha(protocol_path) != TRAINING_PROTOCOL_SHA or protocol['features'] != list(FEATURE_NAMES) or freeze['threshold'] != .5:
        raise ValueError('Frozen training protocol/features differ')
    if sklearn.__version__ != freeze['scikit_learn_version'] or sklearn.__version__ != '1.7.2':
        raise ValueError('Use the original trained sklearn version 1.7.2')
    for item in freeze['code']:
        require_record(Path(item['path']), item)
    processing = json.loads(args.processing_record.read_text())
    if not processing.get('completed_at') or processing['processing'].get('SRTM_used') is not False or not processing.get('dem_reader_patch'):
        raise ValueError('Require completed, locally corrected Copernicus DEM processing')
    for date, dim in [('before', args.before), ('after', args.after)]:
        known = {Path(item['path']).resolve(): item for item in processing['scenes'][date]['artifacts']}
        for path in [dim] + [p for p in dim.with_suffix('.data').glob('*') if p.suffix in ('.img', '.hdr')]:
            require_record(path, known[path.resolve()])
    if sha(args.inference_protocol) != INFERENCE_PROTOCOL_SHA:
        raise ValueError('Frozen inference policy changed')
    policy = json.loads(args.inference_protocol.read_text())
    if policy['training_protocol_sha256'] != TRAINING_PROTOCOL_SHA or policy['threshold'] != .5 or policy['minimum_joint_positive_geometry_samples_7x7'] != 40 or policy['coverage_edge_buffer_m'] != 200:
        raise ValueError('Inference policy changed')
    bbox = policy['aoi_bbox']; west, south, east, north = bbox
    if not -180 < west < east < 180 or not -90 < south < north < 90:
        raise ValueError('Invalid AOI bounds')
    before, left, bmeta = read_snap(args.before); after, right, ameta = read_snap(args.after)
    for key in ('REL_ORBIT', 'PASS', 'MISSION', 'ACQUISITION_MODE'):
        if bmeta.get(key) != ameta.get(key):
            raise ValueError('Radar orbit/sensor metadata differs')
    w, v, grid = integer_overlap(left, right)
    before = crop_arrays(before, w); after = crop_arrays(after, v)
    tx = Transformer.from_crs(4326, grid[0], always_xy=True)
    perimeter = []
    for p, q in [((west,south),(east,south)), ((east,south),(east,north)),
                 ((east,north),(west,north)), ((west,north),(west,south))]:
        for f in np.linspace(0, 1, 41, endpoint=False):
            perimeter.append(tx.transform(p[0] + f * (q[0]-p[0]), p[1] + f * (q[1]-p[1])))
    aoi = geometry_mask([mapping(Polygon(perimeter))], out_shape=before['Sigma0_VV'].shape,
                        transform=grid[1], invert=True, all_touched=False)
    # Load only our hash-bound local artifact after all provenance checks.
    model = joblib.load(args.model)
    args.out.mkdir(parents=True)
    with threadpool_limits(limits=4):
        result = infer_pair(model, before, after, aoi)
    for date in ('before', 'after'):
        write_raster(args.out / (date + '-water.tif'), result[date + '_labels'], grid, 255)
        write_raster(args.out / (date + '-water-score.tif'), np.where(result['assessed'], result[date + '_score'], -9999).astype('float32'), grid, -9999)
    for name in ('new', 'lost'):
        write_raster(args.out / (name + '-water-candidates.tif'), result[name], grid, 255)
    write_raster(args.out / 'aoi-mask.tif', aoi.astype('uint8'), grid, 0)
    write_raster(args.out / 'valid-mask.tif', result['assessed'].astype('uint8'), grid, 0)
    dates = {date: json.loads((ROOT / 'data/s1-snap-native/metadata' / (date + '.stac.json')).read_text())['properties']['datetime'] for date in ('before', 'after')}
    plot_overview(args.out / 'overview.png', result, aoi, grid, dates)
    report = {
        'schema': 'akash-trained-water-v1', 'available': True, 'scope': SCOPE,
        'created_at': datetime.now(timezone.utc).isoformat(), 'requested_bbox': bbox,
        'source_dates': dates, 'counts': result['counts'],
        'grid': {'crs': str(grid[0]), 'transform': list(grid[1]), 'shape': [grid[3], grid[2]],
                 'resolution_m': 10, 'resampling_used_for_alignment': False},
        'model': {**record(args.model), 'threshold': .5, 'scikit_learn_version': sklearn.__version__,
                  'freeze_sha256': sha(args.model_freeze), 'protocol_sha256': TRAINING_PROTOCOL_SHA,
                  'training_dataset': 'Sen1Floods11 S1Hand + LabelHand', 'score_calibrated': False},
        'inference_protocol': record(args.inference_protocol),
        'code': [record(Path(__file__)), record(ROOT / 'src/akash_flood/water_inference.py'),
                 record(ROOT / 'src/akash_flood/flood_model.py'), record(ROOT / 'src/akash_flood/radar_evidence.py')],
        'inputs': [record(args.before), record(args.after)],
        'preprocessing': {'processing_record_sha256': sha(args.processing_record),
                          'processing_record_path': str(args.processing_record),
                          'dem_sampling_correction': processing['dem_reader_patch'],
                          'dem': 'Copernicus GLO30, EGM2008 to WGS84 ellipsoid explicitly converted',
                          'input_units': 'raw calibrated noise-corrected sigma0 linear power -> 10log10 once',
                          'radiometric_terrain_flattening': False,
                          'orbit': 'embedded source state vectors'},
        'label_encoding': {'0': 'predicted nonwater', '1': 'predicted water', '255': 'unknown'},
        'temporal_change_semantics': 'After1/before0 gained-water and before1/after0 lost-water model candidates, never confirmed flooding',
        'outputs': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in args.out.iterdir() if p.is_file()},
        'reference_labels_loaded': False, 'gpu_used': False, 'account_used': False,
        'wall_seconds': time.perf_counter() - start,
        'limitations': [
            'Water handlabels combine permanent water and flooding; the trained model does not identify debris.',
            'Held-out dataset water metrics do not establish Himalayan or Rasuwa hazard accuracy.',
            'GEE training and local SNAP preprocessing, projection and terrain differ.',
            'Nonpositive raw powers and unsupported geometry remain unknown; no epsilon substitution.',
            '10m is output grid spacing; 3x3/7x7 context reduces independent detail.',
            'Classifier scores are not calibrated probabilities.',
            'Temporal classification change is not observed new flooding, road damage, or a road closure.',
            'Absolute geolocation, radiometric terrain flattening and model transfer require separate validation.',
        ],
    }
    (args.out / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'counts': report['counts'], 'wall_seconds': report['wall_seconds'],
                      'manifest': record(args.out / 'manifest.json')}, indent=2))


if __name__ == '__main__':
    main()
