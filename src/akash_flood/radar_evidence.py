"""Conservative change screening on matched, noise-corrected SNAP products.

This detects radar changes and darkening candidates. It does not assign a
landslide/flood label or infer a road's physical passability.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt, uniform_filter

BANDS = ('Sigma0_VV', 'Sigma0_VH', 'elevation', 'localIncidenceAngle',
         'projectedLocalIncidenceAngle', 'incidenceAngleFromEllipsoid', 'layoverShadowMask')
RECIPE = {
    'description': 'Same-orbit calibrated sigma0 change screening with explicit geometry and coverage masks',
    'semantics': 'radar_change_candidate',
    'aoi_bbox': [85.30, 28.12, 85.43, 28.32],
    'smooth_window_pixels': 7,
    'minimum_joint_geometry_samples': 40,
    'coverage_edge_buffer_m': 200,
    'change_db_threshold': 3.0,
    'change_rule': 'Absolute VV and VH mean-power changes >=3 dB, with matching sign',
    'water_rule': 'VV after<-17 dB, VH after<-23 dB; both changes<-3 dB; before not jointly dark',
    'water_vv_dark_db': -17.0,
    'water_vh_dark_db': -23.0,
    'terrain_rule': 'Zero layover/shadow codes on both dates; positive DEM; all three angles in (0,90)',
    'averaging_rule': 'Unclipped noise-corrected powers, including negative samples, averaged on joint geometry support; nonpositive means excluded before logarithms',
    'radiometric_terrain_flattening': False,
    'independent_registration_validation': False,
    'thresholds_chosen_without_reference_labels': True,
    'not_established': ['flood or landslide classification accuracy', 'physical road passability',
                        'absolute geolocation accuracy', 'radiometric terrain flattening'],
}


def read_snap(path):
    path = Path(path)
    doc = ET.parse(path)
    names = {band.findtext('BAND_NAME') for band in doc.findall('.//Spectral_Band_Info')}
    if set(BANDS) - names:
        raise ValueError('SNAP output lacks required observation/terrain bands')
    arrays, grid = {}, None
    for name in BANDS:
        with rasterio.open(path.with_suffix('.data') / (name + '.img')) as ds:
            current = (ds.crs, ds.transform, ds.width, ds.height)
            if ds.count != 1 or ds.crs != rasterio.crs.CRS.from_epsg(32645):
                raise ValueError('Expected single-band EPSG:32645 SNAP data')
            if grid is not None and current != grid:
                raise ValueError('Bands within one scene differ in geometry')
            grid = current
            arrays[name] = ds.read(1)
    attributes = {e.attrib['name']: e.text for e in doc.findall('.//MDElem[@name="Abstracted_Metadata"]/MDATTR')}
    if not (attributes.get('REL_ORBIT') or '').isdigit() or attributes.get('PASS') not in ['ASCENDING','DESCENDING']:
        raise ValueError('Expected explicit relative orbit and pass metadata')
    if attributes.get('abs_calibration_flag') != '1':
        raise ValueError('SNAP calibration flag is absent')
    return arrays, grid, attributes


def integer_overlap(left, right):
    """Exact common windows, refusing subpixel transforms or resampling."""
    crs, a, width, height = left
    other_crs, b, other_width, other_height = right
    if crs != other_crs or a.a != 10 or a.e != -10 or a.b != 0 or a.d != 0:
        raise ValueError('Expected north-up 10 m matching CRS')
    if (b.a, b.b, b.d, b.e) != (a.a, a.b, a.d, a.e):
        raise ValueError('Resolution/rotation differs')
    dx, dy = (b.c-a.c)/10, (a.f-b.f)/10
    if not np.isclose(dx, round(dx), atol=1e-8) or not np.isclose(dy, round(dy), atol=1e-8):
        raise ValueError('Subpixel origin shift; cannot copy without resampling')
    dx, dy = round(dx), round(dy)
    col, row = max(0, dx), max(0, dy)
    stop_col, stop_row = min(width, dx+other_width), min(height, dy+other_height)
    if stop_col <= col or stop_row <= row:
        raise ValueError('Scenes have no common grid cells')
    w = Window(col, row, stop_col-col, stop_row-row)
    v = Window(col-dx, row-dy, w.width, w.height)
    transform = rasterio.windows.transform(w, a)
    return w, v, (crs, transform, int(w.width), int(w.height))


def crop_arrays(arrays, window):
    row, col = int(window.row_off), int(window.col_off)
    return {name: value[row:row+int(window.height), col:col+int(window.width)] for name, value in arrays.items()}


def geometry_support(scene):
    """Mask-zero alone is insufficient: SNAP leaves zeros at missing coverage."""
    shape = scene['Sigma0_VV'].shape
    if any(scene[name].shape != shape for name in BANDS):
        raise ValueError('Band array dimensions differ')
    mask = scene['layoverShadowMask']
    if not np.all(np.isin(mask, [0, 1, 2, 3])):
        raise ValueError('Unexpected layover/shadow encoding')
    valid = mask == 0
    for name in ('Sigma0_VV', 'Sigma0_VH'):
        # Zero is SNAP's declared no-data value; negative noise-subtracted
        # observations are retained for unbiased spatial averaging.
        valid &= np.isfinite(scene[name]) & (scene[name] != 0)
    valid &= np.isfinite(scene['elevation']) & (scene['elevation'] > 0)
    for name in ('localIncidenceAngle', 'projectedLocalIncidenceAngle', 'incidenceAngleFromEllipsoid'):
        valid &= np.isfinite(scene[name]) & (scene[name] > 0) & (scene[name] < 90)
    return valid


def screen(before, after, *, aoi=None, window=7, minimum_samples=40, edge_buffer_pixels=20):
    if window < 1 or window % 2 != 1 or not 1 <= minimum_samples <= window*window:
        raise ValueError('Invalid spatial support recipe')
    support = geometry_support(before) & geometry_support(after)
    if aoi is None:
        aoi = np.ones(support.shape, dtype=bool)
    if aoi.shape != support.shape or edge_buffer_pixels < 0:
        raise ValueError('Invalid AOI/buffer')
    count = uniform_filter(support.astype(np.float64), size=window, mode='constant', cval=0) * window**2
    means = {}
    valid = support & aoi & (count >= minimum_samples-1e-8)
    if edge_buffer_pixels:
        # Buffer the observation footprint, not layover/shadow holes: those
        # holes already limit the neighborhood's joint sample count.
        coverage = np.ones(support.shape, dtype=bool)
        for scene in (before, after):
            for name in ('Sigma0_VV','Sigma0_VH','elevation'):
                coverage &= np.isfinite(scene[name]) & (scene[name] != 0)
        distance = distance_transform_edt(np.pad(coverage,1,constant_values=False))[1:-1,1:-1]
        valid &= distance >= edge_buffer_pixels
    for date, scene in [('before', before), ('after', after)]:
        for pol in ('VV', 'VH'):
            power = scene['Sigma0_'+pol]
            total = uniform_filter(np.where(support, power, 0).astype(np.float64), size=window, mode='constant', cval=0) * window**2
            mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
            valid &= np.isfinite(mean) & (mean > 0)
            means[date+'_'+pol] = mean
    db = {name: np.full(support.shape, np.nan, dtype=np.float32) for name in means}
    for name, mean in means.items():
        np.log10(mean, out=db[name], where=valid)
        db[name][valid] *= 10
    delta_vv, delta_vh = db['after_VV']-db['before_VV'], db['after_VH']-db['before_VH']
    change = valid & (np.abs(delta_vv) >= 3) & (np.abs(delta_vh) >= 3) & (delta_vv*delta_vh > 0)
    dark_before = (db['before_VV'] < -17) & (db['before_VH'] < -23)
    water = valid & ~dark_before & (db['after_VV'] < -17) & (db['after_VH'] < -23) & (delta_vv < -3) & (delta_vh < -3)
    counts = {'aoi_pixels':int(aoi.sum()), 'assessed_pixels':int(valid.sum()),
              'unassessed_pixels':int((aoi & ~valid).sum()),
              'change_candidate_pixels':int(change.sum()), 'new_water_candidate_pixels':int(water.sum()),
              'geometry_supported_pixels':int((support & aoi).sum()),
              'before_negative_power_samples':int(((before['Sigma0_VV']<0)|(before['Sigma0_VH']<0))[aoi].sum()),
              'after_negative_power_samples':int(((after['Sigma0_VV']<0)|(after['Sigma0_VH']<0))[aoi].sum())}
    return {'valid':valid,'geometry_support':support,'means':means,'db':db,'delta_vv':delta_vv,
            'delta_vh':delta_vh,'change':change,'water':water,'counts':counts}
