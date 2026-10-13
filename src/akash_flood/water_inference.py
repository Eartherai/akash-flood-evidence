"""Transfer the frozen water model to joint, explicitly supported SAR pixels.

Training handlabels merge permanent and flood water. Temporal classification
changes here remain candidates, without reference-based Rasuwa validation.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

from .flood_model import predict_water
from .radar_evidence import geometry_support


def positive_db(scene):
    """Log individual calibrated powers; never log absent or negative samples."""
    power = np.stack([scene['Sigma0_VV'], scene['Sigma0_VH']])
    positive = np.isfinite(power).all(axis=0) & (power > 0).all(axis=0)
    db = np.full(power.shape, np.nan, dtype='float32')
    np.log10(power, out=db, where=positive[None])
    db[:, positive] *= 10
    return db, positive


def inference_support(before, after, aoi, *, minimum_samples=40, edge_buffer_pixels=20):
    """Keep the existing conservative 7x7 geometry and footprint policy."""
    if aoi.dtype != bool or minimum_samples != 40 or edge_buffer_pixels != 20:
        raise ValueError('Require the frozen boolean AOI and 40/49, 200m support policy')
    before_db, before_positive = positive_db(before)
    after_db, after_positive = positive_db(after)
    support = geometry_support(before) & geometry_support(after) & before_positive & after_positive
    if aoi.shape != support.shape:
        raise ValueError('AOI dimensions differ from observations')
    count = uniform_filter(support.astype('float64'), size=7, mode='constant', cval=0) * 49
    coverage = np.ones(support.shape, dtype=bool)
    for scene in (before, after):
        for name in ('Sigma0_VV', 'Sigma0_VH', 'elevation'):
            coverage &= np.isfinite(scene[name]) & (scene[name] != 0)
    distance = distance_transform_edt(np.pad(coverage, 1, constant_values=False))[1:-1, 1:-1]
    assessed = aoi & support & (count >= 40 - 1e-8) & (distance >= 20)
    return before_db, after_db, support, assessed


def temporal_layers(before_labels, after_labels, aoi):
    if before_labels.shape != after_labels.shape or aoi.shape != before_labels.shape or aoi.dtype != bool:
        raise ValueError('Require matching labels and boolean AOI')
    if not np.isin(before_labels, [0, 1, 255]).all() or not np.isin(after_labels, [0, 1, 255]).all():
        raise ValueError('Expected 0/1/255 water labels')
    if not np.array_equal(before_labels != 255, after_labels != 255):
        raise ValueError('Temporal layers require identical joint assessment domains')
    assessed = (before_labels != 255) & aoi
    if np.any((before_labels != 255) & ~aoi):
        raise ValueError('Published labels must be unknown outside the AOI')
    new = assessed & (after_labels == 1) & (before_labels == 0)
    lost = assessed & (before_labels == 1) & (after_labels == 0)
    unchanged = assessed & (before_labels == 1) & (after_labels == 1)
    counts = {
        'aoi_pixels': int(aoi.sum()), 'assessed_pixels': int(assessed.sum()),
        'unknown_pixels': int((aoi & ~assessed).sum()),
        'before_water_pixels': int(((before_labels == 1) & assessed).sum()),
        'after_water_pixels': int(((after_labels == 1) & assessed).sum()),
        'new_water_candidate_pixels': int(new.sum()),
        'lost_water_candidate_pixels': int(lost.sum()),
        'unchanged_water_pixels': int(unchanged.sum()),
    }
    return (np.where(assessed, new, 255).astype('uint8'),
            np.where(assessed, lost, 255).astype('uint8'), counts)


def infer_pair(model, before, after, aoi):
    before_db, after_db, support, assessed = inference_support(before, after, aoi)
    result = {}
    for date, db in [('before', before_db), ('after', after_db)]:
        score, labels = predict_water(model, db, support)
        labels[~assessed] = 255
        score[~assessed] = np.nan
        result[date + '_score'] = score
        result[date + '_labels'] = labels
    result['new'], result['lost'], result['counts'] = temporal_layers(
        result['before_labels'], result['after_labels'], aoi)
    result['assessed'] = assessed
    return result
