"""CPU supervised water classification using explicit sigma0 dB features.

The Sen1Floods11 handlabels combine flood and permanent water. This module does
not identify debris, confirmed flood damage or physical infrastructure access.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

FEATURE_NAMES = (
    "VV_db", "VH_db", "VV_minus_VH_db", "VV_mean3_db", "VH_mean3_db",
    "VV_mean7_db", "VH_mean7_db", "VV_std3_db", "VH_std3_db", "joint_valid_fraction7",
)
CLIP_DB = (-50.0, 1.0)


def chip_grids_match(source_grid, label_grid, *, tolerance_pixels=1e-6):
    """Allow numeric affine serialization noise, never a fitted alignment."""
    source_crs, source_transform, source_shape = source_grid
    label_crs, label_transform, label_shape = label_grid
    if source_crs != label_crs or source_shape != label_shape:
        return False
    height, width = source_shape
    for x, y in ((.5, .5), (width - .5, .5), (.5, height - .5), (width - .5, height - .5)):
        px, py = ~source_transform * (label_transform * (x, y))
        if not np.isfinite([px, py]).all() or max(abs(px - x), abs(py - y)) > tolerance_pixels:
            return False
    return True


def features_from_db(vv_vh_db, valid_mask=None):
    """Keep valid numeric zeros; exclude masked/nonfinite neighbor samples."""
    source = np.asarray(vv_vh_db)
    if source.ndim != 3 or source.shape[0] != 2 or min(source.shape[1:]) < 1:
        raise ValueError("Expected VV/VH sigma0 dB array with shape (2,height,width)")
    valid = np.isfinite(source).all(axis=0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.shape != valid.shape or mask.dtype != bool:
            raise ValueError("External observation/geometry mask must be matching boolean array")
        valid &= mask
    clipped = np.clip(source.astype("float64"), *CLIP_DB)
    samples = np.where(valid[None, :, :], clipped, 0)
    counts, means = {}, {}
    for window in (3, 7):
        count = uniform_filter(valid.astype("float64"), size=window, mode="constant", cval=0) * window ** 2
        counts[window] = np.maximum(count, 0)
        for pol in (0, 1):
            total = uniform_filter(samples[pol], size=window, mode="constant", cval=0) * window ** 2
            means[window, pol] = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    stds = []
    for pol in (0, 1):
        second = uniform_filter(samples[pol] ** 2, size=3, mode="constant", cval=0) * 9
        second = np.divide(second, counts[3], out=np.zeros_like(second), where=counts[3] > 0)
        stds.append(np.sqrt(np.maximum(second - means[3, pol] ** 2, 0)))
    cube = np.stack([clipped[0], clipped[1], clipped[0] - clipped[1],
                     means[3, 0], means[3, 1], means[7, 0], means[7, 1],
                     stds[0], stds[1], counts[7] / 49], axis=-1).astype("float32")
    cube[~valid] = np.nan
    return cube, valid


def predict_water(model, vv_vh_db, valid_mask=None, *, threshold=.5, batch_pixels=65536):
    """Return classifier scores and 0/1/255 labels; scores are not calibrated."""
    if threshold != .5 or batch_pixels < 1:
        raise ValueError("Frozen baseline uses threshold .5 and positive batch size")
    features, valid = features_from_db(vv_vh_db, valid_mask)
    scores = np.full(valid.shape, np.nan, dtype="float32")
    labels = np.full(valid.shape, 255, dtype="uint8")
    indices = np.flatnonzero(valid)
    flat = features.reshape(-1, len(FEATURE_NAMES))
    for start in range(0, len(indices), batch_pixels):
        block = indices[start:start + batch_pixels]
        probability = np.asarray(model.predict_proba(flat[block]))
        if probability.shape != (len(block), 2) or not np.array_equal(model.classes_, [0, 1]):
            raise ValueError("Require binary nonwater/water model with classes [0,1]")
        p = probability[:, 1]
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Model emitted invalid classifier scores")
        scores.ravel()[block] = p
        labels.ravel()[block] = (p >= .5).astype("uint8")
    return scores, labels


def confusion_counts(handlabels, predicted):
    labels, predicted = np.asarray(handlabels), np.asarray(predicted)
    if labels.shape != predicted.shape or not np.isin(labels, [-1, 0, 1]).all() or not np.isin(predicted, [0, 1, 255]).all():
        raise ValueError("Expected matching -1/0/1 handlabels and 0/1/255 predictions")
    known = labels != -1
    assessed = known & (predicted != 255)
    return {"tp": int(((labels == 1) & (predicted == 1) & assessed).sum()),
            "fp": int(((labels == 0) & (predicted == 1) & assessed).sum()),
            "fn": int(((labels == 1) & (predicted == 0) & assessed).sum()),
            "tn": int(((labels == 0) & (predicted == 0) & assessed).sum()),
            "label_valid_pixels": int(known.sum()), "assessed_label_pixels": int(assessed.sum()),
            "unsupported_label_pixels": int((known & ~assessed).sum()),
            "unsupported_water_pixels": int(((labels == 1) & ~assessed).sum()),
            "ignored_label_pixels": int((~known).sum()), "total_pixels": int(labels.size)}


def metrics(counts):
    def ratio(a, b):
        return a / b if b else None
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    return {"water_iou": ratio(tp, tp + fp + fn), "water_precision": ratio(tp, tp + fp),
            "water_recall": ratio(tp, tp + fn), "water_f1": ratio(2 * tp, 2 * tp + fp + fn),
            "assessed_pixel_accuracy": ratio(tp + tn, tp + fp + fn + tn),
            "label_valid_assessment_coverage": ratio(counts["assessed_label_pixels"], counts["label_valid_pixels"]),
            "full_label_water_recall_unknown_not_detected": ratio(tp, tp + fn + counts["unsupported_water_pixels"])}
