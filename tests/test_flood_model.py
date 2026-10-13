import numpy as np
import pytest

from akash_flood.flood_model import features_from_db, predict_water, confusion_counts, metrics, FEATURE_NAMES, chip_grids_match
from affine import Affine


def test_affine_serialization_noise_passes_without_resampling():
    source = Affine(8.983152841195215e-05, 0., -1.1498435636729876, 0., -8.983152841195215e-05, 9.474710964665418)
    label = Affine(8.983152841195184e-05, 0., -1.1498435636729876, 0., -8.983152841195184e-05, 9.474710964665418)
    assert chip_grids_match(("EPSG:4326", source, (512, 512)), ("EPSG:4326", label, (512, 512)))


def test_half_cell_shift_or_mismatched_grid_never_passes_serialization_tolerance():
    source = Affine(.001, 0., 20., 0., -.001, 10.)
    grid = ("EPSG:4326", source, (512, 512))
    assert not chip_grids_match(grid, ("EPSG:4326", source * Affine.translation(.5, 0), (512, 512)))
    assert not chip_grids_match(grid, ("EPSG:4326", source, (511, 512)))
    assert not chip_grids_match(grid, ("EPSG:3857", source, (512, 512)))


def test_numeric_zero_db_stays_valid_and_constant_neighborhood_uses_only_real_samples():
    db = np.stack([np.zeros((5, 5)), np.full((5, 5), -20.)])
    features, valid = features_from_db(db)
    assert valid.all() and len(FEATURE_NAMES) == 10
    assert np.allclose(features[..., 0], 0)
    assert np.allclose(features[..., 3], 0) and np.allclose(features[..., 4], -20)
    assert np.allclose(features[..., 7:9], 0, atol=1e-6)
    assert features[0, 0, 9] == pytest.approx(16 / 49)


def test_nonfinite_or_geometry_masked_samples_never_enter_neighborhood_as_valid_zero():
    db = np.full((2, 5, 5), -20.)
    db[:, 2, 2] = np.nan
    mask = np.ones((5, 5), dtype=bool); mask[1, 1] = False
    features, valid = features_from_db(db, mask)
    assert not valid[2, 2] and not valid[1, 1]
    assert np.isnan(features[2, 2]).all()
    assert features[2, 3, 3] == pytest.approx(-20.)
    assert features[2, 3, 7] == pytest.approx(0., abs=1e-6)


def test_context_features_match_known_three_by_three_values():
    values = np.arange(25).reshape(5, 5) - 30.
    features, valid = features_from_db(np.stack([values, values - 5]))
    neighborhood = values[1:4, 1:4]
    assert features[2, 2, 3] == pytest.approx(neighborhood.mean())
    assert features[2, 2, 7] == pytest.approx(neighborhood.std())
    assert features[2, 2, 2] == pytest.approx(5.)


class ThresholdModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, rows):
        water = np.where(rows[:, 0] < -17, .8, .2)
        return np.column_stack([1 - water, water])


def test_prediction_keeps_unknown_cells_and_exact_threshold_in_bounded_batches():
    db = np.array([[[-20., -10.], [-20., -20.]], [[-25., -15.], [-25., -25.]]])
    mask = np.array([[True, True], [False, True]])
    scores, labels = predict_water(ThresholdModel(), db, mask, batch_pixels=1)
    assert np.array_equal(labels, [[1, 0], [255, 1]])
    assert np.isnan(scores[1, 0]) and scores[0, 0] == pytest.approx(.8)
    with pytest.raises(ValueError, match="Frozen"):
        predict_water(ThresholdModel(), db, threshold=.6)


def test_metrics_exclude_minus_one_labels_but_report_unsupported_water():
    labels = np.array([[1, 1, 0, 0, -1, 1]])
    predictions = np.array([[1, 0, 1, 0, 1, 255]])
    c = confusion_counts(labels, predictions)
    assert [c[k] for k in ["tp", "fp", "fn", "tn"]] == [1, 1, 1, 1]
    assert c["ignored_label_pixels"] == 1 and c["unsupported_water_pixels"] == 1
    m = metrics(c)
    assert m["water_iou"] == pytest.approx(1 / 3)
    assert m["water_recall"] == .5
    assert m["full_label_water_recall_unknown_not_detected"] == pytest.approx(1 / 3)
    assert m["label_valid_assessment_coverage"] == .8


def test_empty_water_denominator_is_unknown_instead_of_perfect_accuracy():
    c = confusion_counts(np.array([0, -1]), np.array([0, 255]))
    m = metrics(c)
    assert m["water_iou"] is None and m["water_precision"] is None and m["water_recall"] is None


def test_invalid_class_or_mask_shapes_refused():
    with pytest.raises(ValueError):
        confusion_counts(np.array([2]), np.array([1]))
    with pytest.raises(ValueError):
        features_from_db(np.zeros((2, 3, 3)), np.zeros((2, 3), dtype=bool))
    with pytest.raises(ValueError):
        features_from_db(np.zeros((3, 3)))


def test_nonfinite_model_scores_are_not_silent_water_predictions():
    class BadModel(ThresholdModel):
        def predict_proba(self, rows):
            return np.full((len(rows), 2), np.nan)
    with pytest.raises(ValueError, match="invalid classifier scores"):
        predict_water(BadModel(), np.full((2, 2, 2), -20.))
