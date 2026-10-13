import numpy as np
import pytest

from akash_flood.radar_evidence import BANDS
from akash_flood.water_inference import positive_db, inference_support, temporal_layers, infer_pair


def scene(shape=(51, 51)):
    s = {name: np.full(shape, 35, dtype='float32') for name in BANDS}
    s['Sigma0_VV'].fill(.1); s['Sigma0_VH'].fill(.01)
    s['elevation'].fill(100); s['layoverShadowMask'].fill(0)
    return s


class FixedWaterModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        p = np.where(features[:, 0] < -15, .9, .1)
        return np.column_stack([1 - p, p])


def test_raw_power_log_keeps_zero_db_and_excludes_nonpositive():
    s = scene((2, 2))
    s['Sigma0_VV'][:] = [[1, 0], [-.01, np.inf]]
    db, valid = positive_db(s)
    assert valid.tolist() == [[True, False], [False, False]]
    assert db[:, 0, 0].tolist() == [0, -20]
    assert np.isnan(db[:, ~valid]).all()


def test_unknown_geometry_and_footprint_edges_are_not_nonwater():
    b, a = scene(), scene()
    a['layoverShadowMask'][25, 25] = 1
    _, _, support, valid = inference_support(b, a, np.ones((51, 51), dtype=bool))
    assert not support[25, 25] and not valid[25, 25]
    assert not valid[0, 25] and valid[25, 20]
    result = infer_pair(FixedWaterModel(), b, a, np.ones((51, 51), dtype=bool))
    assert result['before_labels'][25, 25] == 255
    assert result['new'][25, 25] == 255
    assert result['counts']['new_water_candidate_pixels'] == 0


def test_temporal_accounting_and_no_change_from_unknown():
    b = np.array([[0, 1, 1, 0, 255]], dtype='uint8')
    a = np.array([[1, 0, 1, 0, 255]], dtype='uint8')
    new, lost, counts = temporal_layers(b, a, np.ones(b.shape, dtype=bool))
    assert new.tolist() == [[1, 0, 0, 0, 255]]
    assert lost.tolist() == [[0, 1, 0, 0, 255]]
    assert counts == dict(aoi_pixels=5, assessed_pixels=4, unknown_pixels=1,
                         before_water_pixels=2, after_water_pixels=2,
                         new_water_candidate_pixels=1, lost_water_candidate_pixels=1,
                         unchanged_water_pixels=1)


def test_temporal_domains_must_match_and_outside_aoi_is_unknown():
    with pytest.raises(ValueError, match='identical joint'):
        temporal_layers(np.array([[255]]), np.array([[0]]), np.array([[True]]))
    with pytest.raises(ValueError, match='outside the AOI'):
        temporal_layers(np.array([[0]]), np.array([[0]]), np.array([[False]]))


def test_new_water_candidates_use_raw_date_features_not_previous_screening():
    b, a = scene(), scene()
    a['Sigma0_VV'].fill(.001)
    aoi = np.ones((51, 51), dtype=bool)
    result = infer_pair(FixedWaterModel(), b, a, aoi)
    assert result['counts']['assessed_pixels'] > 0
    assert result['counts']['new_water_candidate_pixels'] == result['counts']['assessed_pixels']
    assert result['counts']['before_water_pixels'] == 0
    assert result['counts']['lost_water_candidate_pixels'] == 0
