import numpy as np
import pytest

from akash_flood.flood import FloodConfig, classify_change


def test_water_change_excludes_preexisting_water_and_invalid_terrain():
    pre = np.full((5, 5), -10.0)
    post = pre.copy()
    post[1:4, 1:4] = -21
    pre[2, 2] = -22  # already water/dark: not newly flooded
    slope = np.zeros_like(pre)
    slope[1, 1] = 20
    invalid = np.zeros_like(pre, dtype=bool)
    invalid[3, 3] = True
    labels, counts = classify_change(pre, post, slope_deg=slope, radar_invalid=invalid,
                                    config=FloodConfig(min_component_pixels=1))
    assert labels[2, 2] == 2
    assert labels[1, 1] == labels[3, 3] == 255
    assert counts["new_water_candidate_pixels"] == 6
    assert counts["unassessed_pixels"] == 2


def test_small_components_and_nan_do_not_turn_into_flood():
    pre = np.full((4, 4), -10.)
    post = pre.copy()
    post[0, 0] = -25
    post[3, 3] = np.nan
    labels, _ = classify_change(pre, post)
    assert labels[0, 0] == 0
    assert labels[3, 3] == 255


def test_shapes_and_thresholds_are_checked():
    with pytest.raises(ValueError):
        classify_change(np.zeros((2, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        FloodConfig(change_threshold_db=3)
