"""Registration diagnostics checked against direct spatial correlation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine


SCRIPT = Path(__file__).parents[1] / "scripts/verify_radar_registration.py"
SPEC = importlib.util.spec_from_file_location("radar_registration", SCRIPT)
registration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(registration)


def direct_ncc(before, after, valid, dy, dx):
    height, width = before.shape
    r0, r1 = max(0, -dy), min(height, height - dy)
    c0, c1 = max(0, -dx), min(width, width - dx)
    first = (slice(r0, r1), slice(c0, c1))
    second = (slice(r0 + dy, r1 + dy), slice(c0 + dx, c1 + dx))
    paired = valid[first] & valid[second]
    a, b = before[first][paired], after[second][paired]
    a, b = a - a.mean(), b - b.mean()
    return paired.sum(), np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))


def test_all_fft_offset_scores_match_independent_direct_masked_ncc():
    rng = np.random.default_rng(123)
    before, after = rng.normal(size=(2, 32, 31))
    valid = rng.random((32, 31)) > 0.1
    scores = registration.fft_correlation_scores(before, after, valid)
    assert len(scores) == 49
    for row in scores:
        count, expected = direct_ncc(before, after, valid, row["dy_px"], row["dx_px"])
        assert row["pair_pixels"] == count
        assert row["ncc"] == pytest.approx(expected, abs=2e-14)


@pytest.mark.parametrize("dy,dx", [(0, 0), (2, -1), (-2, 2), (3, -3)])
def test_shift_direction_and_search_boundary_are_reported_without_moving_inputs(dy, dx):
    rng = np.random.default_rng(456)
    db = rng.normal(-12, 2, (256, 256))
    before = 10 ** (db / 10)
    after = np.roll(before, shift=(dy, dx), axis=(0, 1)) * 2  # Constant gain does not affect NCC.
    before_copy, after_copy = before.copy(), after.copy()
    result = registration.diagnose_tile(before, after, np.ones((256, 256), dtype=bool))
    assert result["estimated_after_displacement_px"] == {"dy": dy, "dx": dx}
    assert result["best_ncc"] == pytest.approx(1, abs=1e-12)
    assert result["estimated_after_displacement_m"] == {"south": dy * 10, "east": dx * 10}
    assert any("search boundary" in flag for flag in result["flags"]) == (abs(dy) == 3 or abs(dx) == 3)
    assert np.array_equal(before, before_copy) and np.array_equal(after, after_copy)


def test_constant_signal_is_uninformative_not_zero_registration_success():
    before = np.ones((256, 256))
    result = registration.diagnose_tile(before, before * 2, np.ones((256, 256), dtype=bool))
    assert result["status"] == "uninformative_texture_or_pair_coverage"
    assert result["estimated_after_displacement_px"] is None
    assert all(row["ncc"] is None for row in result["all_offset_scores"])


def test_tile_selection_is_bounded_deterministic_and_counts_every_full_tile():
    valid = np.ones((256 * 5 + 17, 256 * 4 + 3), dtype=bool)
    valid[0:256, 0:256] = False
    result = registration.fixed_tile_selection(valid)
    assert result["full_tile_count"] == 20
    assert result["eligible_tile_count"] == 19
    assert result["selected_tile_count"] == 12
    assert len(result["all_candidate_tiles"]) == 20
    assert result == registration.fixed_tile_selection(valid.copy())
    selected_positions = [(tile["row"], tile["col"]) for tile in result["selected"]]
    assert len(set(selected_positions)) == 12
    assert selected_positions[0] == (0, 256) and selected_positions[-1] == (1024, 768)
    assert result["excluded_edge_pixels"] == valid.size - 20 * 256 * 256


def test_changed_or_uninformative_selected_tile_is_not_dropped_after_scoring():
    rng = np.random.default_rng(789)
    before = 10 ** (rng.normal(-12, 2, (256, 512)) / 10)
    after = before.copy()
    before[:, 256:] = 1
    after[:, 256:] = 2
    result = registration.run_diagnostics(before, after, np.ones((256, 512), dtype=bool))
    assert result["selection"]["selected_tile_count"] == 2
    assert len(result["tiles"]) == 2
    assert result["summary"]["informative_tile_count"] == 1
    assert result["summary"]["uninformative_tile_count"] == 1
    assert result["tiles"][1]["estimated_after_displacement_px"] is None
    assert "pass/fail" in result["summary"]["verification_claim"]


def test_region_smaller_than_fixed_tile_is_explicitly_unscored():
    valid = np.ones((256, 253), dtype=bool)
    result = registration.run_diagnostics(np.ones(valid.shape), np.ones(valid.shape), valid)
    assert result["selection"]["full_tile_count"] == 0
    assert result["tiles"] == []
    assert result["summary"]["median_integer_dx_px"] is None


def write_raster(path, array, transform=Affine(10, 0, 335345, 0, -10, 3119165), nodata=None):
    with rasterio.open(path, "w", driver="GTiff", width=array.shape[1], height=array.shape[0], count=1,
                       dtype=array.dtype, crs="EPSG:32645", transform=transform, nodata=nodata) as dst:
        dst.write(array, 1)


def test_exact_grid_mismatch_is_rejected_before_correlation(tmp_path):
    before, after, valid = [tmp_path / name for name in ("before.tif", "after.tif", "valid.tif")]
    power = np.ones((256, 256), dtype="float32")
    write_raster(before, power)
    write_raster(after, power, Affine(10, 0, 335355, 0, -10, 3119165))
    write_raster(valid, np.ones((256, 256), dtype="uint8"))
    with pytest.raises(ValueError, match="share the exact grid"):
        registration.read_inputs(before, after, valid)


@pytest.mark.parametrize("bad_value", [0, -1, np.nan, np.inf])
def test_masked_valid_power_must_be_finite_and_positive(tmp_path, bad_value):
    before, after, valid = [tmp_path / name for name in ("before.tif", "after.tif", "valid.tif")]
    power = np.ones((256, 256), dtype="float32")
    power[0, 0] = bad_value
    write_raster(before, power)
    write_raster(after, np.ones((256, 256), dtype="float32"))
    write_raster(valid, np.ones((256, 256), dtype="uint8"))
    with pytest.raises(ValueError, match="finite positive linear power"):
        registration.read_inputs(before, after, valid)


def test_nodata_exclusion_must_be_explicit_in_common_mask(tmp_path):
    before, after, valid = [tmp_path / name for name in ("before.tif", "after.tif", "valid.tif")]
    power = np.ones((256, 256), dtype="float32")
    power[0, 0] = -9999
    write_raster(before, power, nodata=-9999)
    write_raster(after, np.ones((256, 256), dtype="float32"))
    mask = np.ones((256, 256), dtype="uint8")
    write_raster(valid, mask)
    with pytest.raises(ValueError, match="over before nodata"):
        registration.read_inputs(before, after, valid)
    mask[0, 0] = 0
    write_raster(valid, mask)
    _, _, returned_mask, metadata = registration.read_inputs(before, after, valid)
    assert not returned_mask[0, 0]
    assert metadata["grid_validation"] == "exact identical affine/CRS/shape"
