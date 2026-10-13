"""Guard scientific grid alignment and non-overwriting processing behavior."""
import importlib.util
from pathlib import Path
import sys

from affine import Affine
import numpy as np
import pytest
from rasterio.coords import BoundingBox

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import process_s1_snap as pipeline
import check_snap_context as quality


def test_fractional_pixel_shift_is_rejected():
    source = Affine(10, 0, 331055, 0, -10, 3136085)
    exact = source * Affine.translation(2, 3)
    window = quality.exact_window(source, (100, 100), exact, (20, 30))
    assert (window.col_off, window.row_off, window.width, window.height) == (2, 3, 30, 20)
    with pytest.raises(ValueError, match="fractional pixel"):
        quality.exact_window(source, (100, 100), source * Affine.translation(.5, 1), (20, 30))
    with pytest.raises(ValueError, match="exceeds source"):
        quality.exact_window(source, (100, 100), exact, (100, 100))


def test_mask_zero_does_not_make_missing_or_back_facing_cells_valid():
    b = {name: np.ones((1, 4), dtype=float) for name in pipeline.BANDS}
    b["layoverShadowMask"][:] = 0
    b["localIncidenceAngle"][0, 1] = 0
    b["projectedLocalIncidenceAngle"][0, 2] = 95
    b["Sigma0_VH"][0, 3] = -.1
    covered, valid = quality.validity(b)
    assert valid.tolist() == [[True, False, False, False]]
    b["layoverShadowMask"][0, 0] = 2
    assert not quality.validity(b)[1].any()


def test_north_up_common_footprint_selects_exact_integer_windows():
    grids = [
        {"crs": "EPSG:32645", "shape": (10, 10), "transform": Affine(10, 0, 100, 0, -10, 200), "bounds": BoundingBox(100, 100, 200, 200)},
        {"crs": "EPSG:32645", "shape": (12, 12), "transform": Affine(10, 0, 90, 0, -10, 210), "bounds": BoundingBox(90, 90, 210, 210)},
    ]
    transform, shape, windows = quality.common_grid(grids)
    assert shape == (10, 10)
    assert transform == grids[0]["transform"]
    assert windows[1].col_off == windows[1].row_off == 1


def test_processing_refuses_existing_directory_before_reading_sources(tmp_path):
    with pytest.raises(FileExistsError):
        pipeline.run([85.28, 28.1, 85.45, 28.34], ["before"], tmp_path, tmp_path / "absent.json")


def test_graph_pins_units_and_does_not_claim_precise_orbits_or_flattening(tmp_path):
    root = pipeline.graph("before", tmp_path / "manifest.safe", tmp_path / "product.dim", [85.28, 28.1, 85.45, 28.34]).getroot()
    operators = [node.findtext("operator") for node in root.findall("node")]
    assert operators == ["Read", "Subset", "ThermalNoiseRemoval", "Calibration", "Terrain-Correction", "Write"]
    assert root.findtext("./node[@id='noise']/parameters/clipNegativeValues") == "false"
    assert root.findtext("./node[@id='terrain']/parameters/applyRadiometricNormalization") == "false"
    assert root.findtext("./node[@id='terrain']/parameters/mapProjection").startswith("PROJCS[")
    assert root.findtext("./node[@id='terrain']/parameters/demName") == "SRTM 1Sec HGT"
    with pytest.raises(ValueError):
        pipeline.validate_bbox([85.5, 28.1, 85.2, 28.4])
