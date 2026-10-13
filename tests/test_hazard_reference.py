"""Hand-counted evaluator checks; fixtures are synthetic, never event labels."""
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine
from shapely.geometry import box
from shapely import to_wkb


SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_hazard_reference.py"
SPEC = importlib.util.spec_from_file_location("hazard_reference_evaluator", SCRIPT)
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)
GATES = {"mapped_domain_valid_fraction": 0.9, "mapped_event_valid_fraction": 0.95}


def toy_grid():
    # Six mapped cells; three mapped-event cells. One extra array cell is outside
    # the AOI, ensuring its event prediction cannot influence the denominator.
    domain = np.array([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=bool)
    event = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=bool)
    return {"shape": (2, 4), "transform": Affine(10, 0, 5, 0, -10, 25),
            "crs": rasterio.crs.CRS.from_epsg(32645), "aoi": domain,
            "domain": domain, "event": event, "vector_mapped_event_area_m2": 300}


def write_raster(path, data, transform=Affine(10, 0, 5, 0, -10, 25), nodata=None, crs="EPSG:32645"):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype=data.dtype, crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(data, 1)


def test_hand_counted_agreement_and_constant_baselines():
    grid = toy_grid()
    prediction = np.array([[1, 0, 1, 1], [1, 0, 0, 1]], dtype=bool)
    result = ev.calculate_metrics(prediction, np.ones((2, 4), dtype=bool), grid, "landslide_extent", GATES)
    agreement = result["mapped_extent_agreement_on_valid_pixels"]
    assert agreement["pixel_contingency"] == {"intersection": 2, "candidate_only": 1, "reference_only": 1, "neither": 2}
    assert agreement["iou"] == 0.5
    assert agreement["dice"] == pytest.approx(2 / 3)
    assert agreement["precision_relative_to_mapped_extent"] == pytest.approx(2 / 3)
    assert agreement["recall_on_observed_mapped_extent"] == pytest.approx(2 / 3)
    assert agreement["agreement_fraction"] == pytest.approx(4 / 6)
    assert result["constant_baselines_same_valid_domain"]["all_event"]["iou"] == 0.5
    assert result["constant_baselines_same_valid_domain"]["all_background"]["iou"] == 0
    assert result["coverage"]["passes_coverage_gates"]


def test_unobservable_reference_is_reported_not_silently_removed():
    grid = toy_grid()
    valid = np.ones((2, 4), dtype=bool)
    valid[0, 1] = False
    result = ev.calculate_metrics(grid["event"], valid, grid, "mass_movement_extent", GATES)
    assert result["mapped_extent_agreement_on_valid_pixels"]["recall_on_observed_mapped_extent"] == 1
    assert result["spatial_overlap_diagnostics"]["mapped_event_intersected_fraction_all_domain"] == pytest.approx(2 / 3)
    assert result["coverage"]["unobservable_event_pixels"] == 1
    assert result["coverage"]["mapped_event_valid_fraction"] == pytest.approx(2 / 3)
    assert result["score_status"] == "partial_coverage_diagnostics_only"


def test_radar_change_gets_overlap_diagnostics_and_no_classifier_scores():
    grid = toy_grid()
    result = ev.calculate_metrics(grid["event"], np.ones((2, 4), dtype=bool), grid, "radar_change_candidate", GATES)
    assert result["score_status"] == "exploratory_radar_change_overlap_only"
    assert result["spatial_overlap_diagnostics"]["mapped_event_intersected_fraction_all_domain"] == 1
    assert "mapped_extent_agreement_on_valid_pixels" not in result
    assert "constant_baselines_same_valid_domain" not in result


@pytest.mark.parametrize("semantics", ["open_water_flood", "debris_extent", "unknown"])
def test_incompatible_class_cannot_receive_landslide_score(semantics):
    grid = toy_grid()
    with pytest.raises(ValueError, match="landslide, not open-water flood"):
        ev.calculate_metrics(grid["event"], grid["domain"], grid, semantics, GATES)


def test_no_coverage_has_null_ratios_not_perfect_empty_scores():
    grid = toy_grid()
    result = ev.calculate_metrics(grid["event"], np.zeros((2, 4), dtype=bool), grid, "landslide_extent", GATES)
    assert result["mapped_extent_agreement_on_valid_pixels"]["iou"] is None
    assert result["mapped_extent_agreement_on_valid_pixels"]["agreement_fraction"] is None
    assert result["spatial_overlap_diagnostics"]["mapped_event_intersected_fraction_all_domain"] == 0
    assert not result["coverage"]["passes_coverage_gates"]


def test_partial_source_coverage_preserves_fixed_domain(tmp_path):
    prediction, mask = tmp_path / "prediction.tif", tmp_path / "mask.tif"
    # One row at the lower edge, two columns beginning one column to the right.
    affine = Affine(10, 0, 15, 0, -10, 15)
    write_raster(prediction, np.array([[1, 0]], dtype="uint8"), affine)
    write_raster(mask, np.array([[1, 1]], dtype="uint8"), affine)
    placed, valid, _ = ev.place_on_grid(prediction, mask, toy_grid())
    assert placed.tolist() == [[False, False, False, False], [False, True, False, False]]
    assert valid.tolist() == [[False, False, False, False], [False, True, True, False]]


@pytest.mark.parametrize("affine,crs", [
    (Affine(10, 0, 0, 0, -10, 20), "EPSG:32645"),
    (Affine(20, 0, 0, 0, -20, 20), "EPSG:32645"),
    (Affine(10, 0, 0, 0, -10, 20), "EPSG:4326"),
])
def test_grid_shift_scale_and_crs_are_rejected(tmp_path, affine, crs):
    prediction, mask = tmp_path / "prediction.tif", tmp_path / "mask.tif"
    for path in (prediction, mask):
        write_raster(path, np.ones((2, 4), dtype="uint8"), affine, crs=crs)
    with pytest.raises(ValueError):
        ev.place_on_grid(prediction, mask, toy_grid())


def test_mask_cannot_assert_validity_over_prediction_nodata(tmp_path):
    prediction, mask = tmp_path / "prediction.tif", tmp_path / "mask.tif"
    write_raster(prediction, np.array([[1, 255, 0, 0], [1, 0, 0, 0]], dtype="uint8"), nodata=255)
    write_raster(mask, np.ones((2, 4), dtype="uint8"))
    with pytest.raises(ValueError, match="prediction nodata"):
        ev.place_on_grid(prediction, mask, toy_grid())


def test_invalid_mask_pixels_do_not_need_binary_prediction_values(tmp_path):
    prediction, mask = tmp_path / "prediction.tif", tmp_path / "mask.tif"
    write_raster(prediction, np.array([[1, 255, 0, 0], [1, 0, 0, 0]], dtype="uint8"), nodata=255)
    write_raster(mask, np.array([[1, 0, 1, 1], [1, 1, 1, 1]], dtype="uint8"))
    _, valid, _ = ev.place_on_grid(prediction, mask, toy_grid())
    assert not valid[0, 1]


def test_mask_and_prediction_must_share_identical_grid(tmp_path):
    prediction, mask = tmp_path / "prediction.tif", tmp_path / "mask.tif"
    write_raster(prediction, np.ones((2, 4), dtype="uint8"))
    write_raster(mask, np.ones((2, 3), dtype="uint8"))
    with pytest.raises(ValueError, match="grids differ"):
        ev.place_on_grid(prediction, mask, toy_grid())


@pytest.mark.parametrize("byte_order,flag", [("<", 1), (">", 0)])
@pytest.mark.parametrize("envelope_code,values", [(0, 0), (1, 4), (2, 6), (3, 6), (4, 8)])
def test_gpkg_header_respects_endianness_and_envelope_size(byte_order, flag, envelope_code, values):
    geometry = box(0, 0, 1, 1)
    blob = b"GP\x00" + bytes([flag | (envelope_code << 1)]) + struct.pack(byte_order + "i", 4326)
    blob += struct.pack(byte_order + "d" * values, *([0.0] * values)) + to_wkb(geometry)
    decoded, srs = ev.decode_gpkg_geometry(blob)
    assert srs == 4326 and decoded.equals(geometry)


def test_truncated_or_extended_gpkg_geometry_is_rejected():
    with pytest.raises(ValueError):
        ev.decode_gpkg_geometry(b"GP\x00\x03" + struct.pack("<i", 4326))
    with pytest.raises(ValueError):
        ev.decode_gpkg_geometry(b"GP\x00\x21" + struct.pack("<i", 4326) + to_wkb(box(0, 0, 1, 1)))


def test_candidate_protocol_can_audit_but_cannot_score(tmp_path):
    protocol = tmp_path / "candidate.json"
    protocol.write_text(json.dumps({"schema_version": "emsr927-hazard-reference-v1", "status": "candidate", "coverage_gates": GATES}))
    ev.load_protocol(protocol, scoring=False)
    with pytest.raises(ValueError, match="Candidate protocol cannot score"):
        ev.load_protocol(protocol, scoring=True)


def test_frozen_protocol_rejects_modified_evaluator(tmp_path):
    protocol = tmp_path / "frozen.json"
    protocol.write_text(json.dumps({"schema_version": "emsr927-hazard-reference-v1", "status": "frozen", "frozen_at": "2026-10-06T00:00:00Z",
                                   "evaluator_sha256": "0" * 64, "prediction_recipe_sha256": "1" * 64, "coverage_gates": GATES}))
    with pytest.raises(ValueError, match="Evaluator differs"):
        ev.load_protocol(protocol, scoring=True)


def test_prediction_manifest_pins_recipe_and_rejects_reference_input(tmp_path):
    prediction, mask, recipe = [tmp_path / name for name in ("prediction.tif", "mask.tif", "recipe.json")]
    for path in (prediction, mask, recipe):
        path.write_bytes(path.name.encode())
    manifest = {"prediction_sha256": ev.sha256(prediction), "valid_mask_sha256": ev.sha256(mask),
                "prediction_recipe_path": "recipe.json", "prediction_semantics": "radar_change_candidate",
                "evaluation_reference_used_for_prediction": False,
                "valid_mask_derivation": "sensor_geometry_and_data_availability_only"}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    protocol = {"prediction_recipe_sha256": ev.sha256(recipe), "prediction_sha256": ev.sha256(prediction), "valid_mask_sha256": ev.sha256(mask)}
    assert ev.validate_prediction_manifest(manifest_path, prediction, mask, protocol)["prediction_semantics"] == "radar_change_candidate"
    manifest["evaluation_reference_used_for_prediction"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="evaluation reference was not used"):
        ev.validate_prediction_manifest(manifest_path, prediction, mask, protocol)


def test_reference_checksum_is_checked_before_opening_sqlite(tmp_path):
    gpkg = tmp_path / "fake.gpkg"
    gpkg.write_bytes(b"not sqlite")
    with pytest.raises(ValueError, match="checksum differs"):
        ev.load_reference(gpkg, "0" * 64)
