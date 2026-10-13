import importlib.util
import json
from pathlib import Path

import pytest
import rasterio

from akash_flood.pipeline import run, validate_contract


def fixture(tmp_path):
    path = Path(__file__).parents[1] / "scripts" / "make_fixture.py"
    spec = importlib.util.spec_from_file_location("fixture_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create(tmp_path / "inputs")


def test_raster_to_road_access_and_report(tmp_path):
    config = fixture(tmp_path)
    out = tmp_path / "out"
    metrics = run(config, out)
    assert metrics["new_water_candidate_pixels"] == 240
    assert metrics["candidate_area_km2"] == pytest.approx(.024)
    assert metrics["newly_disconnected_under_scenario"] == 1
    assert metrics["mapped_building_features"] == 2
    assert metrics["building_features_exposed"] == 1
    buildings = json.loads((out / "exposed_buildings.geojson").read_text())
    assert buildings["features"][0]["properties"]["overlap_m2"] == pytest.approx(400)
    assert buildings["features"][0]["properties"]["confirmed_damage"] is False
    access = json.loads((out / "settlement_access.json").read_text())
    assert [row["status"] for row in access] == ["newly_disconnected", "connected"]
    assert "SYNTHETIC ENGINEERING TEST" in (out / "report.html").read_text()
    assert "SYNTHETIC ENGINEERING TEST" in (out / "map.html").read_text()
    with rasterio.open(out / "hazard_candidates.tif") as ds:
        assert ds.crs.to_epsg() == 32645 and ds.nodata == 255
    with pytest.raises(ValueError, match="empty"):
        run(config, out)


def test_raw_amplitude_cannot_be_passed_as_backscatter(tmp_path):
    config = json.loads(fixture(tmp_path).read_text())
    config["before"]["units"] = "amplitude"
    with pytest.raises(ValueError, match="raw amplitude"):
        validate_contract(config)
