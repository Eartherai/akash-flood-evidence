"""Independent display integrity, scope and hostile source-label tests.

Synthetic local data only; these do not run models or validate physical access.
"""
import copy
from html import escape
import json
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import map_node_drive_access as adapter
import map_fresh_exposure as mapper
import render_node_drive_access as renderer
from test_node_drive_access_review import node_map_fixture


@pytest.fixture
def prepared_report(tmp_path):
    paths = node_map_fixture(tmp_path)
    water = json.loads(paths[0].read_text())
    water["gpu_used"] = False
    water["grid"]["epsg"] = 32645
    paths[0].write_text(json.dumps(water))
    out = tmp_path / "backend"
    adapter.run(*paths[:4], out)
    return paths[0], out / "source-and-output-manifest.json"


def rebound_output(binding_path, name, mutate):
    binding = json.loads(binding_path.read_text())
    path = Path(binding["outputs"][name]["path"])
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))
    binding["outputs"][name] = mapper.record(path)
    binding_path.write_text(json.dumps(binding))


def test_successful_display_preserves_native_clip_and_source_witness_scope(prepared_report, tmp_path):
    result = renderer.run(*prepared_report, tmp_path / "display")
    assert result["status"] == "completed"
    facts = json.loads((tmp_path / "display/report.json").read_text())
    assert facts["buildings"] is None and facts["confirmed_closures"] == 0
    assert facts["local_accuracy_established"] is False
    assert facts["all_other_baseline_edges_assumed_available"] is True
    html = (tmp_path / "display/index.html").read_text()
    assert '<clipPath id="aoi">' in html
    assert html.count('clip-path="url(#aoi)"') == 2
    assert 'full baseline road network is omitted' in html
    assert 'shortest hop count' in html and 'does not establish safe passage' in html
    binding = json.loads((tmp_path / "display/source-and-output-manifest.json").read_text())
    assert binding["rendering"]["aoi_cell_clip"] is True
    assert binding["rendering"]["full_baseline_network_drawn"] is False
    assert binding["rendering"]["selected_witness_geometry_derived_from_bound_osm"] is True
    assert binding["source_edge_certificates_rechecked"] == {"paths": 1, "cuts": 1}
    assert all(mapper.record(item["path"]) == item for item in binding["outputs"].values())


@pytest.mark.parametrize("name", ["summary.json", "access-proof.json", "roads.geojson"])
def test_display_rejects_changed_bound_backend_bytes_before_output(prepared_report, tmp_path, name):
    binding = json.loads(prepared_report[1].read_text())
    with Path(binding["outputs"][name]["path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError): renderer.run(*prepared_report, tmp_path / "display")
    assert not (tmp_path / "display").exists()


@pytest.mark.parametrize("kind", ["buildings-summary", "category", "confirmed-closure", "observed-hazard", "status-count"])
def test_display_rejects_rebound_semantic_category_or_scenario_substitution(prepared_report, tmp_path, kind):
    if kind == "buildings-summary": rebound_output(prepared_report[1], "summary.json", lambda v:v.update(buildings={"mapped_feature_count":0}))
    elif kind == "category": rebound_output(prepared_report[1], "summary.json", lambda v:v["category_availability"].update(buildings=True))
    elif kind == "confirmed-closure": rebound_output(prepared_report[1], "summary.json", lambda v:v["access_scenario"].update(confirmed_closures=1))
    elif kind == "observed-hazard": rebound_output(prepared_report[1], "access-proof.json", lambda v:v.update(observed_hazard_or_closures_used=True))
    else: rebound_output(prepared_report[1], "summary.json", lambda v:v["access_scenario"].update(status_counts={"robust_connected":1}))
    with pytest.raises(ValueError): renderer.run(*prepared_report, tmp_path / "display")
    assert not (tmp_path / "display").exists()


def test_display_rejects_rebound_path_using_a_nonsource_edge(prepared_report, tmp_path):
    def change(proof): proof["settlements"][0]["optimistic_path_witness"]["path_edge_ids"][0] = "unknown-source-edge"
    rebound_output(prepared_report[1], "access-proof.json", change)
    with pytest.raises((ValueError, KeyError, AssertionError)): renderer.run(*prepared_report, tmp_path / "display")
    assert not (tmp_path / "display").exists()


def test_display_rejects_coordinated_nonexistent_destination_identity(prepared_report, tmp_path):
    def change(proof):
        row = proof["settlements"][0]
        row["nearest_baseline_destination"]["destination_id"] = "node/999999999"
        row["optimistic_path_witness"]["destination_id"] = "node/999999999"
    rebound_output(prepared_report[1], "access-proof.json", change)
    with pytest.raises(ValueError): renderer.run(*prepared_report, tmp_path / "display")
    assert not (tmp_path / "display").exists()


def test_display_rejects_invented_physical_group_identity(prepared_report, tmp_path):
    def change(proof):
        original = next(iter(proof["uncertain_links"]))
        proof["uncertain_links"]["way-99999999-nodes-1-2"] = proof["uncertain_links"].pop(original)
    rebound_output(prepared_report[1], "access-proof.json", change)
    with pytest.raises(ValueError): renderer.run(*prepared_report, tmp_path / "display")
    assert not (tmp_path / "display").exists()


@pytest.mark.parametrize("field", ["baseline_road_distance_m", "baseline_distance_including_association_m", "destination_snap_distance_m"])
def test_display_rejects_invented_fixed_baseline_distance(prepared_report, tmp_path, field):
    def change(proof): proof["settlements"][0]["nearest_baseline_destination"][field] += 100
    rebound_output(prepared_report[1], "access-proof.json", change)
    with pytest.raises(ValueError): renderer.run(*prepared_report, tmp_path / "display")
    assert not (tmp_path / "display").exists()


def test_display_rejects_changed_source_settlement_association(prepared_report, tmp_path):
    def change(proof): proof["settlements"][0]["snap_distance_m"] += 100
    rebound_output(prepared_report[1], "access-proof.json", change)
    with pytest.raises(ValueError): renderer.run(*prepared_report, tmp_path / "display")
    assert not (tmp_path / "display").exists()


def test_display_uses_bound_historical_snapshot_and_after_date_offset(prepared_report, tmp_path):
    renderer.run(*prepared_report, tmp_path / "date-display")
    html = (tmp_path / "date-display/index.html").read_text()
    # Synthetic source: July 2026 historical state and 22 August observation
    # after the 20 August event. Kathmandu-specific date text must not leak.
    assert 'Historical OSM 1 September 2024' not in html
    assert '2026-07-01' in html
    assert 'by four days' not in html


def test_scene_rejects_a_required_display_edge_missing_from_bound_graph(prepared_report, tmp_path):
    _, arrays, grid, _, proof, _, graph, settlements, _ = renderer.read_supplement(*prepared_report)
    proof = copy.deepcopy(proof)
    proof["uncertain_links"][next(iter(proof["uncertain_links"]))][0] = "nonsource-edge"
    out = tmp_path / "scene";out.mkdir()
    with pytest.raises(ValueError, match="lacks display geometry"):
        renderer.scene(out, arrays, grid, proof, graph, settlements)


def test_osm_name_cannot_close_json_script_or_create_html(prepared_report, tmp_path, monkeypatch):
    values = list(renderer.read_supplement(*prepared_report))
    malicious = '</script><img src=x onerror="window.pwned=1"><script>window.pwned=2</script>&'
    values[4] = copy.deepcopy(values[4])
    values[7] = copy.deepcopy(values[7])
    values[8] = copy.deepcopy(values[8])
    values[4]["settlements"][0]["name"] = malicious
    values[7][0]["name"] = malicious
    values[8][0]["name"] = malicious
    monkeypatch.setattr(renderer, "read_supplement", lambda *args:tuple(values))
    renderer.run(*prepared_report, tmp_path / "hostile-name-display")
    html = (tmp_path / "hostile-name-display/index.html").read_text()
    assert malicious not in html and '<img src=x' not in html
    match = re.search(r'<script id="evidence" type="application/json">(.*?)</script>', html, re.S)
    assert match and '<' not in match.group(1) and '>' not in match.group(1) and '&' not in match.group(1)
    payload = json.loads(match.group(1))
    assert payload["routes"][0]["name"] == malicious
    assert escape(malicious) in html
    assert 'p.textContent=text' in html


def test_outside_context_source_point_counts_are_distinct_from_aoi_count(prepared_report, tmp_path, monkeypatch):
    values = list(renderer.read_supplement(*prepared_report))
    values[4] = copy.deepcopy(values[4]);values[7] = copy.deepcopy(values[7])
    case = values[0]["case"]
    point = {"id":"node/999","name":"Outside dependency","lon":case["context_bbox"][2]+.001,
             "lat":case["context_bbox"][1]+.001,"kind":"village"}
    values[7].append(point)
    row = copy.deepcopy(values[4]["settlements"][0]);row.update(id=point["id"],name=point["name"])
    row["path_witness"] = row["optimistic_path_witness"] = None
    values[4]["settlements"].append(row)
    monkeypatch.setattr(renderer,"read_supplement",lambda *args:tuple(values))
    result = renderer.run(*prepared_report,tmp_path/"scope-display")
    scope = result["point_scope"]
    assert scope["all_source_settlements"] == 2
    assert scope["settlements_in_context"] == 1
    assert scope["settlements_in_aoi"] == 0
    assert scope["incidental_settlements_outside_context"] == [point]
