"""Synthetic source/graph integration checks; no imagery, model or network calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from akash_flood.observation_access import ObservationGraph

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fresh_observation_adapter", ROOT / "scripts/run_fresh_observation_access_20261012.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def feature(way, u, v, *, unknown=0, outside=0, candidate=False):
    gid = f"way-{way}-nodes-{min(u, v)}-{max(u, v)}"
    def coverage(measure, scale=1):
        return {"measure": measure, "assessed": (100 - unknown - outside) * scale,
                "unassessed": unknown * scale, "outside_aoi": outside * scale,
                "total": 100 * scale, "new_water_candidate": 1 if candidate else 0}
    return {"type": "Feature", "id": gid,
            "geometry": {"type": "LineString", "coordinates": [[u * .001, .001], [v * .001, .001]]},
            "properties": {"physical_segment_id": gid, "source_way_id": way, "source_endpoint_node_ids": [u, v],
                           "source_directed_edges": [{"edge_id": f"{way}:f", "from_node_id": u, "to_node_id": v},
                                                     {"edge_id": f"{way}:r", "from_node_id": v, "to_node_id": u}],
                           "source_directed_edge_count": 2, "centerline_coverage": coverage("length_m"),
                           "coverage": coverage("area_m2", 20), "new_water_candidate_overlap": candidate}}


def point(identifier, node, name=None, **extra):
    coords = [node * .001, .001]
    return {"type": "Feature", "id": identifier, "geometry": {"type": "Point", "coordinates": coords},
            "properties": {"id": identifier, "name": name or identifier, "lon": coords[0], "lat": coords[1], **extra}}


def fixture(tmp_path, *, unknown=0, outside=0, same_node=False, candidate=False, farther_target=False):
    folder = tmp_path / "arbitrary-case-not-a-known-folder"
    folder.mkdir()
    roads = [feature(10, 1, 2, unknown=unknown, outside=outside, candidate=candidate), feature(20, 2, 3)]
    if farther_target:
        roads.append(feature(30, 1, 4))
    graph = ObservationGraph()
    for f in roads:
        graph.add(f)
    source = 3 if same_node else 1
    target = {"destination_id": "target", "destination_snap_node_id": 3}
    candidate_groups = {gid: list(data[0]) for gid, data in graph.groups.items() if data[3]}
    removed = {eid for ids in candidate_groups.values() for eid in ids}
    connected = source in graph.reverse_reach(3, removed)
    witness = graph.path(source, 3, graph.reverse_reach(3, removed if connected else set()))
    witness.update(destination_id="target", scenario="pessimistic" if connected else "optimistic")
    rows = [{"id": "village", "name": "Village", "snap_node_id": source,
             "status": "robust_connected" if connected else "ambiguous", "nearest_baseline_destination": target,
             "path_witness": witness if connected else None, "optimistic_path_witness": None if connected else witness},
            {"id": "no-map", "name": "No map", "snap_node_id": None, "status": "unmapped",
             "nearest_baseline_destination": None, "path_witness": None, "optimistic_path_witness": None},
            {"id": "no-baseline", "name": "No baseline", "snap_node_id": 1, "status": "baseline_disconnected",
             "nearest_baseline_destination": None, "path_witness": None, "optimistic_path_witness": None}]
    case = {"case_id": "explicitly-synthetic-case", "bbox": [0, 0, .004, .004], "context_bbox": [-.01, -.01, .01, .01],
            "event_date": "2020-01-02", "osm_snapshot": "2020-01-01T00:00:00Z"}
    counts = {"synthetic": 1}
    water = folder / "water-manifest.json"
    dump(water, {"schema": "akash-fresh-water-v1", "available": True, "reference_labels_loaded": False,
                 "case": case, "counts": counts, "source_dates": {"before": "2019-12-31", "after": "2020-01-03"}})
    plan = folder / "plan.json"
    dump(plan, {"case": case})
    water_record, plan_record = adapter.record(water), adapter.record(plan)
    proof = {"schema": "synthetic-source-proof", "case": case, "water_source": water_record, "source_plan": plan_record,
             "temporal_counts": counts, "settlements": rows, "certain_closed_edge_ids": [],
             "destination_switching_after_closure": False, "observed_hazard_or_closures_used": False,
             "max_snap_m": 500, "uncertain_links": candidate_groups,
             "status_counts": {"robust_connected" if connected else "ambiguous": 1, "unmapped": 1, "baseline_disconnected": 1}}
    summary = {"schema": "akash-fresh-exposure-v1", "status": "completed", "case": case,
               "water_source": water_record, "plan_source": plan_record, "temporal_counts": counts,
               "source_dates": {"before": "2019-12-31", "after": "2020-01-03"}, "gpu_used": False,
               "reference_labels_loaded": False, "roads": {"mapped_feature_count": len(roads)},
               "source_feature_audits": {"roads": {"source_directed_edges": len(roads) * 2}},
               "access_scenario": {"status_counts": proof["status_counts"], "n_settlements": 3, "n_destinations": 2 if farther_target else 1,
                                   "scenario_is_hypothetical": True, "confirmed_closures": 0,
                                   "destination_switching_after_closure": False}}
    scenario = {"scenario_is_hypothetical": True, "certain_closed_edge_ids": [], "confirmed_closures": 0,
                "positive_overlap_area_tolerance_m2": 1e-6}
    values = {"roads.geojson": {"type": "FeatureCollection", "features": roads}, "summary.json": summary,
              "access-proof.json": proof, "scenario-policy.json": scenario,
              "settlements.geojson": {"type": "FeatureCollection", "features": [point("village", source, "Village"),
                                       point("no-map", 4, "No map"), point("no-baseline", 5, "No baseline")]},
              "destinations.geojson": {"type": "FeatureCollection", "features": [point("target", 3)] +
                                       ([point("farther-still-reachable", 4)] if farther_target else [])}}
    for name, value in values.items():
        dump(folder / name, value)
    extra_input = folder / "bound-opaque-source.bin"
    extra_input.write_bytes(b"synthetic source bytes, never imagery/model")
    binding_path = folder / "source-and-output-manifest.json"
    dump(binding_path, {"schema": "akash-fresh-exposure-binding-v1", "status": "completed", "case": case,
                       "reference_labels_loaded": False, "model_inference_run": False, "network_requests": 0,
                       "water_source": water_record, "plan_source": plan_record, "summary": adapter.record(folder / "summary.json"),
                       "inputs": [water_record, plan_record, adapter.record(extra_input)], "code": [],
                       "outputs": {name: adapter.record(folder / name) for name in values}})
    return binding_path


def freeze_run(tmp_path, **options):
    source = fixture(tmp_path, **options)
    protocol = tmp_path / "frozen.json"
    adapter.freeze(source, protocol)
    output = tmp_path / "new-analysis"
    receipt = adapter.run(protocol, output)
    return source, protocol, output, receipt


@pytest.mark.parametrize("options,reason", [({"unknown": 20}, "unassessed_within_aoi"), ({"outside": 20}, "outside_aoi")])
def test_unknown_or_outside_on_only_route_changes_robust_to_ambiguous(tmp_path, options, reason):
    source, protocol, output, receipt = freeze_run(tmp_path, **options)
    result = adapter.read_json(output / "expanded-scenario.json")
    assert result["settlements"][0]["previous_status"] == "robust_connected"
    assert result["settlements"][0]["status"] == "ambiguous"
    assert result["uncertain_physical_groups"]["way-10-nodes-1-2"] == [reason]
    assert receipt["independent_extremes_passed"] is True
    assert receipt["expanded_status_counts"] == {"ambiguous": 1, "unmapped": 1, "baseline_disconnected": 1}
    assert receipt["scope_status_counts"]["inside_water_aoi"] == {"ambiguous": 1, "unmapped": 1}
    assert receipt["claims"]["network"] is False
    assert receipt["full_track_b_requirements_satisfied"] is False
    assert adapter.read_json(source)["case"] == result["case"]


def test_empty_witness_remains_null_and_no_target_rows_preserved(tmp_path):
    _, _, output, receipt = freeze_run(tmp_path, same_node=True)
    rows = adapter.read_json(output / "expanded-scenario.json")["settlements"]
    assert rows[0]["status"] == "robust_connected"
    assert rows[0]["path_witness"]["path_edge_ids"] == []
    assert rows[0]["path_support"]["fractions_of_projected_centerline"] == {
        "assessed": None, "unknown_within_aoi": None, "outside_aoi": None}
    assert receipt["zero_length_graph_witness_ids"] == ["village"]
    assert [r["status"] for r in rows[1:]] == ["unmapped", "baseline_disconnected"]
    assert all(r["nearest_baseline_destination"] is None for r in rows[1:])
    assert adapter.read_json(output / "independent-extremes.json")["no_target_rows_preserved"] == 2


def test_candidate_uncertainty_keeps_both_directions_and_fixed_destination(tmp_path):
    _, _, output, _ = freeze_run(tmp_path, candidate=True)
    row = adapter.read_json(output / "expanded-scenario.json")["settlements"][0]
    assert row["status"] == row["previous_status"] == "ambiguous"
    assert row["nearest_baseline_destination"] == {"destination_id": "target", "destination_snap_node_id": 3}
    assert row["path_witness"]["destination_node_id"] == 3
    result = adapter.read_json(output / "expanded-scenario.json")
    assert result["uncertain_directed_edge_count"] == 2
    assert result["certain_closed_edge_ids"] == []


def test_available_farther_destination_does_not_replace_original_fixed_target(tmp_path):
    _, _, output, receipt = freeze_run(tmp_path, unknown=20, farther_target=True)
    row = adapter.read_json(output / "expanded-scenario.json")["settlements"][0]
    assert row["status"] == "ambiguous"
    assert row["nearest_baseline_destination"]["destination_id"] == "target"
    assert row["path_witness"]["destination_node_id"] == 3
    assert receipt["independent_extremes_passed"] is True


@pytest.mark.parametrize("name", ["roads.geojson", "bound-opaque-source.bin", "summary.json", "source-and-output-manifest.json"])
def test_source_mutation_rejected_and_failed_attempt_preserved(tmp_path, name):
    source = fixture(tmp_path)
    protocol = tmp_path / "protocol.json"
    adapter.freeze(source, protocol)
    path = source.parent / name
    path.write_bytes(path.read_bytes() + b" ")
    output = tmp_path / "analysis"
    with pytest.raises(ValueError, match="checksum|size"):
        adapter.run(protocol, output)
    receipt = adapter.read_json(output / "receipt.json")
    assert receipt["status"] == "failed" and receipt["no_automatic_retry"] is True
    assert (output / "failure.json").is_file()
    assert not (output / "expanded-scenario.json").exists()
    with pytest.raises(FileExistsError):
        adapter.run(protocol, output)


def test_tampered_policy_or_missing_records_cannot_be_used(tmp_path):
    source = fixture(tmp_path)
    protocol = tmp_path / "protocol.json"
    adapter.freeze(source, protocol)
    value = adapter.read_json(protocol)
    value["policy"]["unknown_is_observed_closed"] = True
    dump(protocol, value)
    with pytest.raises(ValueError, match="declared observation protocol"):
        adapter.run(protocol, tmp_path / "bad-policy")
    value["policy"] = deepcopy(adapter.POLICY)
    value["records"] = value["records"][:-1]
    dump(protocol, value)
    with pytest.raises(ValueError, match="bound record set"):
        adapter.run(protocol, tmp_path / "missing-binding")


def test_independent_forward_check_detects_solver_result_disagreement(tmp_path, monkeypatch):
    source = fixture(tmp_path, unknown=20)
    protocol = tmp_path / "protocol.json"
    adapter.freeze(source, protocol)
    original = ObservationGraph.bounds
    def corrupt(self, old, removed):
        result = original(self, old, removed)
        result["settlements"][0]["status"] = "robust_connected"
        return result
    monkeypatch.setattr(ObservationGraph, "bounds", corrupt)
    with pytest.raises(ValueError, match="Independent forward DFS"):
        adapter.run(protocol, tmp_path / "solver-disagreement")


def test_reference_source_and_cross_case_metadata_rejected(tmp_path):
    source = fixture(tmp_path)
    value = adapter.read_json(source)
    forbidden = tmp_path / "reference-labels"
    forbidden.mkdir()
    bad = forbidden / "labels.bin"
    bad.write_bytes(b"never load this input")
    value["inputs"].append({"path": str(bad), "bytes": 21, "sha256": "0" * 64})
    dump(source, value)
    protocol = tmp_path / "forbidden.json"
    with pytest.raises(ValueError, match="reference|Reference"):
        adapter.freeze(source, protocol)
    assert protocol.with_name(protocol.name + ".failure.json").is_file()
    value["inputs"].pop()
    value["case"]["event_date"] = "2020-01-03"
    dump(source, value)
    with pytest.raises(ValueError, match="case"):
        adapter.freeze(source, tmp_path / "cross-case.json")


def test_bound_runtime_ceiling_preserves_failure(tmp_path, monkeypatch):
    source = fixture(tmp_path)
    protocol = tmp_path / "protocol.json"
    adapter.freeze(source, protocol)
    monkeypatch.setattr(adapter, "rss_bytes", lambda: 2_000_000_001)
    with pytest.raises(MemoryError, match="2 GB"):
        adapter.run(protocol, tmp_path / "over-memory")
    assert adapter.read_json(tmp_path / "over-memory/receipt.json")["status"] == "failed"


def test_offline_freeze_and_analysis_never_start_network_or_subprocess(tmp_path, monkeypatch):
    import socket
    import subprocess
    def forbidden(*args, **kwargs):
        raise AssertionError("analysis tried network or subprocess")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    _, _, _, receipt = freeze_run(tmp_path)
    assert receipt["status"] == "completed"
    assert all(v is False for v in receipt["claims"].values())
