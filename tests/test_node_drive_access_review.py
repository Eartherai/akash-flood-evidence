"""Independent checks for the distinct driving-way/node-POI source policy.

Local fakes and synthetic water products only; no network or new inference.
The earlier broad-core tests and failed attempts remain separate artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import acquire_node_drive_historic_osm as source
import map_node_drive_access as adapter
import map_fresh_exposure as original
from akash_flood.access import _DRIVE_HIGHWAYS, build_road_graph
from test_core_access_review import LocalResponse, LocalSession, basic_source
from test_supplemental_exposure import water_only_fixture


def run_source(tmp_path, *, payload=None, client=None, execute=True):
    if client is None:
        payload = payload or json.dumps({"elements": basic_source()["elements"][:3]}).encode()
        client = LocalSession(LocalResponse(payload, headers={"Content-Length": str(len(payload))}))
    result = source.acquire([85.22, 27.59, 85.38, 27.85], "2024-09-01", "2024-09-28",
                            tmp_path / "node-source", execute=execute, session=client)
    return result, client


def saved_receipt(tmp_path):
    return json.loads((tmp_path / "node-source/network-receipt.json").read_text())


def node_map_fixture(tmp_path):
    manifest, plan_path, previous_osm, _ = water_only_fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    old = json.loads(previous_osm.read_text())
    ways = [item for item in old["elements"] if item["type"] == "way" and item.get("tags", {}).get("highway")]
    node_ids = {identity for item in ways for identity in item["nodes"]}
    nodes = [item for item in old["elements"] if item["type"] == "node" and item["id"] in node_ids]
    nodes[0]["tags"]["building"] = "yes"  # Incidental tag is not a building inventory.
    data = {"elements": [*nodes, *ways]}
    folder = tmp_path / "later-node-drive-source"
    source.acquire(plan["case"]["context_bbox"], plan["case"]["osm_snapshot"],
                   plan["case"]["event_date"], folder, execute=True,
                   session=LocalSession(LocalResponse(json.dumps(data).encode())))
    return manifest, plan_path, folder / "osm.json", folder / "provenance.json", data


def rebind(paths, protocol):
    protocol_path = paths[2].parent / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    provenance = json.loads(paths[3].read_text())
    provenance["protocol"] = original.record(protocol_path)
    paths[3].write_text(json.dumps(provenance))
    receipt_path = paths[2].parent / "network-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["protocol"] = original.record(protocol_path)
    receipt["provenance"] = original.record(paths[3])
    receipt_path.write_text(json.dumps(receipt))


def test_offline_node_drive_query_matches_existing_graph_class_policy(tmp_path):
    client = LocalSession(error=AssertionError("No network in offline mode"))
    result, _ = run_source(tmp_path, client=client, execute=False)
    assert result["status"] == "protocol_only" and client.calls == []
    protocol = json.loads((tmp_path / "node-source/protocol.json").read_text())
    assert protocol["schema"] == "akash-node-drive-historic-map-protocol-v1"
    query = protocol["query"]
    pattern = re.search(r'way\["highway"~"\^\(([^)]+)\)\$"\]', query)
    assert pattern and set(pattern.group(1).split("|")) == _DRIVE_HIGHWAYS
    assert 'node["place"~"^(city|town|village|hamlet)$"](27.59,85.22,27.85,85.38)' in query
    assert 'node["amenity"="hospital"](27.59,85.22,27.85,85.38)' in query
    assert '[date:"2024-09-01T00:00:00Z"]' in query
    assert '(._;>;);out body;' in query
    assert all(unrequested not in query for unrequested in ('nwr[', 'relation[', 'building', 'out meta'))
    assert protocol["categories"] == {
        "roads": "graph_drive_highway_classes_requested", "places": "node_tagged_requested",
        "hospitals": "node_tagged_requested", "buildings": "not_requested_unavailable"}
    assert "Polygon/relation POIs and non-driving highways are not requested" in protocol["scope"]
    assert len(protocol["code"]) == 3
    assert protocol["code"][2] == original.record(Path(__file__).parents[1] / "src/akash_flood/access.py")


def test_node_drive_acquisition_binds_exact_transfer_and_new_scope(tmp_path):
    result, client = run_source(tmp_path)
    assert result["schema"] == "akash-node-drive-historic-map-transport-v1"
    assert result["status"] == "completed" and result["attempted_requests"] == 1
    assert len(client.calls) == 1 and client.response.closed
    assert client.calls[0][1]["allow_redirects"] is False and client.calls[0][1]["verify"] is True
    provenance = json.loads((tmp_path / "node-source/provenance.json").read_text())
    assert provenance["schema"] == "akash-node-drive-historic-map-source-v1"
    assert provenance["building_inventory_available"] is False and provenance["map_completeness_validated"] is False
    assert result["delivered_body_sha256"] == original.record(tmp_path / "node-source/response.json")["sha256"]


def test_node_source_missing_way_node_cannot_be_promoted(tmp_path):
    value = {"elements": basic_source()["elements"][:3]}
    value["elements"][2]["nodes"][-1] = 999
    with pytest.raises(ValueError):
        run_source(tmp_path, payload=json.dumps(value).encode())
    receipt = saved_receipt(tmp_path)
    assert receipt["status"] == "failed" and receipt["attempted_requests"] == 1
    assert receipt["retained_partial_response"]["bytes"] > 0
    assert not (tmp_path / "node-source/osm.json").exists()


@pytest.mark.parametrize("payload", [
    b'{"elements":[],"remark":"runtime exceeded"}',
    b'{"elements":[{"type":"node","id":true,"lon":85.3,"lat":27.7}]}',
    b'{"elements":[{"type":"node","id":1,"lon":85.3,"lat":NaN}]}',
    b'{"elements":[{"type":"node","id":1,"lon":85.3,"lat":27.7}],"elements":[]}',
])
def test_invalid_node_source_fails_with_retained_body(tmp_path, payload):
    with pytest.raises(ValueError):
        run_source(tmp_path, payload=payload)
    assert saved_receipt(tmp_path)["status"] == "failed"
    assert not (tmp_path / "node-source/osm.json").exists()


@pytest.mark.parametrize("kind", ["redirect", "compression", "truncated", "server-error"])
def test_failed_node_transport_is_counted_once_and_cannot_promote(tmp_path, kind):
    payload = json.dumps({"elements": basic_source()["elements"][:3]}).encode()
    response = LocalResponse(payload, status=302 if kind == "redirect" else 504 if kind == "server-error" else 200)
    if kind == "compression": response.headers["Content-Encoding"] = "gzip"
    if kind == "truncated": response.headers["Content-Length"] = str(len(payload) + 1)
    client = LocalSession(response)
    with pytest.raises(ValueError): run_source(tmp_path, client=client)
    assert saved_receipt(tmp_path)["attempted_requests"] == 1 and len(client.calls) == 1
    assert response.closed and not (tmp_path / "node-source/osm.json").exists()


def test_node_source_cap_counts_rejected_chunk_but_keeps_bounded_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "MAX_BYTES", 5)
    client = LocalSession(LocalResponse(b"", chunks=[b"abc", b"def"]))
    with pytest.raises(ValueError): run_source(tmp_path, client=client)
    receipt = saved_receipt(tmp_path)
    assert receipt["received_body_bytes"] == 6 and receipt["retained_partial_response"]["bytes"] == 3
    assert (tmp_path / "node-source/received-response.partial.bin").read_bytes() == b"abc"


@pytest.mark.parametrize("kind", ["auth", "header", "cookies"])
def test_node_source_rejects_credentials(tmp_path, kind):
    client = LocalSession(error=AssertionError("No credentialed network"))
    if kind == "auth": client.auth = ("name", "secret")
    if kind == "header": client.headers = {"Cookie": "session=secret"}
    if kind == "cookies": client.cookies = {"session": "secret"}
    with pytest.raises(ValueError): run_source(tmp_path, client=client)
    assert client.calls == []


@pytest.mark.parametrize("field,value", [
    ("maximum_requests", 2), ("maximum_body_bytes", 40_000_000),
    ("maximum_objects", 1_000_000), ("deadline_seconds", 240),
    ("stop_granularity_bytes", 1024), ("tls_verified", False),
    ("redirects", True), ("automatic_retry", True), ("execute_requested", False),
])
def test_node_reader_rejects_changed_protocol_contract_even_with_rebound_records(tmp_path, field, value):
    paths = node_map_fixture(tmp_path)
    protocol = json.loads((paths[2].parent / "protocol.json").read_text())
    protocol[field] = value
    rebind(paths, protocol)
    case = json.loads(paths[1].read_text())["case"]
    with pytest.raises(ValueError): source.read_core(paths[2], paths[3], case)


@pytest.mark.parametrize("kind", ["empty", "missing-graph-policy", "foreign-code"])
def test_node_reader_requires_all_three_actual_code_bindings(tmp_path, kind):
    paths = node_map_fixture(tmp_path)
    protocol = json.loads((paths[2].parent / "protocol.json").read_text())
    if kind == "empty": protocol["code"] = []
    if kind == "missing-graph-policy": protocol["code"] = protocol["code"][:2]
    if kind == "foreign-code": protocol["code"][2] = original.record(Path(__file__))
    rebind(paths, protocol)
    case = json.loads(paths[1].read_text())["case"]
    with pytest.raises(ValueError): source.read_core(paths[2], paths[3], case)


@pytest.mark.parametrize("field,value", [
    ("schema", "akash-core-historic-map-protocol-v1"),
    ("query", '[out:json][date:"2024-09-01T00:00:00Z"];nwr["amenity"="hospital"];out body;'),
    ("categories", {"roads": "requested", "places": "requested", "hospitals": "requested", "buildings": "requested"}),
])
def test_node_reader_does_not_accept_broad_or_full_category_source_identity(tmp_path, field, value):
    paths = node_map_fixture(tmp_path)
    protocol = json.loads((paths[2].parent / "protocol.json").read_text())
    protocol[field] = value
    rebind(paths, protocol)
    case = json.loads(paths[1].read_text())["case"]
    with pytest.raises(ValueError): source.read_core(paths[2], paths[3], case)


def test_node_mapping_keeps_buildings_null_and_original_graph_inputs_immutable(tmp_path):
    paths = node_map_fixture(tmp_path)
    snapshot = {p: original.record(p) for p in tmp_path.rglob("*") if p.is_file()}
    source_graph = build_road_graph(paths[4])
    expected_edges = [(u, v, key, value["edge_id"]) for u, v, key, value in source_graph.edges(keys=True, data=True)]
    result = adapter.run(*paths[:4], tmp_path / "node-access")
    assert result["schema"] == "akash-node-drive-access-v1"
    assert result["buildings"] is None and result["category_availability"]["buildings"] is False
    assert result["roads"]["mapped_feature_count"] == 3 and result["bridges"]["mapped_feature_count"] == 1
    assert result["access_scenario"]["confirmed_closures"] == 0
    assert result["access_scenario"]["certificate_checks"] == {"paths": 1, "cuts": 1}
    assert result["model_inference_run"] is False and result["network_requests"] == 0
    assert result["prior_water_walltime_includes_these_stages"] is False
    assert not (tmp_path / "node-access/buildings.geojson").exists()
    assert {p: original.record(p) for p in snapshot} == snapshot
    assert [(u, v, key, value["edge_id"]) for u, v, key, value in build_road_graph(paths[4]).edges(keys=True, data=True)] == expected_edges
    proof = json.loads((tmp_path / "node-access/access-proof.json").read_text())
    assert proof["certain_closed_edge_ids"] == []
    assert proof["settlements"][0]["nearest_baseline_destination"]["destination_id"] == "node/2"
    assert proof["settlements"][0]["optimistic_path_witness"]["destination_id"] == "node/2"
    assert proof["settlements"][0]["pessimistic_cut_certificate"]["mapped_destination_node_ids"] == [2]


def test_node_mapping_rejects_original_plan_relocation_and_water_tampering(tmp_path):
    paths = node_map_fixture(tmp_path)
    relocated = tmp_path / "relocated.json"
    relocated.write_bytes(paths[1].read_bytes())
    with pytest.raises(ValueError): adapter.run(paths[0], relocated, *paths[2:4], tmp_path / "relocated-output")
    assert not (tmp_path / "relocated-output").exists()
    with (tmp_path / "water/after-water.tif").open("ab") as f: f.write(b"tampered")
    with pytest.raises(ValueError): adapter.run(*paths[:4], tmp_path / "tampered-output")
    assert not (tmp_path / "tampered-output").exists()
