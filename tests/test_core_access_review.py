"""Independent core-only historical source and conditional-route controls.

All HTTP responses are local fakes. These tests establish software contracts,
not historical OSM completeness, actual road closures or physical safety.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import acquire_core_historic_osm as core
import map_core_access as core_mapper
import map_fresh_exposure as frozen_mapper
from test_supplemental_exposure import water_only_fixture


class LocalResponse:
    def __init__(self, payload, *, status=200, headers=None, chunks=None):
        self.payload = payload
        self.status_code = status
        self.headers = headers or {}
        self.chunks = chunks
        self.closed = False

    def iter_content(self, size):
        assert size == 65_536
        yield from self.chunks if self.chunks is not None else [self.payload]

    def close(self):
        self.closed = True


class LocalSession:
    auth = None
    headers = {}
    cookies = {}

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


def basic_source():
    return {"generator": "local synthetic fixture", "elements": [
        {"type": "node", "id": 1, "lon": 85.30, "lat": 27.70,
         "tags": {"place": "village", "name": "Village"}},
        {"type": "node", "id": 2, "lon": 85.31, "lat": 27.70,
         "tags": {"place": "town", "name": "Town"}},
        {"type": "way", "id": 1, "nodes": [1, 2],
         "tags": {"highway": "secondary", "bridge": "yes"}},
        {"type": "relation", "id": 1, "members": [
            {"type": "way", "ref": 1, "role": ""}]},
    ]}


def acquire_fixture(tmp_path, source=None, *, execute=True, client=None):
    if client is None:
        payload = json.dumps(source if source is not None else basic_source()).encode()
        client = LocalSession(LocalResponse(payload, headers={"Content-Length": str(len(payload))}))
    result = core.acquire([85.22, 27.59, 85.38, 27.85], "2024-09-01",
                          "2024-09-28", tmp_path / "core-source",
                          execute=execute, session=client)
    return result, client


def transport_receipt(tmp_path):
    return json.loads((tmp_path / "core-source/network-receipt.json").read_text())


def test_core_default_is_offline_and_never_requests_building_category(tmp_path):
    client = LocalSession(error=AssertionError("Network forbidden"))
    result, _ = acquire_fixture(tmp_path, execute=False, client=client)
    assert result["status"] == "protocol_only"
    assert client.calls == []
    protocol = json.loads((tmp_path / "core-source/protocol.json").read_text())
    query = protocol["query"]
    assert '[date:"2024-09-01T00:00:00Z"]' in query
    assert 'way["highway"](27.59,85.22,27.85,85.38)' in query
    assert 'nwr["place"~"^(city|town|village|hamlet)$"]' in query
    assert 'nwr["amenity"="hospital"]' in query
    assert '>>' in query and 'out body' in query
    assert 'building' not in query and 'out meta' not in query
    assert protocol["maximum_requests"] == 1


def test_type_qualified_namespaces_and_complete_nested_members_are_accepted(tmp_path):
    source = basic_source()
    source["elements"].append({"type": "relation", "id": 2,
                               "members": [{"type": "relation", "ref": 1, "role": ""}]})
    core.validate_dependencies(source)
    result, client = acquire_fixture(tmp_path, source)
    assert result["status"] == "completed"
    assert len(client.calls) == 1
    assert json.loads((tmp_path / "core-source/osm.json").read_text()) == source
    assert client.response.closed
    assert client.calls[0][1]["verify"] is True
    assert client.calls[0][1]["allow_redirects"] is False
    assert client.trust_env is False


@pytest.mark.parametrize("bad_element", [
    {"type": "way", "id": 50, "nodes": [1, 999], "tags": {"highway": "secondary"}},
    {"type": "relation", "id": 50, "members": [{"type": "node", "ref": 999, "role": ""}]},
    {"type": "relation", "id": 50, "members": [{"type": "way", "ref": 999, "role": "outer"}]},
    {"type": "relation", "id": 50, "members": [{"type": "relation", "ref": 999, "role": ""}]},
])
def test_missing_member_dependency_is_retained_but_cannot_be_promoted(tmp_path, bad_element):
    source = basic_source()
    source["elements"].append(bad_element)
    with pytest.raises(ValueError):
        core.validate_dependencies(source)
    with pytest.raises(ValueError):
        acquire_fixture(tmp_path, source)
    receipt = transport_receipt(tmp_path)
    assert receipt["status"] == "failed" and receipt["attempted_requests"] == 1
    assert receipt["retained_partial_response"]["bytes"] > 0
    assert not (tmp_path / "core-source/osm.json").exists()


@pytest.mark.parametrize("mutate", [
    lambda source: source["elements"][0].update(id=True),
    lambda source: source["elements"][0].update(lat=float("nan")),
    lambda source: source["elements"][0].update(lon=181),
    lambda source: source["elements"][0].update(tags={"name": 5}),
    lambda source: source["elements"][2].update(nodes=[1, False]),
    lambda source: source["elements"][3].update(members=[{"type": "area", "ref": 1, "role": ""}]),
    lambda source: source["elements"][0].update(timestamp="2024-09-01T00:00:01Z"),
    lambda source: source["elements"].append(copy.deepcopy(source["elements"][0])),
    lambda source: source.update(remark="runtime exceeded"),
])
def test_invalid_core_source_is_not_promoted(tmp_path, mutate):
    source = basic_source()
    mutate(source)
    with pytest.raises(ValueError):
        acquire_fixture(tmp_path, source)
    assert transport_receipt(tmp_path)["status"] == "failed"
    assert not (tmp_path / "core-source/osm.json").exists()


@pytest.mark.parametrize("mode", ["redirect", "compression", "truncated", "server-error"])
def test_transport_failure_is_a_single_recorded_attempt(tmp_path, mode):
    payload = json.dumps(basic_source()).encode()
    response = LocalResponse(payload, status=302 if mode == "redirect" else 504 if mode == "server-error" else 200)
    if mode == "compression":
        response.headers["Content-Encoding"] = "gzip"
    if mode == "truncated":
        response.headers["Content-Length"] = str(len(payload) + 1)
    client = LocalSession(response)
    with pytest.raises(ValueError):
        acquire_fixture(tmp_path, client=client)
    receipt = transport_receipt(tmp_path)
    assert receipt["status"] == "failed" and receipt["attempted_requests"] == 1
    assert len(client.calls) == 1 and response.closed
    assert not (tmp_path / "core-source/osm.json").exists()


def test_stream_cap_counts_rejected_chunk_without_promoting_written_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "MAX_BYTES", 5)
    response = LocalResponse(b"", chunks=[b"abc", b"def"])
    with pytest.raises(ValueError):
        acquire_fixture(tmp_path, client=LocalSession(response))
    receipt = transport_receipt(tmp_path)
    assert receipt["received_body_bytes"] == 6
    assert receipt["retained_partial_response"]["bytes"] == 3
    assert (tmp_path / "core-source/received-response.partial.bin").read_bytes() == b"abc"
    assert response.closed and not (tmp_path / "core-source/osm.json").exists()


@pytest.mark.parametrize("mode", ["auth", "authorization", "cookie-header", "cookies"])
def test_public_core_transport_rejects_credentials_before_request(tmp_path, mode):
    client = LocalSession(error=AssertionError("No credentialed request"))
    if mode == "auth":
        client.auth = ("name", "secret")
    elif mode == "authorization":
        client.headers = {"aUtHoRiZaTiOn": "Bearer secret"}
    elif mode == "cookie-header":
        client.headers = {"Cookie": "session=secret"}
    else:
        client.cookies = {"session": "secret"}
    with pytest.raises(ValueError):
        acquire_fixture(tmp_path, client=client)
    assert client.calls == []


def test_existing_attempt_cannot_be_replaced(tmp_path):
    acquire_fixture(tmp_path, execute=False)
    protocol = (tmp_path / "core-source/protocol.json").read_bytes()
    with pytest.raises(FileExistsError):
        acquire_fixture(tmp_path, execute=False)
    assert (tmp_path / "core-source/protocol.json").read_bytes() == protocol


def prepared_core_map(tmp_path):
    manifest, plan_path, old_osm, old_provenance = water_only_fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    source = json.loads(old_osm.read_text())
    # A hospital way may incidentally have a building tag in a core response.
    # It must not turn an unrequested buildings category into an inventory.
    next(item for item in source["elements"]
         if item["type"] == "way" and item.get("tags", {}).get("building"))["tags"]["amenity"] = "hospital"
    payload = json.dumps(source).encode()
    acquisition = tmp_path / "later-core-source"
    core.acquire(plan["case"]["context_bbox"], plan["case"]["osm_snapshot"],
                 plan["case"]["event_date"], acquisition, execute=True,
                 session=LocalSession(LocalResponse(payload)))
    return manifest, plan_path, acquisition / "osm.json", acquisition / "provenance.json"


def rebind_protocol(paths, protocol):
    source_dir = paths[2].parent
    protocol_path = source_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    provenance = json.loads(paths[3].read_text())
    provenance["protocol"] = frozen_mapper.record(protocol_path)
    paths[3].write_text(json.dumps(provenance))
    receipt_path = source_dir / "network-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["protocol"] = frozen_mapper.record(protocol_path)
    receipt["provenance"] = frozen_mapper.record(paths[3])
    receipt_path.write_text(json.dumps(receipt))


def test_core_only_mapping_preserves_inputs_and_keeps_buildings_unavailable(tmp_path):
    paths = prepared_core_map(tmp_path)
    before = {path: frozen_mapper.record(path) for path in tmp_path.rglob("*") if path.is_file()}
    out = tmp_path / "core-access"
    summary = core_mapper.run(*paths, out)
    assert summary["schema"] == "akash-core-access-v1"
    assert summary["buildings"] is None
    assert summary["category_availability"]["buildings"] is False
    assert summary["roads"]["mapped_feature_count"] == 3
    assert summary["bridges"]["mapped_feature_count"] == 1
    assert summary["access_scenario"]["confirmed_closures"] == 0
    assert summary["access_scenario"]["certificate_checks"] == {"paths": 1, "cuts": 1}
    assert summary["access_scenario"]["destination_switching_after_closure"] is False
    assert summary["prior_water_walltime_includes_these_stages"] is False
    assert summary["model_inference_run"] is False and summary["network_requests"] == 0
    assert not (out / "buildings.geojson").exists()
    assert {path: frozen_mapper.record(path) for path in before} == before
    proof = json.loads((out / "access-proof.json").read_text())
    row = proof["settlements"][0]
    assert row["status"] == "ambiguous"
    assert row["nearest_baseline_destination"]["destination_id"] == "node/2"
    assert row["optimistic_path_witness"]["destination_id"] == "node/2"
    assert row["pessimistic_cut_certificate"]["mapped_destination_node_ids"] == [2]
    assert proof["certain_closed_edge_ids"] == []


def test_core_map_rejects_relocated_original_plan_and_changed_water(tmp_path):
    paths = prepared_core_map(tmp_path)
    relocated = tmp_path / "relocated-plan.json"
    relocated.write_bytes(paths[1].read_bytes())
    with pytest.raises(ValueError):
        core_mapper.run(paths[0], relocated, *paths[2:], tmp_path / "relocated-output")
    assert not (tmp_path / "relocated-output").exists()
    with (tmp_path / "water/new-water-candidates.tif").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError):
        core_mapper.run(*paths, tmp_path / "water-changed-output")
    assert not (tmp_path / "water-changed-output").exists()


@pytest.mark.parametrize("name", ["osm.json", "response.json", "protocol.json", "network-receipt.json"])
def test_core_reader_rejects_tampering_in_each_source_binding(tmp_path, name):
    paths = prepared_core_map(tmp_path)
    altered = paths[2].parent / name
    if name == "network-receipt.json":
        # This receipt is the local source record, not its own cryptographic
        # authenticator. A semantic failure must be rejected; harmless JSON
        # whitespace does not falsely imply historical authenticity.
        value = json.loads(altered.read_text())
        value["status"] = "failed"
        altered.write_text(json.dumps(value))
    else:
        with altered.open("ab") as stream:
            stream.write(b" ")
    case = json.loads(paths[1].read_text())["case"]
    with pytest.raises(ValueError):
        core.read_core(paths[2], paths[3], case)


@pytest.mark.parametrize("field,value", [
    ("maximum_requests", 2),
    ("maximum_body_bytes", 40_000_000),
    ("maximum_objects", 1_000_000),
    ("deadline_seconds", 240),
    ("stop_granularity_bytes", 1024),
    ("tls_verified", False),
    ("redirects", True),
    ("automatic_retry", True),
    ("execute_requested", False),
])
def test_core_reader_rejects_different_transport_contract_even_with_rebound_hashes(tmp_path, field, value):
    paths = prepared_core_map(tmp_path)
    source_dir = paths[2].parent
    protocol_path = source_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol[field] = value
    rebind_protocol(paths, protocol)
    case = json.loads(paths[1].read_text())["case"]
    with pytest.raises(ValueError):
        core.read_core(paths[2], paths[3], case)


@pytest.mark.parametrize("mode", ["empty", "one-missing", "foreign-script"])
def test_core_reader_requires_both_actual_source_code_bindings(tmp_path, mode):
    paths = prepared_core_map(tmp_path)
    protocol = json.loads((paths[2].parent / "protocol.json").read_text())
    if mode == "empty":
        protocol["code"] = []
    elif mode == "one-missing":
        protocol["code"] = protocol["code"][:1]
    else:
        protocol["code"][1] = frozen_mapper.record(Path(__file__))
    rebind_protocol(paths, protocol)
    case = json.loads(paths[1].read_text())["case"]
    with pytest.raises(ValueError):
        core.read_core(paths[2], paths[3], case)


def test_core_reader_checks_delivered_body_digest_against_retained_raw(tmp_path):
    paths = prepared_core_map(tmp_path)
    receipt_path = paths[2].parent / "network-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["delivered_body_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt))
    case = json.loads(paths[1].read_text())["case"]
    with pytest.raises(ValueError):
        core.read_core(paths[2], paths[3], case)


@pytest.mark.parametrize("field,value", [
    ("osm_snapshot", "2026-07-02T00:00:00Z"),
    ("event_date", "2026-08-21"),
    ("context_bbox", [85.22, 27.59, 85.38, 27.85]),
])
def test_core_reader_rejects_another_case_identity(tmp_path, field, value):
    paths = prepared_core_map(tmp_path)
    case = json.loads(paths[1].read_text())["case"]
    case[field] = value
    with pytest.raises(ValueError):
        core.read_core(paths[2], paths[3], case)
