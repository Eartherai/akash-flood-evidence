"""Independent bounded-transport checks; fake responses perform no network IO."""
import json

import pytest

from test_acquire_historic_osm import Response, Session, module, run


def receipt(tmp_path):
    return json.loads((tmp_path / "case/network-receipt.json").read_text())


def test_stream_cap_counts_rejected_chunk_and_retains_only_written_prefix(tmp_path, monkeypatch):
    class Chunked(Response):
        def iter_content(self, size):
            assert size == 65_536
            yield b"abc"
            yield b"def"

    monkeypatch.setattr(module, "MAX_BYTES", 5)
    response = Chunked(b"")
    client = Session(response)
    with pytest.raises(ValueError, match="byte cap"):
        run(tmp_path, client, execute=True)
    saved = receipt(tmp_path)
    assert saved["status"] == "failed" and saved["received_body_bytes"] == 6
    assert saved["retained_partial_response"]["bytes"] == 3
    assert (tmp_path / "case/received-response.partial.bin").read_bytes() == b"abc"
    assert response.closed and len(client.calls) == 1
    assert not (tmp_path / "case/osm.json").exists()


def test_advertised_over_budget_body_is_not_read(tmp_path, monkeypatch):
    class Unread(Response):
        def iter_content(self, size):
            pytest.fail("Over-budget advertised body must not be streamed")
            yield b""

    monkeypatch.setattr(module, "MAX_BYTES", 5)
    response = Unread(b"", advertised=6)
    with pytest.raises(ValueError, match="excessive size"):
        run(tmp_path, Session(response), execute=True)
    assert receipt(tmp_path)["received_body_bytes"] == 0
    assert response.closed


@pytest.mark.parametrize("mode", ["redirect", "compression"])
def test_redirect_and_compressed_bodies_are_rejected_without_retry(tmp_path, mode):
    response = Response(b"{}", status=302 if mode == "redirect" else 200)
    if mode == "compression":
        response.headers["Content-Encoding"] = "gzip"
    client = Session(response)
    with pytest.raises(ValueError):
        run(tmp_path, client, execute=True)
    saved = receipt(tmp_path)
    assert saved["attempted_requests"] == 1 and saved["received_body_bytes"] == 0
    assert len(client.calls) == 1 and response.closed
    assert client.calls[0][1]["allow_redirects"] is False
    assert not (tmp_path / "case/osm.json").exists()


@pytest.mark.parametrize("mode", ["auth", "header"])
def test_explicit_session_credentials_are_rejected_before_get(tmp_path, mode):
    client = Session(error=AssertionError("No request with credentials"))
    if mode == "auth":
        client.auth = ("test", "test")
    else:
        client.headers = {"aUtHoRiZaTiOn": "test"}
    with pytest.raises(ValueError, match="no credentials"):
        run(tmp_path, client, execute=True)
    assert not client.calls
    assert client.trust_env is False


def test_prior_attempt_cannot_be_overwritten_even_in_offline_mode(tmp_path):
    run(tmp_path)
    frozen = (tmp_path / "case/protocol.json").read_bytes()
    with pytest.raises(FileExistsError, match="Preserve previous attempts"):
        run(tmp_path)
    assert (tmp_path / "case/protocol.json").read_bytes() == frozen


@pytest.mark.parametrize("elements", [
    [{}],
    [{"type": "node", "id": True, "lon": 85.3, "lat": 27.7}],
    [{"type": "area", "id": 1}],
    [{"type": "node", "id": 1, "lon": float("nan"), "lat": 27.7}],
    [{"type": "node", "id": 1, "lon": 85.3, "lat": 91}],
    [{"type": "way", "id": 1, "nodes": [1, False]}],
    [{"type": "relation", "id": 1, "members": [{"type": "way", "ref": 0}]}],
    [{"type": "node", "id": 1, "lon": 85.3, "lat": 27.7, "tags": {"name": 5}}],
    [{"type": "node", "id": 1, "lon": 85.3, "lat": 27.7},
     {"type": "node", "id": 1, "lon": 85.31, "lat": 27.71}],
])
def test_malformed_or_duplicate_source_objects_cannot_be_promoted(tmp_path, elements):
    # NaN is included deliberately: Python's JSON reader permits it by default,
    # but an invalid geographic coordinate must not become a completed extract.
    payload = json.dumps({"elements": elements}).encode()
    response = Response(payload)
    with pytest.raises(ValueError):
        run(tmp_path, Session(response), execute=True)
    saved = receipt(tmp_path)
    assert saved["status"] == "failed"
    assert saved["received_body_bytes"] == len(payload)
    assert saved["retained_partial_response"]["bytes"] == len(payload)
    assert response.closed
    assert not (tmp_path / "case/response.json").exists()
    assert not (tmp_path / "case/osm.json").exists()


def test_valid_node_way_relation_id_namespaces_and_historic_timestamp_are_accepted(tmp_path):
    elements = [
        {"type": "node", "id": 1, "lon": 85.3, "lat": 27.7,
         "timestamp": "2024-08-30T00:00:00Z", "tags": {"name": "One"}},
        {"type": "node", "id": 2, "lon": 85.31, "lat": 27.71},
        {"type": "way", "id": 1, "nodes": [1, 2], "tags": {"highway": "secondary"}},
        {"type": "relation", "id": 1, "members": [{"type": "way", "ref": 1, "role": "outer"}]},
    ]
    result = run(tmp_path, Session(Response(json.dumps({"elements": elements}).encode())), execute=True)
    assert result["status"] == "completed" and result["element_count"] == 4
    assert json.loads((tmp_path / "case/osm.json").read_text())["elements"] == elements


def test_well_formed_node_with_future_edit_is_rejected_for_the_cutoff(tmp_path):
    payload = json.dumps({"elements": [{"type": "node", "id": 1, "lon": 85.3,
        "lat": 27.7, "timestamp": "2024-09-01T00:00:01Z"}]}).encode()
    with pytest.raises(ValueError, match="edit after the cutoff"):
        run(tmp_path, Session(Response(payload)), execute=True)
    assert receipt(tmp_path)["status"] == "failed"
    assert not (tmp_path / "case/osm.json").exists()
