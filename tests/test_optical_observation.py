"""Observation opportunity keeps dates, missing data and failed transport explicit."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import rasterio

from akash_flood.optical_observation import (AuditedSCLRangeHTTP, PUBLIC_HOST,
    eroded_land_support, select_observation_sources, summarize_availability, update_date_index)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_s2_observation_coverage as cli


def item(identity, when, cloud=0, tile="UM"):
    return {
        "type": "Feature", "id": identity, "collection": "sentinel-2-c1-l2a",
        "geometry": {"type": "Polygon", "coordinates": [[[85, 28], [86, 28], [86, 29], [85, 29], [85, 28]]]},
        "properties": {"datetime": when, "platform": "sentinel-2a", "proj:epsg": 32645,
                       "mgrs:utm_zone": 45, "mgrs:latitude_band": "R", "mgrs:grid_square": tile, "eo:cloud_cover": cloud},
        "assets": {"scl": {
            "href": "https://" + PUBLIC_HOST + "/sentinel-2-c1-l2a/45/R/" + tile + "/2026/8/" + identity + "/SCL.tif",
            "gsd": 20, "file:size": 12345, "file:checksum": "1220" + "a" * 64,
            "proj:shape": [5490, 5490], "proj:transform": [20, 0, 300000, 0, -20, 3200040],
            "raster:bands": [{"nodata": 0, "data_type": "uint8", "spatial_resolution": 20}],
        }},
    }


def catalog(*items):
    return {"type": "FeatureCollection", "features": list(items)}


BOX = [85.30, 28.12, 85.301, 28.121]
EVENT = "2026-08-26"


def test_selector_uses_every_eligible_source_time_id_ties_not_clouds_or_event_day():
    doc = catalog(item("preold", "2026-08-10T10:00:00Z", 0),
                  item("pretie-b", "2026-08-25T10:00:00Z", 0),
                  item("pretie-a", "2026-08-25T10:00:00Z", 100),
                  item("eventday", "2026-08-26T23:59:59Z"),
                  item("postlater", "2026-08-29T10:00:00Z", 0),
                  item("postearliest", "2026-08-27T10:00:00Z", 100))
    sources, excluded = select_observation_sources(doc, BOX, EVENT)
    assert [x["id"] for x in sources["before"]] == ["pretie-a", "pretie-b", "preold"]
    assert [x["id"] for x in sources["after"]] == ["postearliest", "postlater"]
    assert excluded == ["eventday"]


def test_selector_rejects_incomplete_search_mixed_tiles_shifted_grids_and_reference_assets():
    pre, post = item("pre", "2026-08-25T10:00:00Z"), item("post", "2026-08-27T10:00:00Z")
    doc = catalog(pre, post)
    doc["search_complete"] = False
    with pytest.raises(ValueError, match="complete"):
        select_observation_sources(doc, BOX, EVENT)
    post["properties"]["mgrs:grid_square"] = "UN"
    with pytest.raises(ValueError, match="Multiple tiles"):
        select_observation_sources(catalog(pre, post), BOX, EVENT)
    post["properties"]["mgrs:grid_square"] = "UM"
    post["assets"]["scl"]["proj:transform"][2] += 10
    with pytest.raises(ValueError, match="differing native grids"):
        select_observation_sources(catalog(pre, post), BOX, EVENT)
    post = item("post", "2026-08-27T10:00:00Z")
    post["assets"]["scl"]["roles"] = ["reference"]
    with pytest.raises(ValueError, match="excluded"):
        select_observation_sources(catalog(pre, post), BOX, EVENT)


def test_outside_temporal_window_and_partial_footprints_do_not_supply_observation():
    pre, post = item("pre", "2026-08-25T10:00:00Z"), item("post", "2026-08-27T10:00:00Z")
    outside = item("late", "2026-09-21T00:00:01Z")
    partial = item("partial", "2026-08-28T10:00:00Z")
    partial["geometry"]["coordinates"][0][1][0] = 85.1
    partial["geometry"]["coordinates"][0][2][0] = 85.1
    sources, excluded = select_observation_sources(catalog(pre, post, outside, partial), BOX, EVENT)
    assert len(sources["after"]) == 1 and set(excluded) == {"late", "partial"}


def test_24day_window_is_relative_to_event_start_and_never_silently_extends_an_extra_day():
    pre = item("pre", "2026-08-25T10:00:00Z")
    exact = item("boundary", "2026-09-19T00:00:00Z")
    outside = item("outside", "2026-09-19T00:00:01Z")
    sources, excluded = select_observation_sources(catalog(pre, exact, outside), BOX, EVENT)
    assert [x["id"] for x in sources["after"]] == ["boundary"] and excluded == ["outside"]


def test_source_erosion_excludes_water_cloud_snow_and60m_euclidean_neighborhood():
    scl = np.full((21, 21), 4, dtype="uint8")
    scl[10, 10] = 9
    support = eroded_land_support(scl)
    assert not support[10, 13] and support[10, 14] and support[13, 13]
    assert not support[0].any() and not support[:3].any()
    for code in (0, 1, 2, 3, 6, 7, 8, 9, 10, 11):
        assert not eroded_land_support(np.full((9, 9), code, dtype="uint8")).any()
    assert eroded_land_support(np.full((9, 9), 5, dtype="uint8"))[4, 4]
    with pytest.raises(ValueError, match="class domain"):
        eroded_land_support(np.full((9, 9), 255, dtype="uint8"))


def test_per_cell_dates_retain_first_supported_preference_and_fill_only_missing_cells():
    aoi = np.ones((21, 21), bool)
    first = np.full((21, 21), 4, dtype="uint8")
    first[:, :10] = 9
    second = np.full((21, 21), 5, dtype="uint8")
    indexes = np.zeros(aoi.shape, dtype="uint8")
    one = update_date_index(indexes, first, aoi, 1)
    two = update_date_index(indexes, second, aoi, 2)
    assert indexes[10, 16] == 1 and indexes[10, 5] == 2 and indexes[0, 0] == 0
    assert one["selected_source_pixels"] + two["selected_source_pixels"] == np.count_nonzero(indexes)
    joint, counts, fraction = summarize_availability(indexes, indexes.copy(), aoi)
    assert joint[10, 5] and counts["joint_unknown_pixels"] > 0
    assert counts["joint_available_pixels"] + counts["joint_unknown_pixels"] == aoi.sum()
    assert fraction == counts["joint_available_pixels"] / aoi.sum()


def test_pre_and_post_must_both_supply_source_and_outside_aoi_is_distinct():
    aoi = np.ones((5, 5), bool)
    aoi[0] = False
    before = np.zeros(aoi.shape, dtype="uint8")
    after = before.copy()
    before[1, 1] = 1
    after[2, 2] = 2
    joint, counts, _ = summarize_availability(before, after, aoi)
    assert not joint.any() and counts["joint_unknown_pixels"] == 20
    before[0, 0] = 1
    with pytest.raises(ValueError, match="zero outside"):
        summarize_availability(before, after, aoi)


class Response:
    def __init__(self, data=b"abcd", status=206, etag="fixed", content_range="bytes 0-3/100", error=None):
        self.data, self.status_code, self.error = data, status, error
        self.headers = {"Content-Range": content_range, "ETag": etag, "Content-Length": "4"}
        self.closed = False

    def iter_content(self, n):
        yield self.data
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


class Session:
    def __init__(self, results):
        self.results, self.calls = iter(results), []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


URL = "https://" + PUBLIC_HOST + "/sentinel-2-c1-l2a/45/R/UM/2026/8/a/SCL.tif"


def test_successful_range_captures_sha_and_uses_tls_no_redirect_environment_or_retry():
    response = Response()
    session = Session([response])
    client = AuditedSCLRangeHTTP(session=session)
    assert client.range(URL, 0, 4) == (b"abcd", 100)
    audit = client.audit()
    assert audit["received_body_bytes"] == 4 and audit["attempted_get_count"] == 1 and audit["failed_get_count"] == 0
    assert audit["requests"][0]["sha256"] == hashlib.sha256(b"abcd").hexdigest()
    assert session.calls[0][1]["verify"] is True and session.calls[0][1]["allow_redirects"] is False
    assert session.trust_env is False and response.closed


def test_short_object_header_range_is_honestly_reported_if_it_receives_the_whole_object():
    response = Response(content_range="bytes 0-3/4")
    client = AuditedSCLRangeHTTP(session=Session([response]))
    assert client.range(URL, 0, 65536) == (b"abcd", 4)
    audit = client.audit()
    assert audit["source_full_objects_downloaded"] is True
    assert audit["unbounded_whole_scene_fallback_used"] is False
    assert audit["provider_whole_object_checksums_verified"] is False


def test_connection_failure_and_partial_stream_failure_both_count_attempts_and_delivered_bytes():
    response = Response(b"ab", error=ConnectionError("stream broken"))
    session = Session([ConnectionError("connect failed"), response])
    client = AuditedSCLRangeHTTP(session=session)
    with pytest.raises(ConnectionError):
        client.range(URL, 0, 4)
    with pytest.raises(ConnectionError):
        client.range(URL, 0, 4)
    audit = client.audit()
    assert len(session.calls) == 2 and audit["attempted_get_count"] == audit["failed_get_count"] == 2
    assert audit["received_body_bytes"] == 2 and [x["received_body_bytes"] for x in audit["requests"]] == [0, 2]
    assert audit["requests"][1]["partial_sha256"] == hashlib.sha256(b"ab").hexdigest() and response.closed


@pytest.mark.parametrize("response", [Response(status=200), Response(status=302),
    Response(content_range="bytes 1-4/100"), Response(etag=""), Response(b"abc")])
def test_rejected_response_or_short_body_is_a_preserved_failed_attempt(response):
    client = AuditedSCLRangeHTTP(session=Session([response]))
    with pytest.raises(ValueError):
        client.range(URL, 0, 4)
    audit = client.audit()
    assert audit["attempted_get_count"] == audit["failed_get_count"] == 1
    assert audit["requests"][0]["error"]["type"] == "ValueError" and response.closed
    assert audit["received_body_bytes"] == (3 if response.data == b"abc" else 0)


def test_failed_partial_bytes_and_failed_attempts_reduce_future_budgets():
    client = AuditedSCLRangeHTTP(5, session=Session([Response(b"ab", error=ConnectionError())]))
    with pytest.raises(ConnectionError):
        client.range(URL, 0, 4)
    with pytest.raises(ValueError, match="reservation"):
        client.range(URL, 0, 4)
    assert client.audit()["received_body_bytes"] == 2 and len(client.requests) == 1
    client = AuditedSCLRangeHTTP(max_attempts=1, session=Session([ConnectionError()]))
    with pytest.raises(ConnectionError):
        client.range(URL, 0, 4)
    with pytest.raises(ValueError, match="attempted GET"):
        client.range(URL, 0, 4)
    assert len(client.requests) == 1


def test_remote_identity_change_is_refused_and_if_match_is_sent():
    second = Response(b"efgh", content_range="bytes 4-7/100", etag="changed")
    session = Session([Response(), second])
    client = AuditedSCLRangeHTTP(session=session)
    client.range(URL, 0, 4)
    with pytest.raises(ValueError, match="identity changed"):
        client.range(URL, 4, 8)
    assert session.calls[1][1]["headers"]["If-Match"] == "fixed"
    assert client.audit()["failed_get_count"] == 1 and client.bytes_read == 4


def test_transport_rejects_non_scl_credentials_queries_oversized_ranges_before_get():
    client = AuditedSCLRangeHTTP(session=Session([]))
    for url in (URL.replace("SCL.tif", "B04.tif"), URL + "?token=secret", URL.replace("https://", "http://"),
                URL.replace(PUBLIC_HOST, "untrusted.example"), URL.replace("https://", "https://user:pass@")):
        with pytest.raises(ValueError, match="unsigned public"):
            client.range(url, 0, 4)
    with pytest.raises(ValueError, match="limit"):
        client.range(URL, 0, 4_000_001)
    assert not client.requests and not client.client.calls


@pytest.mark.parametrize("attribute,value", [("auth", ("user", "password")),
    ("params", {"token": "secret"}), ("proxies", {"https": "http://user:password@proxy.example"}),
    ("headers", {"X-Amz-Security-Token": "secret"})])
def test_hidden_session_credentials_queries_or_proxies_are_refused(attribute, value):
    session = Session([])
    setattr(session, attribute, value)
    with pytest.raises(ValueError, match="Credential-bearing"):
        AuditedSCLRangeHTTP(session=session)
    assert not session.calls


def test_missing_length_or_encoded_body_is_rejected_before_read_and_recorded():
    for field, value in (("Content-Length", "5"), ("Content-Encoding", "gzip")):
        response = Response()
        response.headers[field] = value
        client = AuditedSCLRangeHTTP(session=Session([response]))
        with pytest.raises(ValueError, match="declared body length"):
            client.range(URL, 0, 4)
        assert client.audit()["failed_get_count"] == 1 and client.bytes_read == 0 and response.closed


def test_default_cli_freezes_sources_without_client_construction_or_pixels(tmp_path, monkeypatch):
    source = tmp_path / "catalog.json"
    cli.dump(source, catalog(item("pre", "2026-08-25T10:00:00Z"), item("post", "2026-08-27T10:00:00Z")))
    monkeypatch.setattr(cli, "AuditedSCLRangeHTTP", lambda: pytest.fail("Default protocol constructed a network client"))
    monkeypatch.setattr(cli, "read_band", lambda *args: pytest.fail("Default protocol read pixels"))
    out = tmp_path / "plan"
    cli.main(["--catalog", str(source), "--bbox", *map(str, BOX), "--event-date", EVENT, "--out", str(out)])
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["execute_requested"] is False and protocol["reference_labels_loaded"] is False
    assert len(protocol["ordered_sources"]) == 2 and not (out / "manifest.json").exists()
    assert not list(out.glob("*.tif"))


@pytest.fixture
def archived_pair(tmp_path):
    sources = {"before": [item("pre", "2026-08-25T10:00:00Z")], "after": [item("post", "2026-08-27T10:00:00Z")]}
    window, grid, _ = cli.window_for(sources["before"][0], BOX)
    old = tmp_path / "old"
    old.mkdir()
    items, crops, requests = {}, {}, []
    for branch, scene in ((name, group[0]) for name, group in sources.items()):
        item_path = old / (branch + ".stac.json")
        cli.dump(item_path, scene)
        items[branch] = cli.record(item_path)
        folder = old / branch
        folder.mkdir()
        crop_path = folder / "scl-dn.tif"
        cli.write_layer(crop_path, np.full((int(grid[3]), int(grid[2])), 4, dtype="uint8"), grid, 0)
        asset = scene["assets"]["scl"]
        receipt = {"crop": cli.record(crop_path), "asset": asset, "source_object_full_checksum_verified": False,
                   "remote_identity": {"bytes": asset["file:size"], "etag": "fixed"},
                   "native_window": list(window.flatten()), "horizontal_aggregation": "none"}
        cli.dump(folder / "scl-crop-receipt.json", receipt)
        crops[branch] = {"scl": receipt}
        requests.append({"url": asset["href"], "status": 206, "file_bytes": asset["file:size"], "etag": "fixed", "sha256": "a" * 64})
    cli.dump(old / "protocol.json", {"schema": "akash-s2-surface-change-protocol-v1", "source_items": items,
             "bbox": BOX, "event_date": EVENT, "reference_labels_loaded": False})
    cli.dump(old / "transport.json", {"requests": requests})
    cli.dump(old / "manifest.json", {"schema": "akash-s2-surface-change-v1", "status": "completed",
             "bbox": BOX, "event_date": EVENT, "reference_labels_loaded": False,
             "protocol": cli.record(old / "protocol.json"), "transport": cli.record(old / "transport.json"), "source_crops": crops})
    return old, sources, window, grid


def test_archived_scl_pair_reuse_is_exact_and_has_no_pixel_read_during_freeze(archived_pair, tmp_path, monkeypatch):
    old, sources, window, grid = archived_pair
    # Metadata-only read is part of provenance validation; neither read_band nor a network client is involved.
    reuse = cli.prepare_scl_reuse(old, sources, BOX, EVENT, window, grid)
    assert len(reuse["crops"]) == 2 and reuse["pixel_reads_during_protocol_preparation"] is False
    folder = tmp_path / "imported"
    folder.mkdir()
    values, receipt = cli.import_scl_crop(sources["before"][0], reuse["crops"]["pre"], window, grid, folder)
    assert (values == 4).all() and receipt["source_object_full_checksum_verified"] is False
    assert receipt["crop"]["sha256"] == reuse["crops"]["pre"]["crop"]["sha256"]


def test_archived_pair_changed_crop_receipt_or_source_item_is_rejected(archived_pair):
    old, sources, window, grid = archived_pair
    changed_sources = copy.deepcopy(sources)
    changed_sources["before"][0]["properties"]["eo:cloud_cover"] = 50
    with pytest.raises(ValueError, match="exact selected"):
        cli.prepare_scl_reuse(old, changed_sources, BOX, EVENT, window, grid)
    receipt = old / "before/scl-crop-receipt.json"
    payload = json.loads(receipt.read_text())
    payload["remote_identity"]["etag"] = "other"
    cli.dump(receipt, payload)
    with pytest.raises(ValueError, match="receipt differs"):
        cli.prepare_scl_reuse(old, sources, BOX, EVENT, window, grid)


def test_reuse_hash_failure_preserves_unknown_instead_of_reading_or_repairing(archived_pair, tmp_path):
    old, sources, window, grid = archived_pair
    reuse = cli.prepare_scl_reuse(old, sources, BOX, EVENT, window, grid)
    Path(reuse["crops"]["pre"]["crop"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="binding changed"):
        cli.import_scl_crop(sources["before"][0], reuse["crops"]["pre"], window, grid, tmp_path)


def test_execute_with_two_verified_imports_writes_native_date_masks_without_network(archived_pair, tmp_path, monkeypatch):
    old, sources, _, grid = archived_pair
    source = tmp_path / "catalog.json"
    cli.dump(source, catalog(*[group[0] for group in sources.values()]))
    monkeypatch.setattr(cli, "read_band", lambda *args: pytest.fail("Verified two-source imports unexpectedly used network"))
    out = tmp_path / "execution"
    cli.main(["--catalog", str(source), "--bbox", *map(str, BOX), "--event-date", EVENT,
              "--reuse-scl", str(old), "--out", str(out), "--execute"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["network_attempted_gets"] == 0
    assert manifest["imported_scl_crop_count"] == 2 and manifest["candidate_layer_generated"] is False
    assert manifest["counts"]["joint_available_pixels"] > 0
    for branch, source_index in (("before", 1), ("after", 2)):
        with rasterio.open(out / (branch + "-date-index.tif")) as ds:
            assert ds.transform == grid[1] and ds.crs == grid[0] and ds.nodata == 0
            assert set(np.unique(ds.read(1))) == {0, source_index}
    assert all(cli.record(Path(binding["path"])) == binding for binding in manifest["outputs"].values())


def test_failed_execute_preserves_partial_byte_and_attempt_ledger(tmp_path, monkeypatch):
    source = tmp_path / "catalog.json"
    cli.dump(source, catalog(item("pre", "2026-08-25T10:00:00Z"), item("post", "2026-08-27T10:00:00Z")))
    client = AuditedSCLRangeHTTP(session=Session([Response(b"ab", error=ConnectionError("partial source failure"))]))
    monkeypatch.setattr(cli, "AuditedSCLRangeHTTP", lambda: client)
    def fail_read(scene, key, window, grid, transport, folder):
        return transport.range(scene["assets"]["scl"]["href"], 0, 4)
    monkeypatch.setattr(cli, "read_band", fail_read)
    out = tmp_path / "failed"
    with pytest.raises(ConnectionError, match="partial source"):
        cli.main(["--catalog", str(source), "--bbox", *map(str, BOX), "--event-date", EVENT,
                  "--out", str(out), "--execute"])
    failure = json.loads((out / "failed.json").read_text())
    transport = json.loads((out / "transport.json").read_text())
    assert failure["received_body_bytes"] == transport["received_body_bytes"] == 2
    assert failure["attempted_get_count"] == transport["failed_get_count"] == 1
    assert not (out / "manifest.json").exists() and failure["no_automatic_retry"] is True
