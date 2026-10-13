"""Protect fresh acquisition identity, bounded transport and derivative claims."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from akash_flood import fresh_native as native


def make_item(day=1):
    start = f"202601{day:02}T100000"
    end = f"202601{day:02}T100025"
    identity = f"S1A_IW_GRDH_1SDV_{start}_{end}_000001_000ABC"
    product = identity + "_AAAA"
    prefix = f"GRD/2026/1/{day}/IW/DV/{product}"
    return {"type": "Feature", "collection": "sentinel-1-grd", "id": identity, "properties": {
        "s1:product_identifier": product, "platform": "sentinel-1a", "sar:instrument_mode": "IW", "sar:product_type": "GRD", "sar:polarizations": ["VV", "VH"], "sat:orbit_state": "ascending", "sat:relative_orbit": 1,
        "start_datetime": f"2026-01-{day:02}T10:00:00Z", "end_datetime": f"2026-01-{day:02}T10:00:25Z", "datetime": f"2026-01-{day:02}T10:00:12Z"},
        "links": [{"rel": "self", "href": f"https://earth-search.aws.element84.com/v1/collections/sentinel-1-grd/items/{identity}"}],
        "assets": {key: {"href": f"s3://{native.BUCKET}/{prefix}/{relative}", "roles": ["data" if key in {"vv", "vh"} else "metadata"]} for key, relative in {"vv": "measurement/iw-vv.tiff", "vh": "measurement/iw-vh.tiff", "safe-manifest": "manifest.safe"}.items()}}


def write_item(tmp_path, name, day=1):
    path = tmp_path / f"{name}.stac.json"
    path.write_text(json.dumps(make_item(day)), encoding="utf-8")
    return native.validate_scene(name, path)


def make_manifest(scene, mutate_alias=None):
    root = ET.Element("XFDU")
    acquire = ET.SubElement(root, "acquisitionPeriod")
    ET.SubElement(acquire, "startTime").text = scene.item["properties"]["start_datetime"]
    ET.SubElement(acquire, "stopTime").text = scene.item["properties"]["end_datetime"]
    identities = {"orbitNumber": "1", "relativeOrbitNumber": "1", "pass": "ASCENDING", "mode": "IW", "productType": "GRD", "missionDataTakeID": str(int(scene.item_id.split("_")[7], 16)), "number": "A"}
    for key, value in identities.items():
        ET.SubElement(root, key).text = value
    ET.SubElement(root, "transmitterReceiverPolarisation").text = "VV"
    ET.SubElement(root, "transmitterReceiverPolarisation").text = "VH"
    section = ET.SubElement(root, "dataObjectSection")
    mapping = {}
    parts = scene.item_id.lower().split("_")
    for pol, sequence in (("vv", "001"), ("vh", "002")):
        body = f"{parts[0]}-iw-grd-{pol}-{parts[4]}-{parts[5]}-{parts[6]}-{parts[7]}-{sequence}"
        for kind, alias, relative, data in (
            ("measurement", f"measurement/{body}.tiff", f"measurement/iw-{pol}.tiff", b"COG-BYTES-" + pol.encode()),
            ("annotation", f"annotation/{body}.xml", f"annotation/iw-{pol}.xml", b"<annotation>fixture</annotation>"),
            ("noise", f"annotation/calibration/noise-{body}.xml", f"annotation/calibration/noise-iw-{pol}.xml", b"<noise>fixture</noise>"),
            ("calibration", f"annotation/calibration/calibration-{body}.xml", f"annotation/calibration/calibration-iw-{pol}.xml", b"<calibration>fixture</calibration>"),
        ):
            obj = ET.SubElement(section, "dataObject", {"ID": f"{kind}-{pol}"})
            stream = ET.SubElement(obj, "byteStream", {"size": "999"})
            ET.SubElement(stream, "fileLocation", {"href": "./" + (mutate_alias(alias) if mutate_alias else alias)})
            ET.SubElement(stream, "checksum", {"checksumName": "MD5"}).text = "a" * 32
            if kind != "measurement":
                metadata = ET.Element(kind); header = ET.SubElement(metadata, "adsHeader")
                values = {"missionId": "S1A", "productType": "GRD", "polarisation": pol.upper(), "mode": "IW", "swath": "IW", "absoluteOrbitNumber": "1", "missionDataTakeId": str(int(parts[7], 16)), "startTime": scene.item["properties"]["start_datetime"], "stopTime": scene.item["properties"]["end_datetime"]}
                for key, value in values.items():
                    ET.SubElement(header, key).text = value
                data = ET.tostring(metadata)
            mapping[f"https://{native.HOST}/{scene.prefix}/{relative}"] = data
    manifest = ET.tostring(root, encoding="utf-8")
    mapping[scene.manifest_url] = manifest
    return manifest, mapping


class FakeResponse:
    def __init__(self, data, status=200, headers=None, chunks=None):
        self.status_code = status
        self.headers = {"Content-Length": str(len(data)), "ETag": '"immutable-fixture"'} | (headers or {})
        self.data = data
        self.chunks = chunks
        self.closed = False

    def close(self):
        self.closed = True

    def iter_content(self, chunk_size):
        yield from self.chunks if self.chunks is not None else [self.data]


class FakeSession:
    def __init__(self, sources):
        self.sources = sources
        self.calls = []
        self.overrides = {}

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.overrides.get((method, url), FakeResponse(self.sources[url]))


def test_source_identity_and_label_inputs_rejected(tmp_path):
    good = make_item()
    path = tmp_path / "before.json"
    for edit, expected in (
        (lambda value: value["assets"]["vv"].update(href=value["assets"]["vv"]["href"].replace("_AAAA/", "_BBBB/")), "source identity"),
        (lambda value: value["assets"].update(damage={"href": "https://example.org/labels.tif", "roles": ["labels"]}), "Non-source"),
        (lambda value: value["assets"]["vv"].update(roles=["LABELS"]), "Published"),
        (lambda value: value["properties"].update(reference_labels_loaded=True), "Published"),
        (lambda value: value["properties"].update(datetime="2026-01-02T10:00:00Z"), "inconsistent"),
    ):
        bad = deepcopy(good); edit(bad); path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=expected):
            native.validate_scene("before", path)
    path.write_text(json.dumps(good))
    with pytest.raises(ValueError, match="checksum"):
        native.validate_scene("before", path, "0" * 64)


def test_pair_rejects_orbit_platform_and_chronology(tmp_path):
    before, after = write_item(tmp_path, "before", 1), write_item(tmp_path, "after", 7)
    native.validate_pair(before, after)
    for key, value in (("sat:relative_orbit", 2), ("platform", "sentinel-1b")):
        changed = deepcopy(after.item); changed["properties"][key] = value
        after.item_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="pairing differs|Platform differs"):
            candidate = native.validate_scene("after", after.item_path)
            native.validate_pair(before, candidate)
    with pytest.raises(ValueError, match="chronologically"):
        native.validate_pair(before, before)


def test_manifest_paths_full_identity_and_source_completeness(tmp_path):
    scene = write_item(tmp_path, "before")
    content, _ = make_manifest(scene)
    _, objects = native.manifest_objects(content, scene)
    assert len(objects) == 8
    for mutation in (lambda alias: "../outside.tiff", lambda alias: alias.replace("-000001-", "-000002-")):
        bad, _ = make_manifest(scene, mutation)
        with pytest.raises(ValueError, match="Unsafe|identity differs"):
            native.manifest_objects(bad, scene)
    with pytest.raises(ValueError, match="document/entity"):
        native.manifest_objects(b'<!DOCTYPE a [<!ENTITY x "boom">]>' + content, scene)
    root = ET.fromstring(content)
    section = root.find("dataObjectSection"); section.remove(section[-1])
    with pytest.raises(ValueError, match="lacks required"):
        native.manifest_objects(ET.tostring(root), scene)


def test_transport_caps_redirects_etag_and_tls_are_enforced(tmp_path):
    url = f"https://{native.HOST}/GRD/2026/1/1/IW/DV/fixture/measurement/iw-vv.tiff"
    session = FakeSession({url: b"1234"})
    client = native.PublicClient(native.NetworkBudget(100), session)
    advertised = client.head(url, 4)
    target = tmp_path / "bytes.tiff"
    receipt = client.get(url, target, advertised)
    assert receipt["sha256"] == hashlib.sha256(b"1234").hexdigest()
    assert client.budget.transferred == 4
    assert all(call[2]["verify"] is True and call[2]["allow_redirects"] is False for call in session.calls)
    assert session.calls[-1][2]["headers"]["If-Match"] == '"immutable-fixture"'
    with pytest.raises(ValueError, match="size"):
        client.head(url, 3)
    session.overrides[("HEAD", url)] = FakeResponse(b"1234", status=302)
    with pytest.raises(ValueError, match="redirects/auth"):
        client.head(url, 4)
    session.overrides[("HEAD", url)] = FakeResponse(b"1234", headers={"ETag": "W/weak"})
    with pytest.raises(ValueError, match="Strong provider"):
        client.head(url, 4)
    session.overrides[("GET", url)] = FakeResponse(b"1234", headers={"ETag": '"different"'})
    with pytest.raises(ValueError, match="changed identity"):
        client.get(url, tmp_path / "changed.tiff", advertised)
    assert not (tmp_path / "changed.tiff.partial").exists()
    for forbidden in (url.replace("https:", "http:"), url + "?token=x", url.replace(native.HOST, "localhost")):
        with pytest.raises(ValueError, match="strict public"):
            client.head(forbidden, 4)


def test_honest_and_dishonest_body_budget_fail_without_completed_bytes(tmp_path):
    url = f"https://{native.HOST}/GRD/fixture/measurement/iw-vv.tiff"
    session = FakeSession({url: b"1234"})
    client = native.PublicClient(native.NetworkBudget(3), session)
    advertised = client.head(url, 4)
    with pytest.raises(ValueError, match="budget"):
        client.get(url, tmp_path / "budget.tiff", advertised)
    assert all(method == "HEAD" for method, _, _ in session.calls)
    client = native.PublicClient(native.NetworkBudget(100), session)
    session.overrides[("GET", url)] = FakeResponse(b"1234", chunks=[b"123", b"456"])
    with pytest.raises(ValueError, match="body exceeds"):
        client.get(url, tmp_path / "oversize.tiff", advertised)
    assert client.budget.transferred == 6
    assert not (tmp_path / "oversize.tiff").exists()
    assert not (tmp_path / "oversize.tiff.partial").exists()


def test_acquisition_preserves_original_and_binds_every_derived_alias(tmp_path):
    scenes = [write_item(tmp_path, "before", 1), write_item(tmp_path, "after", 7)]
    sources = {}
    original = {}
    for scene in scenes:
        content, mapping = make_manifest(scene); sources.update(mapping); original[scene.name] = content
    session = FakeSession(sources)
    out = tmp_path / "output"
    index = native.acquire(scenes, out, tmp_path / "cache", budget_bytes=100_000, session=session)
    assert index["status"] == "completed" and index["not_original_esa_safe"] is True
    assert index["network"]["actual_received_body_bytes"] == sum(map(len, sources.values()))
    for scene in scenes:
        record = index["scenes"][scene.name]
        assert Path(record["original_manifest"]).read_bytes() == original[scene.name]
        root = ET.parse(record["derived_manifest"]).getroot()
        md5s = {e.text for e in root.iter() if e.tag == "checksum"}
        assert "a" * 32 not in md5s
        for entry in record["files"]:
            alias = Path(record["derived_safe"]) / entry["alias"]
            assert native.digest(alias) == entry["sha256"]
            assert native.digest(alias, "md5") == entry["derived_md5"]
            assert entry["original_manifest_checksum"] == "a" * 32
            assert entry["original_manifest_checksum_verifies_provider_bytes"] is False
    with pytest.raises(FileExistsError, match="must not exist"):
        native.acquire(scenes, out, tmp_path / "cache", budget_bytes=100_000, session=session)
    # Cached bytes are rehashed and revalidated against current ETag/size.
    session.calls.clear()
    native.acquire(scenes, tmp_path / "output2", tmp_path / "cache", budget_bytes=100_000, session=session)
    assert all(method == "HEAD" for method, _, _ in session.calls)
    measurement = next((tmp_path / "cache").rglob("iw-vv.tiff"))
    measurement.write_bytes(b"bad")
    with pytest.raises(ValueError, match="Cached source checksum"):
        native.acquire(scenes, tmp_path / "output3", tmp_path / "cache", budget_bytes=100_000, session=session)


def test_preflight_global_budget_stops_before_measurement_download(tmp_path):
    scenes = [write_item(tmp_path, "before", 1), write_item(tmp_path, "after", 7)]
    sources = {url: data for scene in scenes for url, data in make_manifest(scene)[1].items()}
    session = FakeSession(sources)
    out = tmp_path / "too-small"
    with pytest.raises(ValueError, match="Four measurements"):
        native.acquire(scenes, out, tmp_path / "cache", budget_bytes=10, session=session)
    assert len(session.calls) == 4 and all(method == "HEAD" for method, _, _ in session.calls)
    assert not out.exists()


def test_planner_sha_and_selected_source_identity_are_required(tmp_path):
    scenes = [write_item(tmp_path, "before", 1), write_item(tmp_path, "after", 7)]
    pair = {scene.name: {"item_path": str(scene.item_path), "item_sha256": scene.item_sha256, "item_id": scene.item_id, "datetime": scene.acquisition_time, "catalogue_item_canonical_sha256": native.canonical_sha(scene.item)} for scene in scenes}
    pair.update(relative_orbit=1, orbit_state="ascending")
    plan = {"case": {"case_id": "fixture", "bbox": [0, 0, 1, 1]}, "sentinel1": {"selected_pair": pair}}
    path = tmp_path / "plan.json"; path.write_text(json.dumps(plan))
    selected, binding = native.scenes_from_plan(path)
    assert selected[0].item_id == scenes[0].item_id and binding["sha256"] == native.digest(path)
    plan["sentinel1"]["selected_pair"]["before"]["item_sha256"] = "0" * 64
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="checksum"):
        native.scenes_from_plan(path)


def test_explicit_old_cache_import_is_offline_exact_and_not_a_new_download(tmp_path):
    scenes = [write_item(tmp_path, "before", 1), write_item(tmp_path, "after", 7)]
    sources = {url: data for scene in scenes for url, data in make_manifest(scene)[1].items()}
    old = tmp_path / "old-native"
    index = native.acquire(scenes, old, tmp_path / "first-cache", budget_bytes=100_000, session=FakeSession(sources))
    # The earlier cache contract was a list. Build that historical-format fixture.
    (old / "derived-wrapper-index.json").write_text(json.dumps(list(index["scenes"].values())))
    imports = native.inspect_existing_native_cache(scenes, old)
    assert len(imports) == 18
    session = FakeSession({})
    replay = native.acquire(scenes, tmp_path / "replay", tmp_path / "replay-cache", budget_bytes=1, imported_sources=imports, session=session)
    assert session.calls == []
    assert replay["network"]["actual_received_body_bytes"] == 0
    assert replay["network"]["measurement_advertised_network_bytes"] == 0
    assert replay["network"]["imported_asset_remote_revalidation"] is False
    for record in replay["scenes"].values():
        for entry in record["files"]:
            receipt = native.json_read(Path(entry["receipt_path"]))
            assert receipt["remote_revalidated"] is False
            assert receipt["etag"] is None
            assert "no new remote identity/download claim" in receipt["transport"]
    source = Path(next(iter(imports.values()))["path"])
    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="bytes differ"):
        native.inspect_existing_native_cache(scenes, old)


def test_duplicate_input_fields_and_missing_frozen_planner_sha_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"; path.write_text('{"type":"Feature","type":"Feature"}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        native.json_read(path)
    scenes = [write_item(tmp_path, "before", 1), write_item(tmp_path, "after", 7)]
    pair = {scene.name: {"item_path": str(scene.item_path), "item_id": scene.item_id, "datetime": scene.acquisition_time, "catalogue_item_canonical_sha256": native.canonical_sha(scene.item)} for scene in scenes}
    plan = tmp_path / "unbound-plan.json"; plan.write_text(json.dumps({"sentinel1": {"selected_pair": pair}}))
    with pytest.raises(ValueError, match="checksums"):
        native.scenes_from_plan(plan)
