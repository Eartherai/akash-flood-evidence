"""Meaningful integrity, isolation and localhost tests for saved-case release."""
from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import zipfile

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


viewer = load("serve_saved_case")
exporter = load("package_saved_case")


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "fixture sources with spaces"
    root.mkdir()
    expected = []
    for source, target, _, _ in exporter.SOURCE_FILES:
        content = ("synthetic integrity fixture: " + target + "\n").encode()
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        expected.append((source, target, len(content), hashlib.sha256(content).hexdigest()))
    payload, document, digest = exporter.prepare_payload(root, tuple(expected))
    extracted = tmp_path / "relocated saved case with spaces"
    extracted.mkdir()
    for name, content in payload.items():
        path = extracted / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root, tuple(expected), payload, document, digest, extracted


def change_manifest(root, modify):
    path = root / "manifest.json"
    doc = json.loads(path.read_text())
    modify(doc)
    raw = exporter.json_bytes(doc)
    path.write_bytes(raw)
    (root / "manifest.sha256").write_text(hashlib.sha256(raw).hexdigest() + "  manifest.json\n")


def test_exact_allowlist_has_no_runtime_labels_or_checking_geometries():
    assert len(exporter.SOURCE_FILES) == 22
    assert {row[1] for row in exporter.SOURCE_FILES} == set(viewer.SOURCE_PATHS)
    assert sum(row[2] for row in exporter.SOURCE_FILES) == 15_744_793
    assert not any(any(token in row[0].lower() for token in
                   (".gguf", "water-model", "sen1floods11-cpu", "evaluation-only", ".gpkg", "unosat"))
                   for row in exporter.SOURCE_FILES)


def test_dry_run_creates_no_files_and_performs_no_network(prepared, tmp_path, monkeypatch):
    root, expected, *_ = prepared
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("No subprocess allowed"))
    output = tmp_path / "new dry run output" / "case.zip"
    result = exporter.export(root, output, expected=expected)
    assert result["status"] == "dry_run" and result["network_requests"] == 0
    assert all(result[flag] is False for flag in viewer.FALSE_FLAGS)
    assert not output.parent.exists()


def test_execute_is_deterministic_preserves_source_bytes_and_relocates(prepared, tmp_path):
    root, expected, payload, _, _, _ = prepared
    first, second = tmp_path / "a.zip", tmp_path / "b.zip"
    exporter.export(root, first, execute=True, expected=expected)
    exporter.export(root, second, execute=True, expected=expected)
    assert first.read_bytes() == second.read_bytes()
    relocated = tmp_path / "new location without original sources"
    with zipfile.ZipFile(first) as archive:
        assert len(archive.namelist()) == 27
        assert all(n.startswith(exporter.ZIP_ROOT + "/") for n in archive.namelist())
        assert all(i.date_time == exporter.ZIP_TIMESTAMP for i in archive.infolist())
        archive.extractall(relocated)
    for source, target, *_ in expected:
        assert (root / source).read_bytes() == payload[target]
    root.rename(tmp_path / "source path deliberately unavailable")
    bundle = viewer.VerifiedBundle(relocated / exporter.ZIP_ROOT)
    assert bundle.summary()["verified_file_count"] == 25
    assert bundle.read("report/report.json") == payload["report/report.json"]
    assert bundle.document["source_chain_dependencies_all_included"] is False


def test_existing_archive_and_receipt_are_never_overwritten(prepared, tmp_path):
    root, expected, *_ = prepared
    output = tmp_path / "existing.zip"
    output.write_bytes(b"prior failure must stay")
    with pytest.raises(FileExistsError):
        exporter.export(root, output, execute=True, expected=expected)
    assert output.read_bytes() == b"prior failure must stay"
    output.unlink()
    receipt = output.with_name(output.name + ".provenance.json")
    receipt.write_bytes(b"prior receipt")
    with pytest.raises(FileExistsError):
        exporter.export(root, output, execute=True, expected=expected)
    assert receipt.read_bytes() == b"prior receipt"


@pytest.mark.parametrize("kind", ["missing", "changed", "symlink", "parent_symlink"])
def test_export_rejects_unbound_source(prepared, tmp_path, kind):
    root, expected, *_ = prepared
    path = root / expected[0][0]
    if kind == "missing":
        path.unlink()
    elif kind == "changed":
        content = path.read_bytes()
        path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    elif kind == "symlink":
        copy = tmp_path / "external copy"
        copy.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(copy)
    else:
        folder = path.parent
        renamed = folder.with_name(folder.name + " external")
        folder.rename(renamed)
        folder.symlink_to(renamed, target_is_directory=True)
    with pytest.raises(ValueError):
        exporter.prepare_payload(root, expected)


def test_viewer_uses_no_subprocess_or_remote_request(prepared, monkeypatch):
    *_, extracted = prepared
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("No process allowed"))
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("No outbound requests"))
    bundle = viewer.VerifiedBundle(extracted)
    assert bundle.read("report/optical-overview.png")
    assert bundle.summary()["model_inference_run"] is False


def test_checksum_detects_manifest_edit(prepared):
    *_, extracted = prepared
    with (extracted / "manifest.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="manifest checksum"):
        viewer.VerifiedBundle(extracted)


@pytest.mark.parametrize("flag", viewer.FALSE_FLAGS)
def test_no_inference_contract_is_enforced(prepared, flag):
    *_, extracted = prepared
    change_manifest(extracted, lambda doc: doc.update({flag: True}))
    with pytest.raises(ValueError, match="non-inference"):
        viewer.VerifiedBundle(extracted)


@pytest.mark.parametrize("mutation", ["wrong_case", "missing_file", "extra_file", "duplicate", "traversal", "nonfinite", "bad_bytes"])
def test_rejects_invalid_manifest_records(prepared, mutation):
    *_, extracted = prepared
    def change(doc):
        if mutation == "wrong_case": doc["case_id"] = "another-case"
        elif mutation == "missing_file": doc["files"].pop()
        elif mutation == "extra_file": doc["files"].append(dict(doc["files"][0]))
        elif mutation == "duplicate": doc["files"][1] = dict(doc["files"][0])
        elif mutation == "traversal": doc["files"][0]["path"] = "../outside"
        elif mutation == "nonfinite": doc["files"][0]["sha256"] = "NaN"
        elif mutation == "bad_bytes": doc["files"][0]["bytes"] = True
    change_manifest(extracted, change)
    with pytest.raises(ValueError):
        viewer.VerifiedBundle(extracted)


def test_payload_tamper_and_symlink_rejected_at_startup(prepared, tmp_path):
    *_, extracted = prepared
    path = extracted / "report/report.json"
    original = path.read_bytes()
    path.write_bytes(b"X" + original[1:])
    with pytest.raises(ValueError, match="checksum"):
        viewer.VerifiedBundle(extracted)
    path.unlink()
    outside = tmp_path / "outside report"
    outside.write_bytes(original)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        viewer.VerifiedBundle(extracted)


def request(server, path, method="GET", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.request(method, path, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


@pytest.fixture
def running(prepared):
    *_, extracted = prepared
    bundle = viewer.VerifiedBundle(extracted)
    server = viewer.make_server(bundle)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield server, bundle, extracted
    server.shutdown()
    server.server_close()
    worker.join(timeout=5)


def test_http_serves_only_declared_assets_and_disables_connections(running):
    server, bundle, _ = running
    status, headers, _ = request(server, "/")
    assert status == 302 and headers["Location"] == "/report/report.html"
    for name in bundle.files:
        status, headers, content = request(server, "/" + name)
        assert status == 200 and content == bundle.read(name)
        assert "connect-src 'none'" in headers["Content-Security-Policy"]
        assert headers["Cache-Control"] == "no-store"
    status, _, content = request(server, "/report/report.html", method="HEAD")
    assert status == 200 and content == b""


@pytest.mark.parametrize("path", ["/../README.md", "/report/../../source/osm.json", "/%2e%2e/README.md", "/report%2freport.html", "/report/./report.html", "/report/report.html?query=1", "/.env", "/api/copilot", "/Users/earther/.env", "/report/", "//evil.example/report/report.html"])
def test_http_traversal_and_unlisted_paths_404(running, path):
    server, _, _ = running
    assert request(server, path)[0] == 404


def test_http_rejects_foreign_origin_and_host_and_inference_post(running):
    server, _, _ = running
    assert request(server, "/report/report.html", headers={"Origin": "https://untrusted.example"})[0] == 403
    assert request(server, "/report/report.html", headers={"Host": "untrusted.example"})[0] == 403
    assert request(server, "/api/copilot", method="POST")[0] == 405
    assert request(server, "/report/report.html", headers={"Origin": f"http://127.0.0.1:{server.server_port}"})[0] == 200


def test_http_rechecks_artifact_after_startup(running):
    server, _, extracted = running
    path = extracted / "report/report.json"
    before = path.read_bytes()
    path.write_bytes(b"X" + before[1:])
    status, _, content = request(server, "/report/report.json")
    assert status == 409 and b"checksum" in content and b"synthetic integrity" not in content


def test_standalone_viewer_runs_with_isolated_standard_library(prepared):
    *_, extracted = prepared
    result = subprocess.run([sys.executable, "-I", "-S", str(extracted / "serve_saved_case.py"),
                             "--verify-only"], cwd=extracted, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["verified_file_count"] == 25
