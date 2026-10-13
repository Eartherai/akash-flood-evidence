"""Offline setup boundaries; real native smoke is recorded separately."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import tarfile
from types import SimpleNamespace
from zipfile import ZipFile, ZipInfo

import pytest

SPEC = importlib.util.spec_from_file_location("native_setup", Path(__file__).parents[1] / "scripts/setup_native_runtime.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def pin(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../../b", "a\\..\\escape", "a//b", "a/./b", "C:/bad", "", "."])
def test_reject_unsafe_archive_member(name):
    with pytest.raises(ValueError, match="Unsafe"):
        setup.safe_member(name)


def test_normalize_official_directory_separator():
    assert str(setup.safe_member("snap\\modules\\/")) == "snap/modules"


def test_cached_assets_are_hash_checked_without_network(tmp_path):
    source = tmp_path / "source"; source.write_bytes(b"pinned asset")
    target = tmp_path / "target"
    budget = setup.DownloadBudget()
    receipt = setup.acquire(pin(b"pinned asset"), source, target, budget, False)
    assert target.read_bytes() == b"pinned asset"
    assert budget.used == 0 and budget.receipts == []
    assert receipt["reused_verified_source"] is True
    source.write_bytes(b"changed asset")
    with pytest.raises(ValueError, match="checksum/size"):
        setup.acquire(pin(b"pinned asset"), source, tmp_path / "second", budget, False)


def test_missing_asset_never_implicitly_downloads(tmp_path):
    with pytest.raises(ValueError, match="network|Missing"):
        setup.acquire(setup.DMG, None, tmp_path / "file", setup.DownloadBudget(), False)
    assert not (tmp_path / "file").exists()


def test_cache_copy_preserves_existing_destination(tmp_path):
    source = tmp_path / "source"; source.write_bytes(b"asset")
    target = tmp_path / "target"; target.write_bytes(b"previous")
    with pytest.raises(FileExistsError):
        setup.acquire(pin(b"asset"), source, target, setup.DownloadBudget(), False)
    assert target.read_bytes() == b"previous"


def test_destination_rejects_symlink_and_existing_directory(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "data/runtime").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        setup.new_destination(tmp_path, "data/runtime/file")
    with pytest.raises(FileExistsError):
        setup.new_destination(tmp_path, "outside")


def test_zip_rejects_escape_before_writing(tmp_path):
    archive = tmp_path / "vendor.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("fine.txt", "fine"); z.writestr("../escape", "bad")
    target = tmp_path / "runtime"
    with pytest.raises(ValueError, match="Unsafe"):
        setup.extract_zip(archive, target)
    assert not target.exists() and not (tmp_path / "escape").exists()


def test_zip_rejects_normalized_duplicate_and_links(tmp_path):
    archive = tmp_path / "vendor.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("snap\\x", "one"); z.writestr("snap/x", "two")
    with pytest.raises(ValueError, match="Duplicate"):
        setup.extract_zip(archive, tmp_path / "runtime")
    archive = tmp_path / "linked.zip"
    with ZipFile(archive, "w") as z:
        member = ZipInfo("link"); member.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(member, "outside")
    with pytest.raises(ValueError, match="symlink"):
        setup.extract_zip(archive, tmp_path / "runtime")


def test_zip_resource_cap(tmp_path):
    archive = tmp_path / "vendor.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("large", b"12345")
    with pytest.raises(ValueError, match="resource"):
        setup.extract_zip(archive, tmp_path / "runtime", remaining=4)


def test_jre_accepts_internal_license_link_and_executable(tmp_path):
    archive = tmp_path / "jre.tar.gz"
    with tarfile.open(archive, "w:gz") as t:
        for name, value, mode in (("Contents/Home/bin/java", b"java", 0o755), ("Contents/Home/legal/base/LICENSE", b"terms", 0o644)):
            item = tarfile.TarInfo(name); item.size = len(value); item.mode = mode
            t.addfile(item, io.BytesIO(value))
        link = tarfile.TarInfo("Contents/Home/legal/module/LICENSE"); link.type = tarfile.SYMTYPE; link.linkname = "../base/LICENSE"; t.addfile(link)
    target = tmp_path / "runtime"
    setup.extract_jre(archive, target)
    assert (target / "Contents/Home/legal/module/LICENSE").read_bytes() == b"terms"
    assert (target / "Contents/Home/bin/java").stat().st_mode & 0o111


def test_jre_rejects_external_link(tmp_path):
    archive = tmp_path / "jre.tar.gz"
    with tarfile.open(archive, "w:gz") as t:
        link = tarfile.TarInfo("Contents/bad"); link.type = tarfile.SYMTYPE; link.linkname = "../../outside"; t.addfile(link)
    with pytest.raises(ValueError, match="Unsafe|dangling"):
        setup.extract_jre(archive, tmp_path / "runtime")
    assert not (tmp_path / "outside").exists()


def test_install_preflight_refuses_existing_assets_before_mutation(tmp_path, monkeypatch):
    (tmp_path / "data/snap-runtime").mkdir(parents=True)
    marker = tmp_path / "data/snap-runtime/marker"; marker.write_text("saved runtime")
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    monkeypatch.setattr(setup, "supported", lambda: True)
    args = SimpleNamespace(root=tmp_path)
    with pytest.raises(FileExistsError):
        setup.install(args)
    assert marker.read_text() == "saved runtime"
    assert not list(tmp_path.glob(".akash-native-setup-*"))


def test_installer_rejects_unproved_platform_before_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    monkeypatch.setattr(setup, "supported", lambda: False)
    with pytest.raises(ValueError, match="macOS arm64"):
        setup.install(SimpleNamespace(root=tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_relative_contract_refuses_modified_payload_and_escape(tmp_path):
    folder = tmp_path / "data/native-setup"; folder.mkdir(parents=True)
    payload = tmp_path / "data/asset"; payload.write_bytes(b"asset")
    contract = folder / "asset-contract.json"
    value = {"schema": "akash-native-assets-v1", "files": [setup.record(payload, tmp_path)]}
    contract.write_text(json.dumps(value))
    assert setup.bound_contract(tmp_path)["files"] == value["files"]
    payload.write_bytes(b"other")
    with pytest.raises(ValueError, match="checksum/size"):
        setup.bound_contract(tmp_path)
    value["files"][0]["path"] = "../outside"; contract.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Unsafe"):
        setup.bound_contract(tmp_path)


def test_rebase_only_known_generated_assets_and_verify_bytes(tmp_path):
    path = tmp_path / "data/snap-runtime/file"; path.parent.mkdir(parents=True); path.write_bytes(b"asset")
    binding = pin(b"asset") | {"path": "/old/staging/data/snap-runtime/file"}
    result = setup.rebase_records({"vendor": binding}, tmp_path)
    assert result["vendor"]["path"] == str(path)
    assert binding["path"] == "/old/staging/data/snap-runtime/file"
    with pytest.raises(ValueError, match="Unknown generated"):
        setup.rebase_records(pin(b"asset") | {"path": "/unknown/file"}, tmp_path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum/size"):
        setup.rebase_records(binding, tmp_path)


def test_download_budget_limits():
    for cap, timeout in ((0, 10), (setup.NETWORK_CAP + 1, 10), (10, 0), (10, 1201)):
        with pytest.raises(ValueError, match="bounds"):
            setup.DownloadBudget(cap, timeout)


class FakeResponse(io.BytesIO):
    status = 200
    def __init__(self, body, spec, length=None):
        super().__init__(body); self.spec = spec; self.headers = {} if length is None else {"Content-Length": str(length)}
    def geturl(self):
        return self.spec["url"]


def mock_download(monkeypatch, body, expected=b"abc", length=None):
    spec = pin(expected) | {"url": "https://step.esa.int/pinned-fixture"}
    monkeypatch.setattr(setup, "DMG", spec)
    class Opener:
        def open(self, *args, **kwargs):
            return FakeResponse(body, spec, length)
    monkeypatch.setattr(setup, "build_opener", lambda *args: Opener())
    return spec


def test_download_verified_exact_body_no_overwrite(tmp_path, monkeypatch):
    spec = mock_download(monkeypatch, b"abc", length=3)
    budget = setup.DownloadBudget(cap=3)
    target = tmp_path / "asset"
    budget.download(spec, target)
    assert target.read_bytes() == b"abc" and budget.used == 3
    assert budget.receipts[0]["status"] == "verified"
    with pytest.raises(ValueError, match="remaining"):
        budget.download(spec, tmp_path / "second")
    with pytest.raises(FileExistsError):
        setup.DownloadBudget(cap=3).download(spec, target)


@pytest.mark.parametrize("body", [b"ab", b"abcd", b"xyz"])
def test_failed_download_retains_attempt_bytes_without_publishing(tmp_path, monkeypatch, body):
    spec = mock_download(monkeypatch, body)
    budget = setup.DownloadBudget(cap=4)
    target = tmp_path / "asset"
    with pytest.raises(ValueError):
        budget.download(spec, target)
    assert not target.exists()
    assert budget.used == len(body)
    assert budget.receipts[0]["body_bytes"] == len(body)
    assert budget.receipts[0]["status"] == "failed"
    assert target.with_name("asset.part").exists()


def test_changed_advertised_download_size_never_reads_body(tmp_path, monkeypatch):
    spec = mock_download(monkeypatch, b"abc", length=999)
    budget = setup.DownloadBudget(cap=3)
    with pytest.raises(ValueError, match="Advertised"):
        budget.download(spec, tmp_path / "asset")
    assert budget.used == 0 and budget.receipts[0]["status"] == "failed"


def test_redirect_requires_source_review():
    request = setup.Request("https://step.esa.int/pinned")
    with pytest.raises(ValueError, match="redirects"):
        setup.StrictRedirects().redirect_request(request, None, 302, "Moved", {}, "https://example.com/other")


def test_doctor_reports_missing_assets_without_starting_process(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "supported", lambda: True)
    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kwargs: pytest.fail("doctor started subprocess"))
    result = setup.doctor(tmp_path)
    assert result["status"] == "blocked" and result["blockers"]
    assert result["model_loaded"] is False and result["jvm_started"] is False and result["network_requests"] == 0


def test_mount_device_normalizes_symlink_spelling(tmp_path):
    mount_parent = tmp_path / "real"; mount_parent.mkdir()
    alias = tmp_path / "alias"; alias.symlink_to(mount_parent, target_is_directory=True)
    entities = [{"dev-entry": "/dev/disk-fixture", "mount-point": str(alias / "volume")}]
    assert setup.mounted_device(entities, mount_parent / "volume") == "/dev/disk-fixture"


def test_download_never_reads_beyond_total_body_budget(tmp_path, monkeypatch):
    spec = mock_download(monkeypatch, b"abcdef")
    budget = setup.DownloadBudget(cap=3)
    with pytest.raises(ValueError, match="budget"):
        budget.download(spec, tmp_path / "asset")
    assert budget.used == 3
    assert not (tmp_path / "asset").exists()


def test_doctor_contract_rejects_retargeted_jre_link(tmp_path):
    folder = tmp_path / "data/native-setup"; folder.mkdir(parents=True)
    payload = tmp_path / "data/snap-runtime/license"; payload.parent.mkdir(parents=True); payload.write_bytes(b"terms")
    link = payload.with_name("linked-license"); link.symlink_to("license")
    value = {"schema": "akash-native-assets-v1", "files": [setup.record(payload, tmp_path)], "symlinks": [{"path": link.relative_to(tmp_path).as_posix(), "target": "license"}]}
    (folder / "asset-contract.json").write_text(json.dumps(value))
    assert setup.bound_contract(tmp_path)["symlinks"] == value["symlinks"]
    link.unlink(); link.symlink_to("/tmp/outside")
    with pytest.raises(ValueError, match="symlink"):
        setup.bound_contract(tmp_path)
