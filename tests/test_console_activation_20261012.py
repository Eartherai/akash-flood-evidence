"""Synthetic installed records test trusted startup, never satellite inference.

Pins are replaced only inside the temporary native-setup source fixture. No
real model, JVM, provider, held-out labels or installed runtime is executed.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess

import pytest

from akash_flood import console_activation_20261012 as resolver
from akash_flood.fresh_console_20261012 import ConsoleService, DEFAULTS, checked_payload, read_json

ROOT = Path(__file__).resolve().parents[1]


def pin(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def rec(path):
    return {"path": str(path.resolve()), **pin(path.read_bytes())}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


@pytest.fixture
def installed(tmp_path, monkeypatch):
    root = (tmp_path / "source relocated with spaces").resolve(); root.mkdir()
    def put(relative, data=b"synthetic fixture asset"):
        path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        return path
    for relative in ("src/akash_flood/flood_model.py", "scripts/train_sen1_cpu.py", "uv.lock"):
        put(relative, ("fixture " + relative).encode())
    model = put("outputs/sen1-cpu-v1/water-model.joblib", b"synthetic, not a loadable model")
    protocol = put("outputs/sen1-cpu-v1/training-protocol.json", b'{"fixture_only":true}\n')
    geoid = put("data/copdem-ellipsoid/us_nga_egm08_25.tif", b"synthetic, not a raster")
    original = write(root / "outputs/sen1-cpu-v1/original-model-evaluation-freeze.json", {
        "frozen_at": "fixture-before-any-labels", "scikit_learn_version": "1.7.2",
        "python_version": "fixture-python", "threshold": .5, "holdout_labels_loaded": False,
        "reference_inputs_loaded": False, "forbidden_inputs": ["checking labels"], "gpu_used": False,
        "model": rec(model), "protocol": rec(protocol),
        "code": [rec(root / p) for p in ("src/akash_flood/flood_model.py", "scripts/train_sen1_cpu.py", "uv.lock")]})
    # Retain the real validator implementation; only fixture pin constants vary.
    setup_source = (ROOT / "scripts/setup_native_runtime.py").read_text()
    for name, path in (("CHECKPOINT", model), ("FREEZE", original), ("PROTOCOL", protocol), ("GEOID", geoid)):
        setup_source = re.sub(r"^" + name + r" = .*?$", name + " = " + repr(pin(path.read_bytes())), setup_source, flags=re.M)
    put("scripts/setup_native_runtime.py", setup_source.encode())
    runtime = root / "data/snap-runtime/portable"
    vendor = put("data/snap-runtime/portable/snap/modules/org-esa-snap-snap-dem.jar")
    for index in range(1061):
        put(f"data/snap-runtime/portable/snap/modules/fixture-{index}.jar", b"jar fixture")
    java = put("data/snap-runtime/portable/jre/Contents/Home/bin/java", b"not executed"); java.chmod(0o755)
    overlay_jar = put("data/snap-external-dem-pixel-correction/external-dem-center-overlay.jar")
    original_overlay = write(root / "data/snap-external-dem-pixel-correction/manifest.json", {"fixture_original": True})
    activation = root / "data/native-setup/activation-relocated"
    freeze = copy.deepcopy(json.loads(original.read_text())); freeze["original_freeze"] = rec(original)
    freeze_path = write(activation / "model-evaluation-freeze.json", freeze)
    overlay = {"scope": {"class": "org/esa/snap/dem/dataio/FileElevationModel.class",
        "old_method": "computeCornerBasedIndex", "new_method": "computeIndex",
        "changed_constant_count": 1, "method_bodies_unchanged": True},
        "vendor_jar": rec(vendor), "overlay_jar": rec(overlay_jar),
        "relocation": {"original_manifest": rec(original_overlay)}}
    overlay_path = write(activation / "overlay-manifest.json", overlay)
    proof = {"status": "verified on analytic DEM fixtures", "geographic_centers_preserved": True,
        "original_runtime_modified": False, "corrected_max_abs_error_m": 1e-10,
        "samples_per_jvm": 116, "overlay_manifest": rec(overlay_path),
        "test_scope": "synthetic receipt fixture, no actual proof executed"}
    proof_path = write(activation / "dem-proof/proof.json", proof)
    activation_path = write(activation / "activation.json", {"schema": "akash-native-activation-v1",
        "raw_case_replay": False, "model_freeze": rec(freeze_path),
        "overlay_manifest": rec(overlay_path), "jvm_proof": rec(proof_path)})
    assets = []
    for folder in resolver.NATIVE_DIRS:
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and not path.is_relative_to(activation):
                assets.append({**rec(path), "path": path.relative_to(root).as_posix()})
    # The contract is recorded before its own file and installation receipt exist.
    write(root / "data/native-setup/asset-contract.json", {"schema": "akash-native-assets-v1", "files": assets, "symlinks": []})
    write(root / "data/native-setup/installation-receipt.json", {"schema": "akash-native-setup-v1",
        "status": "prepared", "platform": "macOS arm64", "model_refitted": False,
        "raw_case_replay": False, "sources": {"geoid": {"source": rec(geoid)}}})
    requirements = tmp_path / "organizer requirements.txt"; requirements.write_text("Supplied organizer source text fixture\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("Startup metadata must not execute or contact a provider")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    return root, activation_path, requirements


def resolve(installed):
    root, activation, requirements = installed
    return resolver.resolve_startup(root, activation=activation, requirements=requirements)


def refresh_activation(installed, key, payload):
    _, activation, _ = installed
    value = json.loads(activation.read_text())
    path = Path(value[key]["path"]); write(path, payload)
    value[key] = rec(path); write(activation, value)


def test_relocated_metadata_derives_actual_runner_inputs_without_execution(installed):
    root, activation, requirements = installed
    flags, receipt = resolve(installed)
    assert flags == {"requirements": requirements.resolve(),
        "model-freeze": activation.parent / "model-evaluation-freeze.json",
        "dem-reader-patch": activation.parent / "overlay-manifest.json",
        "dem-reader-proof": activation.parent / "dem-proof/proof.json",
        "geoid": root / "data/copdem-ellipsoid/us_nga_egm08_25.tif"}
    assert receipt["network_requests"] == receipt["subprocesses"] == 0
    assert receipt["model_loaded"] is receipt["jvm_started"] is receipt["raw_satellite_replayed"] is False
    assert len(receipt["installed_assets"]) >= 1062


@pytest.mark.parametrize("key", ["model_freeze", "overlay_manifest", "jvm_proof"])
def test_changed_activated_metadata_is_refused(installed, key):
    _, activation, _ = installed
    value = json.loads(activation.read_text()); Path(value[key]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        resolve(installed)


def test_activation_must_not_escape_the_source_root(installed, tmp_path):
    root, activation, requirements = installed
    outside = tmp_path / "activation.json"; outside.write_bytes(activation.read_bytes())
    with pytest.raises(ValueError):
        resolver.resolve_startup(root, activation=outside, requirements=requirements)


def test_symlink_activation_is_refused(installed):
    root, activation, requirements = installed
    link = activation.parent / "activation-link.json"; link.symlink_to(activation)
    with pytest.raises(ValueError, match="symlink"):
        resolver.resolve_startup(root, activation=link, requirements=requirements)


def test_requirements_must_be_explicit_with_an_activation(installed):
    root, activation, _ = installed
    with pytest.raises(ValueError, match="requirements"):
        resolver.resolve_startup(root, activation=activation)


@pytest.mark.parametrize("value", ['{"schema":1,"schema":2}', '{"schema":NaN}', '[]'])
def test_unchecked_json_is_not_accepted(installed, value):
    _, activation, _ = installed; activation.write_text(value)
    with pytest.raises(ValueError):
        resolve(installed)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "data/native-setup/../escape", "data\\escape"])
def test_installed_inventory_rejects_escaping_paths(installed, path):
    root, _, _ = installed
    contract = root / "data/native-setup/asset-contract.json"
    value = json.loads(contract.read_text()); value["files"][0]["path"] = path; write(contract, value)
    with pytest.raises(ValueError, match="Unsafe"):
        resolve(installed)


def test_duplicate_asset_binding_is_refused(installed):
    root, _, _ = installed; path = root / "data/native-setup/asset-contract.json"
    value = json.loads(path.read_text()); value["files"].append(value["files"][0]); write(path, value)
    with pytest.raises(ValueError, match="duplicate"):
        resolve(installed)


def test_modified_native_asset_is_refused_before_a_plan(installed):
    root, _, _ = installed
    (root / "data/snap-runtime/portable/snap/modules/fixture-0.jar").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        resolve(installed)


def test_unrecorded_native_runtime_file_is_refused(installed):
    root, _, _ = installed
    (root / "data/snap-runtime/portable/jre/Contents/Home/bin/unrecorded").write_bytes(b"not in installation")
    with pytest.raises(ValueError, match="unrecorded"):
        resolve(installed)


def test_frozen_inference_threshold_cannot_be_changed(installed):
    _, activation, _ = installed
    value = json.loads((activation.parent / "model-evaluation-freeze.json").read_text()); value["threshold"] = .6
    refresh_activation(installed, "model_freeze", value)
    with pytest.raises(ValueError, match="inference contract"):
        resolve(installed)


@pytest.mark.parametrize("key,value", [("samples_per_jvm", 0), ("geographic_centers_preserved", False), ("corrected_max_abs_error_m", .01)])
def test_unproved_dem_activation_is_refused(installed, key, value):
    _, activation, _ = installed
    proof = json.loads((activation.parent / "dem-proof/proof.json").read_text()); proof[key] = value
    refresh_activation(installed, "jvm_proof", proof)
    with pytest.raises(ValueError, match="proof"):
        resolve(installed)


def test_console_forwards_startup_bindings_and_freezes_their_sources(installed, monkeypatch):
    root, activation, requirements = installed
    runner = root / "scripts/fixture_planning_runner.py"
    runner.write_text('''import argparse, json, hashlib
from pathlib import Path
def rec(p):
 p=Path(p).resolve();return {"path":str(p),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
def parser():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path)
 for key in ('model-freeze','dem-reader-patch','dem-reader-proof','geoid','requirements'):p.add_argument('--'+key,type=Path)
 p.parse_args=lambda argv:p.parse_known_args(argv)[0]
 return p
def build_plan(args):
 return {"out":str(args.out),"source_runner":rec(__file__),"external_inputs":{key:rec(getattr(args,key)) for key in ('model_freeze','dem_reader_patch','dem_reader_proof','geoid','requirements')},"blockers":[],"stages":[],"full_track_b_requirements_satisfied":False}
''')
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "1.7.2")
    service = ConsoleService(output=root / "outputs/console-plan-fixture", runners={"water": runner},
        activation=activation, requirements=requirements, source_root=root)
    try:
        job = service.make_plan(copy.deepcopy(DEFAULTS))
        plan = read_json(job["console_plan"]["path"])
        assert plan["external_inputs"]["model_freeze"]["path"] == str(activation.parent / "model-evaluation-freeze.json")
        assert plan["external_inputs"]["requirements"]["path"] == str(requirements.resolve())
        assert plan["external_inputs"]["console_native_startup"]["activation"] == rec(activation)
        assert job["plan_network_requests"] == job["plan_subprocesses_run"] == job["run_attempts"] == 0
        assert not Path(job["output_path"]).exists()
        assert service.capabilities()["native_startup"]["activation_supplied"] is True
    finally:
        service.close()


def test_relocated_plan_uses_the_real_base_runner_parser_without_execution(installed, monkeypatch):
    root, activation, requirements = installed
    # Source files only, about a few MB; fixture assets are explicitly synthetic.
    # Existing fixture pins/metadata stay intact and no binary/runtime is copied.
    for folder in ("scripts", "src"):
        for source in (ROOT / folder).rglob("*.py"):
            target = root / source.relative_to(ROOT)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "1.7.2")
    service = ConsoleService(output=root / "outputs/real-base-parser-plan-only",
        runners={"water": root / "scripts/run_fresh_case.py"}, activation=activation,
        requirements=requirements, source_root=root)
    try:
        job = service.make_plan(copy.deepcopy(DEFAULTS))
        plan = read_json(job["console_plan"]["path"])
        assert plan["blockers"] == []
        for key, path in {"model_freeze": activation.parent / "model-evaluation-freeze.json",
                "dem_reader_patch": activation.parent / "overlay-manifest.json",
                "dem_reader_proof": activation.parent / "dem-proof/proof.json",
                "requirements": requirements.resolve(),
                "geoid": root / "data/copdem-ellipsoid/us_nga_egm08_25.tif"}.items():
            assert plan["external_inputs"][key] == rec(path)
        stages = {stage["name"]: stage for stage in plan["stages"]}
        assert str(activation.parent / "model-evaluation-freeze.json") in stages["infer_water"]["argv"]
        assert str(activation.parent / "overlay-manifest.json") in stages["process_s1"]["argv"]
        assert str(requirements.resolve()) in stages["plan_case"]["argv"]
        assert plan["execution"]["subprocesses_run"] == plan["execution"]["network_requests_made_by_launcher"] == 0
        assert job["run_attempts"] == 0 and not Path(job["output_path"]).exists()
    finally:
        service.close()


@pytest.mark.parametrize("field", ["activation", "requirements", "model_freeze", "geoid"])
def test_browser_cannot_supply_filesystem_prerequisites(field):
    with pytest.raises(ValueError):
        checked_payload(copy.deepcopy(DEFAULTS) | {field: "/tmp/untrusted"})
