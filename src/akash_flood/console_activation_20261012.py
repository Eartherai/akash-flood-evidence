"""Resolve proved native prerequisites for trusted console startup only.

This performs checksum and metadata checks. It never starts Java, loads a
checkpoint, executes a model, creates an activation or contacts a provider.
Browser requests cannot select any of these filesystem paths.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re

JSON_CAP = 16_000_000
RECORD_LIMIT = 20_000
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
NATIVE_DIRS = ("data/snap-runtime", "data/copdem-ellipsoid",
               "data/snap-external-dem-pixel-correction", "data/native-setup",
               "outputs/sen1-cpu-v1")


def _regular(path, root=None):
    path = Path(path).absolute()
    boundary = Path(root).resolve(strict=True) if root else None
    if boundary:
        path.relative_to(boundary)
    cursor = path
    while cursor != (boundary or cursor.parent):
        if cursor.is_symlink():
            raise ValueError("Native startup paths cannot contain symlinks")
        cursor = cursor.parent
    if path.is_symlink() or not path.is_file():
        raise ValueError("Native startup requires an explicit regular file")
    if boundary and not path.resolve(strict=True).is_relative_to(boundary):
        raise ValueError("Native startup path leaves the installed source root")
    return path.resolve(strict=True)


def _record(path, root=None):
    path = _regular(path, root)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate native startup JSON key")
        result[key] = value
    return result


def _json(path, root):
    path = _regular(path, root)
    if path.stat().st_size > JSON_CAP:
        raise ValueError("Native startup metadata exceeds its byte cap")
    value = json.loads(path.read_text(), object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite native startup JSON")))
    if not isinstance(value, dict):
        raise ValueError("Native startup metadata must be a JSON object")
    return value


def _verify(item, root, expected_path=None):
    if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
            or type(item.get("bytes")) is not int or item["bytes"] < 0
            or not isinstance(item.get("sha256"), str) or not SHA256.fullmatch(item["sha256"])):
        raise ValueError("Malformed native startup file binding")
    path = Path(item["path"])
    if not path.is_absolute():
        raise ValueError("Activation bindings must use current absolute paths")
    if expected_path is not None and path.absolute() != Path(expected_path).absolute():
        raise ValueError("Native startup binding is not at its canonical installed path")
    actual = _record(path, root)
    if any(actual[key] != item[key] for key in ("path", "bytes", "sha256")):
        raise ValueError("Native startup checksum or size mismatch")
    return actual


def _records(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            yield value
        for child in value.values():
            yield from _records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _records(child)


def _native_module(root):
    path = _regular(root / "scripts/setup_native_runtime.py", root)
    spec = importlib.util.spec_from_file_location("akash_console_native_setup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_startup(root, *, activation=None, requirements=None):
    """Return runner flags and source-bound receipt, with no raw execution."""
    root = Path(root).resolve(strict=True)
    inputs, receipt = {}, {"schema": "akash-console-native-startup-v1",
        "network_requests": 0, "subprocesses": 0, "model_loaded": False,
        "jvm_started": False, "raw_satellite_replayed": False,
        "full_track_b_requirements_satisfied": False}
    if requirements is not None:
        requirements_record = _record(requirements)
        if not 1 <= requirements_record["bytes"] <= 1_000_000:
            raise ValueError("Supply a nonempty bounded organizer requirements text")
        Path(requirements_record["path"]).read_text(encoding="utf-8")
        inputs["requirements"] = Path(requirements_record["path"])
        receipt["requirements"] = requirements_record
    if activation is None:
        return inputs, receipt if inputs else None
    if requirements is None:
        raise ValueError("Supply --requirements with --activation; no organizer text is guessed")

    activation_path = _regular(activation, root)
    activation_path.relative_to(root / "data/native-setup")
    if activation_path.name != "activation.json":
        raise ValueError("Supply the setup tier's explicit activation.json")
    active = _json(activation_path, root)
    if active.get("schema") != "akash-native-activation-v1" or active.get("raw_case_replay") is not False:
        raise ValueError("Unsupported native activation scope")
    native = _native_module(root)
    contract_path = root / "data/native-setup/asset-contract.json"
    install_path = root / "data/native-setup/installation-receipt.json"
    contract, installed = _json(contract_path, root), _json(install_path, root)
    if (contract.get("schema") != "akash-native-assets-v1"
            or installed.get("schema") != "akash-native-setup-v1"
            or installed.get("status") != "prepared"
            or installed.get("platform") != "macOS arm64"
            or installed.get("model_refitted") is not False
            or installed.get("raw_case_replay") is not False):
        raise ValueError("Missing prepared native installation contract")
    files = contract.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= RECORD_LIMIT:
        raise ValueError("Invalid installed native asset inventory")
    asset_records, by_path = [], {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Malformed installed native asset path")
        relative = PurePosixPath(item["path"])
        if (relative.is_absolute() or not relative.parts or ".." in relative.parts
                or "\\" in item["path"] or str(relative) != item["path"]
                or not any(item["path"].startswith(folder + "/") for folder in NATIVE_DIRS)
                or item["path"] in by_path):
            raise ValueError("Unsafe or duplicate installed native asset path")
        actual = _verify({**item, "path": str(root.joinpath(*relative.parts))}, root)
        asset_records.append(actual); by_path[item["path"]] = actual
    # This existing checker also rejects changed or escaping JRE symlinks.
    # Regular file checks above add strict JSON/path validation to that contract.
    native.bound_contract(root)
    # New activation/proof receipts are intentionally outside the install-time
    # inventory. Runtime, auxiliaries, overlay and model directories are not:
    # reject an unrecorded extra file that could affect future native execution.
    payload_dirs = tuple(folder for folder in NATIVE_DIRS if folder != "data/native-setup")
    declared_payloads = {path for path in by_path
        if any(path.startswith(folder + "/") for folder in payload_dirs)}
    actual_payloads = {path.relative_to(root).as_posix() for folder in payload_dirs
        for path in (root / folder).rglob("*") if path.is_file() and not path.is_symlink()}
    if actual_payloads != declared_payloads:
        raise ValueError("Installed native payload inventory contains missing or unrecorded files")
    runtime = root / "data/snap-runtime/portable"
    if runtime.is_symlink() or len(list(runtime.rglob("*.jar"))) != 1062:
        raise ValueError("Only the checked 1062-jar prepared portable SNAP layout is supported")
    if not os.access(runtime / "jre/Contents/Home/bin/java", os.X_OK):
        raise ValueError("Prepared native Java executable is missing or not executable")
    geoid = root / "data/copdem-ellipsoid/us_nga_egm08_25.tif"
    geoid_record = by_path.get(geoid.relative_to(root).as_posix())
    source_geoid = installed.get("sources", {}).get("geoid", {}).get("source", {})
    if not geoid_record or any(geoid_record[k] != native.GEOID[k] or source_geoid.get(k) != native.GEOID[k]
                              for k in ("bytes", "sha256")):
        raise ValueError("The installed EGM2008 geoid differs from the pinned native setup source")

    paths = {"model_freeze": activation_path.parent / "model-evaluation-freeze.json",
        "overlay_manifest": activation_path.parent / "overlay-manifest.json",
        "jvm_proof": activation_path.parent / "dem-proof/proof.json"}
    bindings = {key: _verify(active.get(key), root, path) for key, path in paths.items()}
    freeze, overlay, proof = (_json(paths[key], root) for key in paths)
    original = root / "outputs/sen1-cpu-v1/original-model-evaluation-freeze.json"
    model = root / "outputs/sen1-cpu-v1/water-model.joblib"
    protocol = root / "outputs/sen1-cpu-v1/training-protocol.json"
    original_value = native.validate_model_sources(model, original, protocol, root)
    model_bindings = [_verify(freeze.get("model"), root, model),
        _verify(freeze.get("protocol"), root, protocol),
        _verify(freeze.get("original_freeze"), root, original)]
    codes = freeze.get("code")
    code_paths = ("src/akash_flood/flood_model.py", "scripts/train_sen1_cpu.py", "uv.lock")
    if not isinstance(codes, list) or len(codes) != len(code_paths):
        raise ValueError("Activation must retain the frozen three source bindings")
    model_bindings += [_verify(item, root, root / relative) for item, relative in zip(codes, code_paths)]
    if any(freeze.get(key) != original_value[key] for key in
            ("frozen_at", "scikit_learn_version", "python_version", "threshold",
             "holdout_labels_loaded", "reference_inputs_loaded", "forbidden_inputs", "gpu_used")):
        raise ValueError("Activation changes the frozen inference contract")
    overlay_records = [_verify(item, root) for item in _records(overlay)]
    scope = {"class": "org/esa/snap/dem/dataio/FileElevationModel.class",
        "old_method": "computeCornerBasedIndex", "new_method": "computeIndex",
        "changed_constant_count": 1, "method_bodies_unchanged": True}
    if any(overlay.get("scope", {}).get(k) != v for k, v in scope.items()):
        raise ValueError("Unexpected external DEM sampler correction scope")
    _verify(overlay.get("vendor_jar"), root, runtime / "snap/modules/org-esa-snap-snap-dem.jar")
    _verify(overlay.get("overlay_jar"), root,
            root / "data/snap-external-dem-pixel-correction/external-dem-center-overlay.jar")
    _verify(proof.get("overlay_manifest"), root, paths["overlay_manifest"])
    error = proof.get("corrected_max_abs_error_m")
    if (proof.get("status") != "verified on analytic DEM fixtures"
            or proof.get("geographic_centers_preserved") is not True
            or proof.get("original_runtime_modified") is not False
            or isinstance(error, bool) or not isinstance(error, (int, float))
            or not math.isfinite(error) or not 0 <= error < 1e-6
            or type(proof.get("samples_per_jvm")) is not int or proof["samples_per_jvm"] < 100):
        raise ValueError("Missing or incompatible independent DEM sampler proof")
    proof_records = [_verify(item, root) for item in _records(proof)]
    inputs.update({"model-freeze": paths["model_freeze"], "dem-reader-patch": paths["overlay_manifest"],
                   "dem-reader-proof": paths["jvm_proof"], "geoid": geoid})
    receipt.update(activation=_record(activation_path, root), asset_contract=_record(contract_path, root),
        installation_receipt=_record(install_path, root), native_setup_source=_record(root / "scripts/setup_native_runtime.py", root),
        resolver_source=_record(Path(__file__)), activated_bindings=bindings,
        model_bindings=model_bindings, overlay_bindings=overlay_records,
        proof_bindings=proof_records, installed_assets=asset_records, geoid=geoid_record,
        runtime_scope="Existing prepared macOS arm64 portable SNAP layout only")
    return inputs, receipt
