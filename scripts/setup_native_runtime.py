"""Explicit SNAP 14 macOS arm64 bootstrap and frozen-model environment doctor.

Source distribution contains no weights, vendor binary or upstream Java source.
Network is opt-in and restricted to pinned provider assets. Original scientific
receipts are preserved; activation creates new, visibly derived path bindings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import plistlib
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.parse import urlsplit
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
DMG = {"url": "https://download.esa.int/step/snap/14.0/installers/esa-snap_sentinel_macos_arm-14.0.0.dmg", "bytes": 1125044359, "sha256": "eb4f70e1399d6eefbae0972dffd8719e9cc001ce78c8dbd7ae6447776e4d2c43"}
GEOID = {"url": "https://cdn.proj.org/us_nga_egm08_25.tif", "bytes": 80585622, "sha256": "4191d471eefebf24091b56dbc604353cb3b8cf8cc70e448bb9ae56a272bef17a"}
EGM96 = {"url": "https://step.esa.int/auxdata/dem/egm96/ww15mgh_b.zip", "bytes": 2605769, "sha256": "b5db7a14c3062decb8daaaf60f257ff374084be79aeda9ce333dd5b9c815751c"}
DEM_SOURCE = {"url": "https://raw.githubusercontent.com/senbox-org/snap-engine/059c4901869eacd2136590a1c21530abbd31d2f3/snap-dem/src/main/java/org/esa/snap/dem/dataio/FileElevationModel.java", "bytes": 6826, "sha256": "99b2073ce87174c6a8af8cabd2c9825d9c923c4721432e66413752a0c5797aaf"}
CHECKPOINT = {"bytes": 54300, "sha256": "8b8b5795aa890f01fa611d4bd46ed0b97b9e8b5974ebd6dfed5e27e0f8d54d2d"}
FREEZE = {"bytes": 2371, "sha256": "786b7a35555fe46ca9168db677957d53e91a0f60a1557673b94b939c94c1fa12"}
PROTOCOL = {"bytes": 83237, "sha256": "57dc4cc412d09ef06889a29d6f3a7956d1c5666dbf57c58750b77575e3b749d2"}
PAYLOADS = {"76.dat": "7905b44f7ef29f8145cd3906ed3fffdf2d32f75b5bc85cc5fa1018c5651d5ee4", "68.dat": "1deb0c86973920e69fad86a82aa1a64b1afeebad02e88e43497037ce1791d3c1", "73.dat": "779cf0995fa187962a364b2938c00345cadd9fb47e41ce8fb0a44b1496e08a22", "jre.tar.gz": "0d8a22d3b20b6b5ad2e36344728304f9207313548ef473e0fc62d2072135e72c"}
NETWORK_CAP = 1_210_000_000
EXTRACTION_CAP = 1_500_000_000
INSTALL_DIRS = ("data/snap-runtime", "data/copdem-ellipsoid", "data/snap-external-dem-pixel-correction", "data/native-setup", "outputs/sen1-cpu-v1")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path, root=None):
    path = Path(path).resolve(strict=True)
    return {"path": path.relative_to(root).as_posix() if root else str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Require an explicit regular asset file: " + str(path))
    if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
        raise ValueError("Pinned asset checksum/size mismatch: " + str(path))
    return path.resolve()


def supported():
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def new_destination(root, relative):
    root = Path(root).resolve(strict=True)
    relative = safe_member(relative)
    target = root.joinpath(*relative.parts)
    if any(p.is_symlink() for p in (target, *target.parents) if p != root.parent):
        raise ValueError("Refuse symlink installation destination")
    if target.exists():
        raise FileExistsError("Preserve prior runtime/evidence; use a fresh source extraction: " + str(target))
    return target


def safe_member(name):
    # Official Install4j ZIPs use backslashes only in directory entries.
    name = name.replace("\\", "/")
    path = PurePosixPath(name.rstrip("/"))
    if not path.parts or path.is_absolute() or ":" in name or any(p in {"", ".", ".."} for p in path.parts) or str(path) != name.rstrip("/"):
        raise ValueError("Unsafe archive member: " + name)
    return path


class StrictRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        if newurl != request.full_url:
            raise ValueError("Asset redirects require explicit source review")
        return super().redirect_request(request, fp, code, message, headers, newurl)


class DownloadBudget:
    def __init__(self, cap=NETWORK_CAP, timeout=1200):
        if not 1 <= cap <= NETWORK_CAP or not 1 <= timeout <= 1200:
            raise ValueError("Network budget/timeout exceeds supported bounds")
        self.cap, self.timeout, self.used = cap, timeout, 0
        self.receipts = []

    def download(self, spec, target):
        if spec not in (DMG, GEOID, EGM96, DEM_SOURCE):
            raise ValueError("Unreviewed download source")
        if spec["bytes"] > self.cap - self.used:
            raise ValueError("Pinned file cannot fit remaining network budget")
        target = Path(target)
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        start, size = time.monotonic(), 0
        item = {"url": spec["url"], "started_at": now(), "body_bytes": 0, "status": "started"}
        self.receipts.append(item)
        try:
            opener = build_opener(StrictRedirects())
            with opener.open(Request(spec["url"], headers={"Accept-Encoding": "identity"}), timeout=30) as response, partial.open("xb") as stream:
                if response.status != 200 or response.geturl() != spec["url"]:
                    raise ValueError("Require exact reviewed HTTPS source and complete response")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != spec["bytes"]:
                    raise ValueError("Advertised asset size changed")
                while True:
                    if time.monotonic() - start > self.timeout:
                        raise TimeoutError("Asset transfer deadline exceeded")
                    # The extra byte detects oversized bodies without unbounded reads.
                    remaining = self.cap - self.used
                    if remaining <= 0:
                        if size == spec["bytes"] and length is not None:
                            break
                        raise ValueError("Asset transfer exhausted the body budget before EOF validation")
                    chunk = response.read(min(1024 * 1024, spec["bytes"] - size + 1, remaining))
                    if not chunk:
                        break
                    size += len(chunk); self.used += len(chunk); item["body_bytes"] = size
                    if size > spec["bytes"] or self.used > self.cap:
                        raise ValueError("Asset transfer body exceeds pinned size or budget")
                    stream.write(chunk)
            verify(partial, spec)
            os.link(partial, target)  # Atomic no overwrite.
            partial.unlink()
            item.update(status="verified", elapsed_seconds=time.monotonic() - start)
            return target
        except Exception as exc:
            item.update(status="failed", error=type(exc).__name__, elapsed_seconds=time.monotonic() - start)
            raise


def acquire(spec, supplied, target, budget, allow_download):
    if supplied is not None:
        source = verify(supplied, spec)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        verify(target, spec)
        return {"source": record(source), "network_bytes": 0, "reused_verified_source": True}
    if not allow_download:
        raise ValueError("Missing pinned asset; supply its cache path or explicitly use --allow-download")
    budget.download(spec, target)
    return {"source_url": spec["url"], "network_bytes": spec["bytes"], "reused_verified_source": False}


def extract_zip(archive, target, remaining=EXTRACTION_CAP):
    with ZipFile(archive) as zipped:
        members = zipped.infolist()
        if len(members) > 10000 or sum(m.file_size for m in members) > remaining:
            raise ValueError("Vendor ZIP extraction exceeds resource bound")
        names = set()
        for member in members:
            path = safe_member(member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("ZIP symlink is not supported")
            if member.is_dir():
                continue
            if path in names:
                raise ValueError("Duplicate normalized ZIP destination")
            names.add(path)
        written = 0
        for member in members:
            path = target.joinpath(*safe_member(member.filename).parts)
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as src, path.open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            os.chmod(path, 0o755 if (member.external_attr >> 16) & 0o111 else 0o644)
            written += member.file_size
        return written


def extract_jre(archive, target, remaining=EXTRACTION_CAP):
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        names = {safe_member(m.name).as_posix(): m for m in members}
        if len(members) != len(names) or len(members) > 10000 or sum(m.size for m in members) > remaining:
            raise ValueError("JRE archive exceeds resource bound or duplicates a member")
        for member in members:
            path = target.joinpath(*safe_member(member.name).parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, path.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                os.chmod(path, member.mode & 0o777)
            elif not member.issym():
                raise ValueError("Unsupported JRE archive member type")
        for member in members:
            if not member.issym():
                continue
            path = target.joinpath(*safe_member(member.name).parts)
            link = (path.parent / member.linkname).resolve()
            if Path(member.linkname).is_absolute() or not link.is_relative_to(target.resolve()) or not link.is_file():
                raise ValueError("Unsafe or dangling JRE symlink")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(member.linkname)
        return sum(m.size for m in members)


def mounted_device(entities, mount):
    return next(e["dev-entry"] for e in entities if e.get("mount-point") and Path(e["mount-point"]).resolve() == Path(mount).resolve())


def extract_dmg(dmg, runtime):
    if not supported():
        raise ValueError("Only native macOS arm64 is supported; Linux/Windows/Intel are unproved")
    verify(dmg, DMG)
    runtime.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="akash-snap-mount-") as mount_dir:
        mount = Path(mount_dir).resolve() / "volume"
        process = subprocess.run(["hdiutil", "attach", "-readonly", "-nobrowse", "-noautoopen", "-mountpoint", str(mount), "-plist", str(dmg)], capture_output=True, timeout=120)
        if process.returncode:
            raise RuntimeError("Read-only DMG mount failed: " + process.stderr.decode(errors="replace")[-1000:])
        entities = plistlib.loads(process.stdout)["system-entities"]
        device = str(mount)
        try:
            device = mounted_device(entities, mount)
            base = mount / "ESA SNAP Installer.app/Contents/Resources/app"
            extracted = 0
            for name, digest in PAYLOADS.items():
                if sha(base / name) != digest:
                    raise ValueError("Pinned official payload changed: " + name)
                if name.endswith(".dat"):
                    extracted += extract_zip(base / name, runtime, EXTRACTION_CAP - extracted)
                else:
                    extracted += extract_jre(base / name, runtime / "jre", EXTRACTION_CAP - extracted)
            if len(list(runtime.rglob("*.jar"))) != 1062:
                raise ValueError("Prepared SNAP jar count differs from reviewed runtime")
            return {"extracted_regular_bytes": extracted, "runtime_jars": 1062, "installer_started": False, "global_installation": False, "license_agreement_submitted": False}
        finally:
            result = subprocess.run(["hdiutil", "detach", device], capture_output=True, timeout=60)
            if result.returncode:
                raise RuntimeError("DMG unmount failed; manual cleanup needed for " + device)


def validate_model_sources(checkpoint, freeze, protocol, root):
    verify(checkpoint, CHECKPOINT); verify(freeze, FREEZE); verify(protocol, PROTOCOL)
    value = json.loads(Path(freeze).read_text())
    if value["model"]["sha256"] != CHECKPOINT["sha256"] or value["protocol"]["sha256"] != PROTOCOL["sha256"] or value["threshold"] != .5 or value["scikit_learn_version"] != "1.7.2":
        raise ValueError("Frozen model contract changed")
    paths = ("src/akash_flood/flood_model.py", "scripts/train_sen1_cpu.py", "uv.lock")
    if len(value["code"]) != len(paths):
        raise ValueError("Unexpected frozen code bindings")
    for relative, binding in zip(paths, value["code"]):
        verify(root / relative, binding)
    return value


def install(args):
    root = Path(args.root).resolve(strict=True)
    if root != ROOT:
        raise ValueError("Run the setup script from the target source extraction; --root must match its own source root")
    if not supported():
        raise ValueError("Only native macOS arm64 setup is supported")
    targets = {rel: new_destination(root, rel) for rel in INSTALL_DIRS}
    for name, spec in (("dmg", DMG), ("geoid", GEOID), ("egm96", EGM96), ("dem_source", DEM_SOURCE)):
        source = getattr(args, name)
        if source:
            verify(source, spec)
        elif not args.allow_download:
            raise ValueError("Missing --" + name.replace("_", "-") + "; network is disabled")
    validate_model_sources(args.checkpoint, args.freeze, args.protocol, root)
    if shutil.disk_usage(root).free < 4_000_000_000:
        raise ValueError("Setup needs at least 4 GB free disk for pinned assets and staging")
    budget = DownloadBudget(args.network_cap, args.download_timeout)
    stage = Path(tempfile.mkdtemp(prefix=".akash-native-setup-", dir=root))
    receipt = {"schema": "akash-native-setup-v1", "started_at": now(), "status": "preparing", "platform": "macOS arm64", "gpu_used": False, "raw_case_replay": False, "hazard_accuracy_established": False, "model_refitted": False, "sources": {}, "network_cap_bytes": budget.cap}
    try:
        runtime_dir = stage / "data/snap-runtime"; runtime_dir.mkdir(parents=True)
        receipt["sources"]["dmg"] = acquire(DMG, args.dmg, runtime_dir / "esa-snap_sentinel_macos_arm-14.0.0.dmg", budget, args.allow_download)
        receipt["extraction"] = extract_dmg(runtime_dir / "esa-snap_sentinel_macos_arm-14.0.0.dmg", runtime_dir / "portable")
        receipt["sources"]["egm96"] = acquire(EGM96, args.egm96, runtime_dir / "user-cache/auxdata/dem/egm96/ww15mgh_b.zip", budget, args.allow_download)
        receipt["sources"]["geoid"] = acquire(GEOID, args.geoid, stage / "data/copdem-ellipsoid/us_nga_egm08_25.tif", budget, args.allow_download)
        meta = stage / "data/native-setup"; meta.mkdir(parents=True)
        receipt["sources"]["dem_source"] = acquire(DEM_SOURCE, args.dem_source, meta / "FileElevationModel.vendor.java", budget, args.allow_download)
        model_dir = stage / "outputs/sen1-cpu-v1"; model_dir.mkdir(parents=True)
        for name, source, spec in (("water-model.joblib", args.checkpoint, CHECKPOINT), ("original-model-evaluation-freeze.json", args.freeze, FREEZE), ("training-protocol.json", args.protocol, PROTOCOL)):
            acquire(spec, source, model_dir / name, budget, False)
        # Build against staged vendor bytes, then derive final-path bindings below.
        import prepare_snap_external_dem as patch
        patch.build(runtime_dir / "portable/snap/modules/org-esa-snap-snap-dem.jar", runtime_dir / "portable/LICENSE.txt", meta / "FileElevationModel.vendor.java", stage / "data/snap-external-dem-pixel-correction")
        files = [record(p, stage) for rel in INSTALL_DIRS for p in sorted((stage / rel).rglob("*")) if p.is_file() and not p.is_symlink()]
        links = [{"path": p.relative_to(stage).as_posix(), "target": os.readlink(p)} for rel in INSTALL_DIRS for p in sorted((stage / rel).rglob("*")) if p.is_symlink()]
        write_json(meta / "asset-contract.json", {"schema": "akash-native-assets-v1", "files": files, "symlinks": links, "source_asset_rights": "Separately supplied checkpoint/protocol; no redistribution grant asserted"})
        receipt.update(status="prepared", network_body_bytes=budget.used, network_requests=budget.receipts)
        write_json(meta / "installation-receipt.json", receipt)
        # All targets were checked before mutation. Failed stages are retained.
        for relative, target in targets.items():
            new_destination(root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            (stage / relative).rename(target)
        shutil.rmtree(stage)  # Only this newly created staging directory remains.
        return activate(root, root / "data/native-setup/activation-1", with_proof=args.prove)
    except Exception as exc:
        failure = stage / "failed-installation.json"
        if stage.exists() and not failure.exists():
            receipt.update(status="failed", error=str(exc), network_body_bytes=budget.used, network_requests=budget.receipts)
            write_json(failure, receipt)
        raise


def bound_contract(root):
    contract_path = root / "data/native-setup/asset-contract.json"
    value = json.loads(contract_path.read_text())
    if value.get("schema") != "akash-native-assets-v1":
        raise ValueError("Unknown installed asset contract")
    for item in value["files"]:
        path = root.joinpath(*safe_member(item["path"]).parts)
        verify(path, item)
    expected_links = {item["path"]: item["target"] for item in value.get("symlinks", [])}
    actual_links = {p.relative_to(root).as_posix(): os.readlink(p) for rel in INSTALL_DIRS for p in (root / rel).rglob("*") if p.is_symlink()}
    if actual_links != expected_links:
        raise ValueError("Installed JRE symlink contract differs")
    for relative, target in expected_links.items():
        path = root.joinpath(*safe_member(relative).parts)
        if Path(target).is_absolute() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError("Installed JRE link escapes the source root")
    return value


def rebase_records(value, root):
    """Only original setup-stage records are rebased; bytes/hashes stay bound."""
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            old = Path(value["path"])
            anchors = [(root / rel, tuple(Path(rel).parts)) for rel in INSTALL_DIRS]
            for target, parts in anchors:
                for i in range(len(old.parts) - len(parts) + 1):
                    if old.parts[i:i + len(parts)] == parts:
                        path = target.joinpath(*old.parts[i + len(parts):])
                        verify(path, value)
                        return {**value, "path": str(path.resolve())}
            if old.name == "setup_native_runtime.py" or old.name == "prepare_snap_external_dem.py":
                path = root / "scripts" / old.name
                verify(path, value)
                return {**value, "path": str(path)}
            raise ValueError("Unknown generated record path; no guessed relocation: " + str(old))
        return {key: rebase_records(child, root) for key, child in value.items()}
    if isinstance(value, list):
        return [rebase_records(child, root) for child in value]
    return value


def activate(root, out, *, with_proof=False):
    root, out = Path(root).resolve(strict=True), Path(out).absolute()
    bound_contract(root)
    if out.exists() or out.is_symlink() or any(p.is_symlink() for p in out.parents):
        raise FileExistsError("Activation requires a new non-symlink directory")
    original = root / "outputs/sen1-cpu-v1/original-model-evaluation-freeze.json"
    model = root / "outputs/sen1-cpu-v1/water-model.joblib"
    protocol = root / "outputs/sen1-cpu-v1/training-protocol.json"
    freeze = validate_model_sources(model, original, protocol, root)
    out.mkdir(parents=True)
    derived = {key: freeze[key] for key in ("frozen_at", "scikit_learn_version", "python_version", "threshold", "holdout_labels_loaded", "reference_inputs_loaded", "forbidden_inputs", "gpu_used")}
    derived.update(model=record(model), protocol=record(protocol), code=[record(root / p) for p in ("src/akash_flood/flood_model.py", "scripts/train_sen1_cpu.py", "uv.lock")], original_freeze=record(original), relocation={"derived_at": now(), "scope": "New path bindings only; original freeze preserved byte for byte", "model_or_threshold_changed": False, "training_report_and_dataset_ledger": "Historical bindings remain inside original freeze; absent from this minimal runtime tier"})
    write_json(out / "model-evaluation-freeze.json", derived)
    original_overlay = root / "data/snap-external-dem-pixel-correction/manifest.json"
    overlay = rebase_records(json.loads(original_overlay.read_text()), root)
    overlay["relocation"] = {"derived_at": now(), "original_manifest": record(original_overlay), "bytes_changed_in_overlay_jar": False}
    write_json(out / "overlay-manifest.json", overlay)
    result = {"schema": "akash-native-activation-v1", "created_at": now(), "model_freeze": record(out / "model-evaluation-freeze.json"), "overlay_manifest": record(out / "overlay-manifest.json"), "jvm_proof": None, "raw_case_replay": False}
    if with_proof:
        command = [sys.executable, str(root / "scripts/prove_snap_external_dem.py"), "--overlay-manifest", str(out / "overlay-manifest.json"), "--out", str(out / "dem-proof")]
        run_logged(command, out / "dem-proof-launch.log", root, 180)
        result["jvm_proof"] = record(out / "dem-proof/proof.json")
    write_json(out / "activation.json", result)
    return result


def run_logged(command, log, root, timeout):
    with Path(log).open("x") as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=root, timeout=timeout)
    if process.returncode:
        raise RuntimeError("Command failed; preserved log " + str(log))


def doctor(root, activation=None):
    root = Path(root).resolve(strict=True)
    checks, blockers = {}, []
    checks["platform_supported"] = supported()
    if not supported():
        blockers.append("Supported native runtime is macOS arm64 only")
    checks["python_version"] = platform.python_version()
    for package in ("numpy", "scipy", "rasterio", "shapely", "pyproj", "scikit-learn", "jpype1"):
        try:
            checks[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            checks[package] = None
            blockers.append("Missing Python dependency: " + package)
    for package, version in (("scikit-learn", "1.7.2"), ("jpype1", "1.6.0")):
        if checks[package] and checks[package] != version:
            blockers.append(package + " must be " + version)
    try:
        checks["bound_regular_asset_files"] = len(bound_contract(root)["files"])
        runtime = root / "data/snap-runtime/portable"
        checks["runtime_jars"] = len(list(runtime.rglob("*.jar")))
        if checks["runtime_jars"] != 1062:
            raise ValueError("Runtime jar count changed")
        # Read-only doctor never starts Java, loads a checkpoint or downloads.
        if not os.access(runtime / "jre/Contents/Home/bin/java", os.X_OK):
            raise ValueError("Bundled native Java executable is not executable")
        if activation:
            value = json.loads(Path(activation).read_text())
            for key in ("model_freeze", "overlay_manifest", "jvm_proof"):
                if value.get(key) is None:
                    raise ValueError("Activation lacks independent JVM proof")
                verify(value[key]["path"], value[key])
            freeze = json.loads(Path(value["model_freeze"]["path"]).read_text())
            for item in [freeze["model"], freeze["protocol"], *freeze["code"]]:
                verify(item["path"], item)
            from process_s1_copdem import verified_reader_patch
            verified_reader_patch(Path(value["overlay_manifest"]["path"]), Path(value["jvm_proof"]["path"]))
            checks["activation_source_bindings"] = "verified"
        else:
            blockers.append("Supply --activation pointing to a freshly proved activation.json")
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        blockers.append(str(exc))
    return {"schema": "akash-native-doctor-v1", "checked_at": now(), "status": "ready_for_supported_raw_prerequisite_review" if not blockers else "blocked", "checks": checks, "blockers": blockers, "network_requests": 0, "model_loaded": False, "jvm_started": False, "gpu_used": False, "full_raw_replay_proved": False, "hazard_accuracy_established": False}


def smoke(root, activation, out):
    """Actual fresh JVM proof and frozen inference, using only synthetic arrays."""
    root, out = Path(root).resolve(strict=True), Path(out).absolute()
    check = doctor(root, activation)
    if check["blockers"]:
        raise ValueError("Environment doctor blockers: " + "; ".join(check["blockers"]))
    if out.exists() or out.is_symlink() or any(p.is_symlink() for p in out.parents):
        raise FileExistsError("Smoke requires a new output directory")
    out.mkdir(parents=True)
    current = json.loads(Path(activation).read_text())
    run_logged([sys.executable, str(root / "scripts/prove_snap_external_dem.py"), "--overlay-manifest", current["overlay_manifest"]["path"], "--out", str(out / "dem-proof")], out / "jvm-proof-launch.log", root, 180)
    import socket
    def deny_socket(*args, **kwargs):
        raise RuntimeError("Synthetic smoke disables outbound Python socket use")
    socket.create_connection = deny_socket
    socket.socket.connect = deny_socket
    import joblib
    import numpy as np
    import akash_flood.flood_model as module
    from threadpoolctl import threadpool_limits
    if not Path(module.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Imported model feature source is outside this extraction")
    freeze = json.loads(Path(current["model_freeze"]["path"]).read_text())
    model = joblib.load(freeze["model"]["path"])
    rows, cols = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    values = np.stack([-35 + cols * .4 + rows * .02, -42 + cols * .42 + rows * .01]).astype("float32")
    valid = np.ones((64, 64), dtype=bool); valid[4:8, 7:11] = False
    values[:, 20:24, 20:24] = np.nan
    with threadpool_limits(limits=4):
        scores, labels = module.predict_water(model, values, valid)
        second_scores, second_labels = module.predict_water(model, values, valid)
    if not np.array_equal(labels, second_labels) or not np.array_equal(scores, second_scores, equal_nan=True):
        raise ValueError("Frozen synthetic inference was not deterministic")
    if int((labels == 255).sum()) != 32 or not np.isnan(scores[labels == 255]).all():
        raise ValueError("Unknown synthetic observations were not preserved")
    np.save(out / "synthetic-scores.npy", scores, allow_pickle=False)
    np.save(out / "synthetic-labels.npy", labels, allow_pickle=False)
    proof = json.loads((out / "dem-proof/proof.json").read_text())
    result = {"schema": "akash-native-synthetic-smoke-v1", "completed_at": now(), "status": "passed", "source_root": str(root), "doctor": check, "model": record(freeze["model"]["path"]), "original_freeze": freeze["original_freeze"], "threshold": .5, "synthetic_cells": int(labels.size), "label_counts": {str(v): int((labels == v).sum()) for v in (0, 1, 255)}, "deterministic_repeat": True, "jvm_samples_per_process": proof["samples_per_jvm"], "corrected_max_abs_error_m": proof["corrected_max_abs_error_m"], "model_fit_called": False, "reference_labels_loaded": False, "gpu_used": False, "network_requested_by_smoke": False, "raw_satellite_processing": False, "hazard_accuracy_established": False, "artifacts": [record(out / name) for name in ("synthetic-scores.npy", "synthetic-labels.npy", "dem-proof/proof.json")]}
    write_json(out / "smoke.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--root", type=Path, default=ROOT)
    p = sub.add_parser("install")
    p.add_argument("--root", type=Path, default=ROOT)
    for name in ("dmg", "geoid", "egm96", "dem-source", "checkpoint", "freeze", "protocol"):
        p.add_argument("--" + name, type=Path, required=name in {"checkpoint", "freeze", "protocol"})
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--network-cap", type=int, default=NETWORK_CAP)
    p.add_argument("--download-timeout", type=int, default=1200)
    p.add_argument("--prove", action="store_true", help="Run independent JVM analytic proof after install; requires jpype1==1.6.0")
    p = sub.add_parser("activate")
    p.add_argument("--root", type=Path, default=ROOT); p.add_argument("--out", type=Path, required=True); p.add_argument("--prove", action="store_true")
    p = sub.add_parser("doctor")
    p.add_argument("--root", type=Path, default=ROOT); p.add_argument("--activation", type=Path)
    p = sub.add_parser("smoke")
    p.add_argument("--root", type=Path, default=ROOT); p.add_argument("--activation", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = {"schema": "akash-native-setup-plan-v1", "supported_here": supported(), "platform_contract": "macOS arm64 only", "network_opt_in": True, "pinned_remote_assets": {"snap_dmg": DMG, "egm2008_geoid": GEOID, "snap_egm96_dependency": EGM96, "upstream_dem_source": DEM_SOURCE}, "maximum_network_body_bytes": NETWORK_CAP, "dmg_body_cap_bytes": DMG["bytes"], "maximum_extracted_bytes": EXTRACTION_CAP, "separately_supplied_no_model_download": {"checkpoint": CHECKPOINT, "original_freeze": FREEZE, "training_protocol": PROTOCOL}, "fresh_destinations": list(INSTALL_DIRS), "network_requests": 0, "subprocesses": 0, "license_grant_or_agreement": False}
    elif args.command == "install":
        result = install(args)
    elif args.command == "activate":
        result = activate(args.root, args.out, with_proof=args.prove)
    elif args.command == "doctor":
        result = doctor(args.root, args.activation)
    else:
        result = smoke(args.root, args.activation, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.command == "doctor" and result["blockers"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
