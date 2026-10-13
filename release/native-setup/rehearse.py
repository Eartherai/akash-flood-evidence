"""Rehearse a new source ZIP, cached native setup and sandboxed relocation.

Consumes separately held local prerequisites. Never publishes or changes old
case/checkpoint evidence. Run from the complete local working checkout only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from check_isolation import check as check_isolation

ROOT = Path(__file__).resolve().parents[2]


def record(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def run(command, cwd, log, timeout=600):
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError("Rehearsal command failed; preserved " + str(log))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new rehearsal path")
    if not Path("/usr/bin/sandbox-exec").is_file():
        raise RuntimeError("macOS sandbox-exec is required for this isolated rehearsal")
    output.mkdir(parents=True)
    name = "native_source_packager"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/package_source_candidate_20261008.py")
    package = importlib.util.module_from_spec(spec); spec.loader.exec_module(package)
    archive = output / "akash-source-with-native-setup.zip"
    package_receipt = package.prepare(ROOT, archive)
    assets = {"dmg": ROOT / "data/snap-runtime/esa-snap_sentinel_macos_arm-14.0.0.dmg", "geoid": ROOT / "data/copdem-ellipsoid/us_nga_egm08_25.tif", "egm96": ROOT / "data/snap-runtime/user-cache/auxdata/dem/egm96/ww15mgh_b.zip", "dem-source": ROOT / "research/copdem-review-evidence/FileElevationModel-14.0.0.java", "checkpoint": ROOT / "outputs/sen1-cpu-v1/water-model.joblib", "freeze": ROOT / "outputs/sen1-cpu-v1/model-evaluation-freeze.json", "protocol": ROOT / "research/sen1-cpu-training-protocol.json"}
    before = {key: record(path) for key, path in assets.items()}
    start = datetime.now(timezone.utc).isoformat()
    # Temporary runtime trees are removed after receipts/logs are copied out.
    with tempfile.TemporaryDirectory(prefix="akash-source-native-rehearsal-") as temporary:
        folder = Path(temporary).resolve()
        with ZipFile(archive) as zipped:
            zipped.extractall(folder)  # package.prepare already applied traversal/allowlist checks.
        extraction = folder / package.BUNDLE
        run(["uv", "sync", "--offline", "--frozen"], extraction, output / "dependency-sync.log")
        overlay = ["uv", "run", "--offline", "--with", "scikit-learn==1.7.2", "--with", "jpype1==1.6.0", "python"]
        argv = overlay + ["scripts/setup_native_runtime.py", "install", "--prove"]
        for key, path in assets.items():
            argv.extend(["--" + key, str(path)])
        run(argv, extraction, output / "cached-install.log")
        # Moving the installed extraction must not require the old temp root.
        moved = folder / "source relocated with spaces"
        extraction.rename(moved)
        run(overlay + ["scripts/setup_native_runtime.py", "activate", "--out", "data/native-setup/activation-after-move", "--prove"], moved, output / "relocation-activation.log")
        activation = moved / "data/native-setup/activation-after-move/activation.json"
        profile = '(version 1) (allow default) (deny network-outbound) (deny file-read* (subpath ' + json.dumps(str(ROOT)) + '))'
        profile_path = output / "isolation.sb"; profile_path.write_text(profile + "\n")
        check_isolation(profile_path, ROOT, output / "isolation-guard-check.json")
        # Keep uv and its temporary overlay interpreter in the same sandboxed
        # command lifetime; uv may remove that interpreter after it returns.
        command = ["/usr/bin/sandbox-exec", "-f", str(profile_path), *overlay, "scripts/setup_native_runtime.py", "smoke", "--activation", str(activation), "--out", str(moved / "output/isolated-native-smoke")]
        run(command, moved, output / "isolated-smoke.log")
        smoke = moved / "output/isolated-native-smoke"
        result = json.loads((smoke / "smoke.json").read_text())
        for source, name in ((moved / "data/native-setup/installation-receipt.json", "installation-receipt.json"), (moved / "data/native-setup/asset-contract.json", "asset-contract.json"), (activation, "activation.json"), (smoke / "smoke.json", "smoke.json"), (smoke / "dem-proof/proof.json", "dem-proof.json"), (smoke / "dem-proof/vendor.json", "vendor-jvm.json"), (smoke / "dem-proof/corrected.json", "corrected-jvm.json"), (smoke / "synthetic-scores.npy", "synthetic-scores.npy"), (smoke / "synthetic-labels.npy", "synthetic-labels.npy")):
            shutil.copyfile(source, output / name)
        result_scope = {key: result[key] for key in ("status", "synthetic_cells", "label_counts", "deterministic_repeat", "jvm_samples_per_process", "corrected_max_abs_error_m", "reference_labels_loaded", "gpu_used", "raw_satellite_processing", "hazard_accuracy_established")}
    after = {key: record(path) for key, path in assets.items()}
    if before != after:
        raise ValueError("An original prerequisite changed during rehearsal")
    receipt = {"schema": "akash-native-source-rehearsal-v1", "started_at": start, "completed_at": datetime.now(timezone.utc).isoformat(), "status": "passed", "source_archive": package_receipt, "separately_supplied_assets": before, "original_assets_unchanged": True, "actual_source_extraction": True, "source_and_installed_assets_moved_before_smoke": True, "original_project_reads_denied_during_smoke": True, "outbound_network_denied_for_smoke_process_tree": True, "isolation_denials_actually_verified": True, "dependency_install_from_uv_cache": True, "asset_download_network_bytes": 0, "temporary_runtime_tree_removed": True, "synthetic_result": result_scope, "limitations": ["macOS arm64 only", "Separately supplied locally trained model/protocol; no redistribution rights established", "No raw satellite acquisition or processing replay", "No flood/debris or geolocation accuracy established"], "artifacts": [record(p) for p in sorted(output.iterdir()) if p.is_file()]}
    (output / "rehearsal.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(output / "rehearsal.json"), "synthetic_result": result_scope}, indent=2))


if __name__ == "__main__":
    main()
