"""Capture a dated final source review ZIP and relocated base synthetic smoke.

Uses the established positive allowlist/verifier without changing that original
script. Models/vendor assets and execution outputs are never included. Local
preparation only; this assigns no project license and publishes nothing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
DATE = "2026-10-10"


def now():
    return datetime.now(timezone.utc).isoformat()


def record(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_policy(root):
    spec = importlib.util.spec_from_file_location("established_source_policy", root / "scripts/package_source_candidate_20261008.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(root, output, policy):
    if output.exists() or output.is_symlink():
        raise FileExistsError("Preserve earlier source candidate; use a fresh output")
    files = policy.collect(root)
    required = {"scripts/build_situation_brief_20261010.py", "scripts/build_submission_report_20261010.py", "scripts/build_optical_composite_viewer.py", "scripts/serve_multi_case_copilot.py", "scripts/setup_native_runtime.py", "web/optical-composite-audit.html"}
    if not required <= files.keys():
        raise ValueError("Final implementation is incomplete; missing " + ", ".join(sorted(required - files.keys())))
    rows = [{"path": name, "bytes": len(data), "sha256": policy.digest(data)} for name, data in sorted(files.items())]
    manifest = {"schema": "akash-source-review-candidate-v1", "source_date": DATE, "bundle_root": policy.BUNDLE, "scope": "local source-only review candidate; no public publication or project license grant", "source_roots": sorted(policy.SOURCE_SUFFIXES), "root_files": sorted(policy.ROOT_FILES), "excluded_tests": policy.EXCLUDED_TESTS, "excluded_asset_categories": ["raw/reference checking data and geometry", "satellite imagery and DEM/geoid grids", "trained checkpoints and copilot model weights", "vendor/generated SNAP source and binaries", "virtual environments, caches, archives and old execution outputs", "unrelated user files and papers"], "base_install": "uv sync --frozen", "optional_inference_overlay": "scikit-learn==1.7.2 is required separately for frozen-model work; model and prerequisites are excluded", "files": rows}
    content = json.dumps(manifest, indent=2, ensure_ascii=False).encode() + b"\n"
    if sum(map(len, files.values())) + len(content) > policy.MAX_TOTAL_BYTES:
        raise ValueError("Source candidate exceeds established resource bound")
    payload = dict(files); payload[policy.MANIFEST] = content
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".source-final-", suffix=".zip", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as zipped:
            for relative, data in sorted(payload.items()):
                info = ZipInfo(policy.BUNDLE + "/" + relative, (2026, 10, 10, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                zipped.writestr(info, data)
        # Hash/count every member and ensure the whole source set stayed stable.
        verified = policy.verify(temporary)
        if policy.collect(root) != files:
            raise ValueError("Included source membership or bytes changed during capture")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"created_at": now(), **record(output), **verified, "source_payload_bytes": sum(map(len, files.values())), "files_by_root": dict(sorted(Counter(name.split("/")[0] if "/" in name else "root" for name in files).items())), "files": rows, "excluded_tests": policy.EXCLUDED_TESTS, "excluded_asset_categories": manifest["excluded_asset_categories"], "source_date": DATE, "required_new_implementation_members_present": sorted(required)}


def run_logged(command, cwd, log, timeout=300):
    with log.open("x") as stream:
        process = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    if process.returncode:
        raise RuntimeError("Source verification command failed; preserved " + str(log))


def source_smoke(root, output, policy, prefix):
    with tempfile.TemporaryDirectory(prefix="akash-final-source-smoke-") as temporary:
        folder = Path(temporary).resolve()
        with ZipFile(output) as zipped:
            zipped.extractall(folder)  # Established verifier checked every member.
        extraction = folder / policy.BUNDLE
        run_logged(["uv", "sync", "--offline", "--frozen"], extraction, prefix.with_suffix(".install.log"))
        smoke = extraction / "output/final synthetic smoke"
        run_logged(["uv", "run", "--offline", "--no-sync", "python", "release/smoke_source.py", "--out", str(smoke), "--forbid-original-root", str(root)], extraction, prefix.with_suffix(".smoke.log"))
        receipt = json.loads((smoke / "smoke-receipt.json").read_text())
        if (receipt["network_requests"] != 0 or receipt["original_workspace_reads"] != 0 or receipt["model_inference_run"] is not False or receipt["gpu_used"] is not False or receipt["scientific_accuracy_established"] is not False):
            raise ValueError("Relocated source smoke scope changed")
        # Preserve the fixture receipt and witnesses before deleting the temp tree.
        write_json(prefix.with_suffix(".smoke.json"), receipt)
        write_json(prefix.with_suffix(".certificates.json"), json.loads((smoke / "uncertainty-certificate.json").read_text()))
        return {"status": "passed", "dependency_install": "uv sync --offline --frozen; populated cache", "fixture_kind": receipt["data_kind"], "synthetic_shape": [64, 64], "new_water_candidate_pixels": receipt["synthetic_metrics"]["new_water_candidate_pixels"], "graph_nodes": receipt["graph_nodes"], "directed_edges": receipt["directed_edges"], "bound_status_counts": receipt["bound_status_counts"], "certificate_checks": receipt["certificate_checks"], "wall_seconds": receipt["wall_seconds"], "network_requests": receipt["network_requests"], "original_workspace_reads": receipt["original_workspace_reads"], "network_guard_probe_blocked": receipt["network_guard_probe_blocked"], "original_read_guard_probe_blocked": receipt["original_read_guard_probe_blocked"], "model_inference_run": False, "native_runtime_run": False, "raw_satellite_processing": False, "gpu_used": False, "scientific_accuracy_established": False, "temporary_extraction_removed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if root != ROOT:
        raise ValueError("Use this wrapper from the target source root")
    output = root / "output/release/akash-source-candidate-2026-10-10.zip"
    adjacent = output.with_name(output.name + ".verification.json")
    prefix = root / "release/native-setup/source-final-review-2026-10-10"
    for path in (output, adjacent, prefix.with_suffix(".json"), prefix.with_suffix(".install.log"), prefix.with_suffix(".smoke.log"), prefix.with_suffix(".smoke.json"), prefix.with_suffix(".certificates.json")):
        if path.exists() or path.is_symlink():
            raise FileExistsError("Preserve prior candidate/review/log: " + str(path))
    started = now()
    policy = load_policy(root)
    captured = capture(root, output, policy)
    smoke = source_smoke(root, output, policy, prefix)
    # Changes after capture also fail review, without overwriting the candidate.
    current = policy.collect(root)
    for item in captured["files"]:
        if item["path"] not in current or len(current[item["path"]]) != item["bytes"] or policy.digest(current[item["path"]]) != item["sha256"]:
            raise ValueError("A captured source changed during smoke")
    if set(current) != {item["path"] for item in captured["files"]}:
        raise ValueError("Source membership changed after capture")
    value = {"schema": "akash-final-source-candidate-review-v1", "started_at": started, "completed_at": now(), "status": "passed", "source_date": DATE, "scope": "Final local source-only review candidate and relocated prepared-raster/graph fixture smoke; no publication/license grant", "candidate": captured, "positive_allowlist_verification": "All archive members and source records matched established policy, sizes and SHA256; source membership/bytes stable through capture and smoke", "relocated_smoke": smoke, "complete_checkout_suite": {"passed": 883, "warnings": 2241, "seconds": 24.66, "scope": "Existing frozen implementation receipt; full suite was not repeated for packaging", "log": record(root / "release/native-setup/full-suite-after-freeze-2026-10-10.log")}, "native_setup_rehearsal": {"repeated": False, "review": record(root / "research/native-setup-review-2026-10-10.json")}, "publication": False, "project_license_assigned": False, "assets_or_weights_bundled": False, "originals_overwritten": False, "new_copilot_execution_receipts": "Preserved independently outside the source-only tier", "limitations": ["Included real-input/PDF/report scripts depend on omitted assets and optional libraries", "Checkpoint/runtime/geoid/reference data are excluded; no complete raw replay from this ZIP alone", "Synthetic software checks are not hazard validation", "Project/model/inherited-asset redistribution rights remain unresolved"], "artifacts": [record(prefix.with_suffix(suffix)) for suffix in (".install.log", ".smoke.log", ".smoke.json", ".certificates.json")]}
    write_json(adjacent, value)
    write_json(prefix.with_suffix(".json"), {**value, "adjacent_verification": record(adjacent)})
    print(json.dumps({"status": value["status"], "candidate": {key: captured[key] for key in ("path", "bytes", "sha256", "member_count", "bound_source_files", "uncompressed_bytes", "source_payload_bytes")}, "verification": record(adjacent), "smoke": smoke}, indent=2))


if __name__ == "__main__":
    main()
