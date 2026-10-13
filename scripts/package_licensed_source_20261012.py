"""Prepare/verify the source-only MIT export without changing older policies.

The original positive source allowlist is retained, with root LICENSE and two
exact dated geographic-viewer module paths added.
The human owner selected MIT for original code; upstream assets retain their
own terms. This script neither publishes a repository nor verifies ownership.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import package_source_candidate_20261008 as policy

SCHEMA = "akash-source-mit-review-v1"
BUNDLE = policy.BUNDLE
MANIFEST = policy.MANIFEST
ROOT_FILES = policy.ROOT_FILES | {"LICENSE"}
ADDITIVE_SOURCE_PATHS = frozenset({"web/observation-geography-core-20261012.mjs",
                                   "web/observation-geography-ui-20261012.mjs"})
LICENSE_SHA256 = "1ebce672ecef3f367b9173e6b2bfd506f9b0ea5c0baf289d7af4c6b9ac174704"
BASE_POLICY = "scripts/package_source_candidate_20261008.py"
THIS_POLICY = "scripts/package_licensed_source_20261012.py"
OWNER_RECEIPT = "research/license-selection-2026-10-11/owner-statement.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def allowed(relative):
    return relative == "LICENSE" or relative in ADDITIVE_SOURCE_PATHS or policy.allowed(relative)


def collect(root):
    root = Path(root).resolve(strict=True)
    files = policy.collect(root)
    for relative in sorted({"LICENSE"} | ADDITIVE_SOURCE_PATHS):
        path = root / relative
        if (any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent)
                or not path.is_file() or not path.resolve().is_relative_to(root)):
            raise ValueError(f"Additional source must be a regular in-root file: {relative}")
        data = path.read_bytes()
        data.decode("utf-8")
        if len(data) > policy.MAX_FILE_BYTES or b"\0" in data:
            raise ValueError(f"Additional source is not bounded UTF-8: {relative}")
        files[relative] = data
    if digest(files["LICENSE"]) != LICENSE_SHA256:
        raise ValueError("LICENSE differs from the owner-approved text")
    if len(files) + 1 > policy.MAX_MEMBERS or sum(map(len, files.values())) > policy.MAX_TOTAL_BYTES:
        raise ValueError("Licensed source exceeds fixed resource caps")
    return files


def verify(archive_path):
    with ZipFile(archive_path) as archive:
        members = archive.infolist()
        if (len(members) > policy.MAX_MEMBERS
                or sum(item.file_size for item in members) > policy.MAX_TOTAL_BYTES):
            raise ValueError("Archive exceeds resource caps")
        names = [item.filename for item in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        if f"{BUNDLE}/{MANIFEST}" not in names:
            raise ValueError("Missing licensed-source manifest")
        payloads = {}
        for item in members:
            if not item.filename.startswith(BUNDLE + "/"):
                raise ValueError("Member outside bundle root")
            relative = item.filename[len(BUNDLE) + 1:]
            if relative != MANIFEST and not allowed(relative):
                raise ValueError("Member outside fixed positive allowlist")
            if item.is_dir() or item.file_size > policy.MAX_FILE_BYTES or item.flag_bits & 1:
                raise ValueError("Directory, oversized or encrypted member")
            mode = item.external_attr >> 16
            if mode and not stat.S_ISREG(mode):
                raise ValueError("Nonregular archive member")
            data = archive.read(item)  # Reading every member also validates its ZIP CRC.
            data.decode("utf-8")
            if b"\0" in data:
                raise ValueError("Binary archive payload")
            payloads[relative] = data
        manifest_payload = payloads.pop(MANIFEST)
        manifest = json.loads(manifest_payload)
        if manifest.get("schema") != SCHEMA or manifest.get("bundle_root") != BUNDLE:
            raise ValueError("Unexpected licensed-source schema/root")
        if (manifest.get("root_files") != sorted(ROOT_FILES)
                or manifest.get("additive_source_paths") != sorted(ADDITIVE_SOURCE_PATHS)
                or manifest.get("source_roots") != sorted(policy.SOURCE_SUFFIXES)
                or manifest.get("excluded_tests") != policy.EXCLUDED_TESTS
                or manifest.get("selection_policy") != {"base": BASE_POLICY, "licensed_extension": THIS_POLICY}):
            raise ValueError("Source selection differs from fixed policy")
        license_record = manifest.get("project_license", {})
        if (license_record.get("spdx_id") != "MIT"
                or license_record.get("license_sha256") != LICENSE_SHA256
                or license_record.get("scope") != "Original Akash project source code only"
                or license_record.get("upstream_terms_preserved") is not True
                or license_record.get("ownership_independently_verified") is not False):
            raise ValueError("License scope differs from approved source-only terms")
        records = manifest.get("files")
        if (not isinstance(records, list)
                or any(not isinstance(item, dict) for item in records)
                or len({item.get("path") for item in records}) != len(records)
                or {item.get("path") for item in records} != payloads.keys()
                or not (ROOT_FILES | ADDITIVE_SOURCE_PATHS) <= payloads.keys()):
            raise ValueError("Invalid, duplicate or incomplete manifest records")
        for item in records:
            data = payloads[item["path"]]
            if (type(item.get("bytes")) is not int or item["bytes"] != len(data)
                    or item.get("sha256") != digest(data)):
                raise ValueError(f"Hash/size mismatch: {item['path']}")
        if digest(payloads["LICENSE"]) != LICENSE_SHA256:
            raise ValueError("Archived license differs from approved text")
        for field, path in (("base_policy_sha256", BASE_POLICY), ("licensed_policy_sha256", THIS_POLICY)):
            if path not in payloads or manifest.get(field) != digest(payloads[path]):
                raise ValueError("Source policy identity mismatch")
    return {"status": "passed", "schema": SCHEMA, "member_count": len(members),
            "bound_source_files": len(records), "crc_validated_members": len(members),
            "uncompressed_bytes": sum(item.file_size for item in members),
            "manifest_sha256": digest(manifest_payload), "license_sha256": LICENSE_SHA256,
            "project_source_license": "MIT", "upstream_terms_preserved": True,
            "public_publication_performed": False}


def prepare(root, output):
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve previous exports; choose a new output path")
    files = collect(root)
    owner_path = root / OWNER_RECEIPT
    owner = json.loads(owner_path.read_text())
    if (owner.get("human_statement") != "Entirely my original code; use MIT"
            or owner.get("license_sha256") != LICENSE_SHA256):
        raise ValueError("Owner assertion does not match approved license")
    manifest = {
        "schema": SCHEMA, "source_date": "2026-10-12", "bundle_root": BUNDLE,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Local source-only MIT review export; no public repository publication",
        "selection_policy": {"base": BASE_POLICY, "licensed_extension": THIS_POLICY},
        "base_policy_sha256": digest(files[BASE_POLICY]),
        "licensed_policy_sha256": digest(files[THIS_POLICY]),
        "source_roots": sorted(policy.SOURCE_SUFFIXES), "root_files": sorted(ROOT_FILES),
        "additive_source_paths": sorted(ADDITIVE_SOURCE_PATHS),
        "excluded_tests": policy.EXCLUDED_TESTS,
        "project_license": {"spdx_id": "MIT", "path": "LICENSE", "license_sha256": LICENSE_SHA256,
                            "scope": "Original Akash project source code only",
                            "owner_assertion_recorded_utc": owner["recorded_utc"],
                            "owner_assertion_sha256": digest(owner_path.read_bytes()),
                            "ownership_independently_verified": False,
                            "upstream_terms_preserved": True,
                            "third_party_boundaries": "release/THIRD-PARTY-BOUNDARIES.md"},
        "excluded_asset_categories": ["raw/reference checking data and geometry",
                                      "satellite imagery and DEM/geoid grids",
                                      "trained checkpoints and copilot model weights",
                                      "vendor/generated SNAP source and binaries",
                                      "virtual environments, caches, archives and old execution outputs",
                                      "unrelated user files and papers"],
        "base_install": "uv sync --frozen",
        "optional_inference_overlay": "scikit-learn==1.7.2 separately; excluded checkpoint/prerequisites required",
        "files": [{"path": path, "bytes": len(data), "sha256": digest(data)}
                  for path, data in sorted(files.items())],
    }
    files[MANIFEST] = json.dumps(manifest, indent=2, ensure_ascii=False).encode() + b"\n"
    if sum(map(len, files.values())) > policy.MAX_TOTAL_BYTES:
        raise ValueError("Source plus manifest exceeds fixed byte cap")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".licensed-source-", suffix=".zip", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for path, data in sorted(files.items()):
                info = ZipInfo(f"{BUNDLE}/{path}", (2026, 10, 12, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, data)
        if collect(root) != {path: data for path, data in files.items() if path != MANIFEST}:
            raise ValueError("Source selection or contents changed during capture")
        verify(temporary)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {**verify(output), "path": str(output), "bytes": output.stat().st_size,
            "sha256": digest(output.read_bytes()), "captured_utc": manifest["captured_utc"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=policy.ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if bool(args.out) == bool(args.verify):
        parser.error("Choose exactly one of --out and --verify")
    print(json.dumps(verify(args.verify) if args.verify else prepare(args.root, args.out), indent=2))


if __name__ == "__main__":
    main()
