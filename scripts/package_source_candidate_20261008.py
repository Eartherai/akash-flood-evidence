"""Prepare/verify a source-only local review ZIP; never publish or execute assets.

Only UTF-8 source files from explicit directories and release notes are copied.
The manifest binds each archive member and records the unavailable-data test tier.
This script does not choose a license for the project or its upstream components.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "akash-source-candidate"
MANIFEST = "release/SOURCE-MANIFEST.json"
ROOT_FILES = frozenset({"README.md", "pyproject.toml", "uv.lock"})
SOURCE_SUFFIXES = {
    "src": frozenset({".py"}),
    "scripts": frozenset({".py"}),
    "tests": frozenset({".py"}),
    "web": frozenset({".html", ".css", ".js"}),
    "release": frozenset({".md", ".py"}),
}
EXCLUDED_TESTS = {
    "tests/test_checking_native_water_extent.py": "imports excluded research/evaluation-only source at collection",
    "tests/test_kathmandu_checking_transport.py": "imports excluded research/evaluation-only source when optional pyshp is installed",
    "tests/test_research_server.py": "imports research server requiring excluded original OSM data at collection",
    "tests/test_research_radar_server.py": "imports research server requiring excluded original OSM data at collection",
    "tests/test_trained_water_server.py": "imports research server requiring excluded original OSM data at collection",
}
MAX_FILE_BYTES = 2_000_000
MAX_TOTAL_BYTES = 20_000_000
MAX_MEMBERS = 1000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def allowed(relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return False
    if "\\" in relative or ":" in relative or str(path) != relative:
        return False
    if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
        return False
    if relative in ROOT_FILES:
        return True
    return (len(path.parts) > 1 and path.parts[0] in SOURCE_SUFFIXES
            and path.suffix in SOURCE_SUFFIXES[path.parts[0]]
            and relative not in EXCLUDED_TESTS)


def collect(root):
    root = Path(root).resolve(strict=True)
    paths = [root / item for item in sorted(ROOT_FILES)]
    for directory in SOURCE_SUFFIXES:
        folder = root / directory
        if folder.is_symlink():
            raise ValueError(f"Refuse symlink source directory: {directory}")
        paths.extend(path for path in folder.rglob("*") if allowed(path.relative_to(root).as_posix()))
    result = {}
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        if not allowed(relative):
            continue
        if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent):
            raise ValueError(f"Refuse symlink source file: {relative}")
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Not a regular in-root source file: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Source file exceeds cap: {relative}")
        data = path.read_bytes()
        data.decode("utf-8")
        if b"\0" in data:
            raise ValueError(f"Binary source content: {relative}")
        result[relative] = data
    if not ROOT_FILES <= result.keys():
        raise ValueError("Missing required root source files")
    if len(result) + 1 > MAX_MEMBERS or sum(map(len, result.values())) > MAX_TOTAL_BYTES:
        raise ValueError("Source package exceeds bounded members/bytes")
    return result


def prepare(root, output):
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise ValueError("Preserve existing source ZIP; choose a new output path")
    files = collect(root)
    manifest = {
        "schema": "akash-source-review-candidate-v1",
        "source_date": "2026-10-08",
        "bundle_root": BUNDLE,
        "scope": "local source-only review candidate; no public publication or project license grant",
        "source_roots": sorted(SOURCE_SUFFIXES),
        "root_files": sorted(ROOT_FILES),
        "excluded_tests": EXCLUDED_TESTS,
        "excluded_asset_categories": ["raw/reference checking data and geometry", "satellite imagery and DEM/geoid grids",
                                      "trained checkpoints and copilot model weights", "vendor/generated SNAP source and binaries",
                                      "virtual environments, caches, archives and old execution outputs", "unrelated user files and papers"],
        "base_install": "uv sync --frozen",
        "optional_inference_overlay": "scikit-learn==1.7.2 is required separately for frozen-model work; model and prerequisites are excluded",
        "files": [{"path": path, "bytes": len(data), "sha256": digest(data)} for path, data in sorted(files.items())],
    }
    payload = json.dumps(manifest, indent=2, ensure_ascii=False).encode() + b"\n"
    if sum(map(len, files.values())) + len(payload) > MAX_TOTAL_BYTES:
        raise ValueError("Source package plus manifest exceeds byte cap")
    files[MANIFEST] = payload
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".source-candidate-", suffix=".zip", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for path, data in sorted(files.items()):
                info = ZipInfo(f"{BUNDLE}/{path}", (2026, 10, 8, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, data)
        # Fail on an input mutation instead of blessing an unstable source snapshot.
        for path, data in files.items():
            if path != MANIFEST and (root / path).read_bytes() != data:
                raise ValueError(f"Source changed during capture: {path}")
        os.link(temporary, output)  # Atomic no-overwrite publication to the local path.
    finally:
        temporary.unlink(missing_ok=True)
    verified = verify(output)
    return {"created_at": datetime.now(timezone.utc).isoformat(), "path": str(output),
            "bytes": output.stat().st_size, "sha256": digest(output.read_bytes()), **verified}


def verify(archive_path):
    with ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > MAX_MEMBERS or sum(item.file_size for item in members) > MAX_TOTAL_BYTES:
            raise ValueError("Archive exceeds resource caps")
        names = [item.filename for item in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        expected_manifest = f"{BUNDLE}/{MANIFEST}"
        if expected_manifest not in names:
            raise ValueError("Missing source manifest")
        payloads = {}
        for item in members:
            if not item.filename.startswith(BUNDLE + "/"):
                raise ValueError("Archive member outside bundle root")
            relative = item.filename[len(BUNDLE) + 1:]
            if relative != MANIFEST and not allowed(relative):
                raise ValueError("Archive member outside source allowlist")
            if item.is_dir() or item.file_size > MAX_FILE_BYTES:
                raise ValueError("Unexpected directory or oversized member")
            mode = item.external_attr >> 16
            if mode and not stat.S_ISREG(mode):
                raise ValueError("Nonregular archive member")
            data = archive.read(item)
            data.decode("utf-8")
            if b"\0" in data:
                raise ValueError("Binary archive payload")
            payloads[relative] = data
        manifest = json.loads(payloads.pop(MANIFEST))
        if manifest.get("schema") != "akash-source-review-candidate-v1" or manifest.get("bundle_root") != BUNDLE:
            raise ValueError("Unexpected source manifest schema/root")
        if (manifest.get("excluded_tests") != EXCLUDED_TESTS or manifest.get("root_files") != sorted(ROOT_FILES)
                or manifest.get("source_roots") != sorted(SOURCE_SUFFIXES)):
            raise ValueError("Manifest source selection differs from fixed policy")
        records = manifest.get("files")
        if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
            raise ValueError("Invalid manifest file records")
        if len({item.get("path") for item in records}) != len(records):
            raise ValueError("Duplicate manifest paths")
        if {item.get("path") for item in records} != payloads.keys():
            raise ValueError("Manifest/archive members differ")
        if not ROOT_FILES <= payloads.keys():
            raise ValueError("Manifest omits required root source files")
        for item in records:
            data = payloads[item["path"]]
            if type(item.get("bytes")) is not int or item["bytes"] != len(data) or item.get("sha256") != digest(data):
                raise ValueError(f"Source member hash/size mismatch: {item['path']}")
    return {"member_count": len(members), "bound_source_files": len(records),
            "uncompressed_bytes": sum(item.file_size for item in members), "manifest_sha256": digest(json.dumps(manifest, indent=2, ensure_ascii=False).encode() + b"\n")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if bool(args.out) == bool(args.verify):
        parser.error("Choose exactly one of --out and --verify")
    print(json.dumps(verify(args.verify) if args.verify else prepare(args.root, args.out), indent=2))


if __name__ == "__main__":
    main()
