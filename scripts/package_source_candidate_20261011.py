"""Capture the current source with the existing positive allowlist and a dated manifest.

Older packages and the original packager are preserved. This is local packaging,
not publication or a grant of rights to any upstream source or data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile

import package_source_candidate_20261008 as policy


def prepare(root: Path, output: Path):
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Preserve previous packages; select a new output path")
    with tempfile.TemporaryDirectory(prefix="akash-source-closeout-") as temporary:
        staged = Path(temporary) / "source.zip"
        policy.prepare(root, staged)
        with ZipFile(staged) as archive:
            payloads = [(item, archive.read(item)) for item in archive.infolist()]
        manifest_name = policy.BUNDLE + "/" + policy.MANIFEST
        captured = datetime.now(timezone.utc).isoformat()
        output.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(output, "x") as archive:
            for item, data in payloads:
                if item.filename == manifest_name:
                    manifest = json.loads(data)
                    manifest.update(source_date="2026-10-11", captured_utc=captured,
                                    selection_policy="scripts/package_source_candidate_20261008.py")
                    data = json.dumps(manifest, indent=2, ensure_ascii=False).encode() + b"\n"
                item.date_time = (2026, 10, 11, 0, 0, 0)
                archive.writestr(item, data)
    result = policy.verify(output)
    return dict(result, path=str(output), captured_utc=captured,
                bytes=output.stat().st_size,
                sha256=hashlib.sha256(output.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=policy.ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.out), indent=2))


if __name__ == "__main__":
    main()
