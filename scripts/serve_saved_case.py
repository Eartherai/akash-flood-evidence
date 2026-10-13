"""Verify and display an immutable saved case using the Python standard library.

This viewer does not run satellite processing, model inference, a copilot,
training, graph analysis or checking-label evaluation. It reads only the files
named in its separate saved-case release manifest.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit


SCHEMA = "akash-saved-case-replay-v1"
CASE_ID = "fresh-d22bcbfb6d7bc937"
MAX_FILE_BYTES = 30_000_000
MAX_TOTAL_BYTES = 64_000_000
SOURCE_PATHS = (
    "report/access-proof.json", "report/map.png", "report/optical-manifest.json",
    "report/optical-overview.png", "report/optical-protocol.json", "report/report.html",
    "report/report.json", "report/source-and-output-manifest.json",
    "documents/akash-technical-evidence-report.pdf",
    "documents/akash-technical-evidence-report.provenance.json",
    "documents/akash-attributed-situation-report.pdf",
    "documents/akash-attributed-situation-report.provenance.json",
    "evidence/execution-receipt.json", "evidence/reproducibility-audit.json",
    "evidence/water/manifest.json", "evidence/water/policy.json",
    "evidence/processing/run-metadata.json", "evidence/exposure/summary.json",
    "evidence/exposure/scenario-policy.json",
    "evidence/exposure/source-and-output-manifest.json",
    "source/osm.json", "source/osm.provenance.json",
)
GENERATED_PATHS = ("README.md", "NOTICES.txt", "serve_saved_case.py")
ALLOWED_FILES = frozenset((*SOURCE_PATHS, *GENERATED_PATHS))
CONTROL_FILES = frozenset(("manifest.json", "manifest.sha256"))
FALSE_FLAGS = ("raw_pipeline_rerun", "model_inference_run", "training_run",
               "copilot_available", "hazard_accuracy_established")


def distinct_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def checked_path(root: Path, relative: str) -> Path:
    """Resolve only a declared relative regular file; refuse all symlinks."""
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or ":" in relative or "%" in relative or "\x00" in relative):
        raise ValueError("Invalid bundle path")
    parts = PurePosixPath(relative).parts
    if (PurePosixPath(relative).is_absolute() or any(p in ("", ".", "..") for p in parts)
            or "/".join(parts) != relative):
        raise ValueError("Bundle paths must be canonical and relative")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Bundle symlinks are prohibited")
    if not current.is_file() or not current.resolve().is_relative_to(root):
        raise ValueError("Missing regular bundle file: " + relative)
    return current


def file_bytes(root: Path, relative: str, size: int | None = None,
               digest: str | None = None) -> bytes:
    path = checked_path(root, relative)
    actual_size = path.stat().st_size
    if actual_size > MAX_FILE_BYTES or (size is not None and actual_size != size):
        raise ValueError("Bundle file size mismatch: " + relative)
    with path.open("rb") as stream:
        content = stream.read(MAX_FILE_BYTES + 1)
    if len(content) != actual_size or len(content) > MAX_FILE_BYTES:
        raise ValueError("Bundle file changed while reading: " + relative)
    if digest is not None and hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Bundle file checksum mismatch: " + relative)
    return content


class VerifiedBundle:
    def __init__(self, root):
        supplied = Path(root).absolute()
        if supplied.is_symlink():
            raise ValueError("Bundle root cannot be a symlink")
        self.root = supplied.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Bundle root must be a directory")
        raw = file_bytes(self.root, "manifest.json")
        if len(raw) > 1_000_000:
            raise ValueError("Manifest exceeds 1 MB")
        checksum = file_bytes(self.root, "manifest.sha256").decode("ascii")
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if checksum != self.manifest_sha256 + "  manifest.json\n":
            raise ValueError("Release manifest checksum mismatch")
        document = json.loads(raw, object_pairs_hook=distinct_keys,
                              parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
        if (not isinstance(document, dict) or document.get("schema") != SCHEMA
                or document.get("mode") != "saved_case_replay"
                or document.get("case_id") != CASE_ID
                or any(document.get(flag) is not False for flag in FALSE_FLAGS)
                or document.get("source_file_count") != 22):
            raise ValueError("Require the explicit non-inference saved-case contract")
        rows = document.get("files")
        if not isinstance(rows, list) or len(rows) != len(ALLOWED_FILES):
            raise ValueError("Require the exact saved-case file allowlist")
        self.files = {}
        total = 0
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Invalid file record")
            name, size, digest = row.get("path"), row.get("bytes"), row.get("sha256")
            if name not in ALLOWED_FILES or name in self.files:
                raise ValueError("Unexpected or duplicate bundle file")
            if (type(size) is not int or size < 0 or size > MAX_FILE_BYTES
                    or not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest)):
                raise ValueError("Invalid bundle size/checksum")
            if row.get("role") != ("preserved_source_artifact" if name in SOURCE_PATHS else "viewer_documentation"):
                raise ValueError("Invalid bundle artifact role")
            file_bytes(self.root, name, size, digest)
            self.files[name] = {"bytes": size, "sha256": digest}
            total += size
        if self.files.keys() != ALLOWED_FILES or total > MAX_TOTAL_BYTES:
            raise ValueError("Incomplete or oversized saved-case bundle")
        if document.get("payload_bytes") != total:
            raise ValueError("Declared payload bytes differ from verified files")
        for name in CONTROL_FILES:
            self.files[name] = {"bytes": len(raw if name == "manifest.json" else checksum.encode()),
                                "sha256": hashlib.sha256(raw if name == "manifest.json" else checksum.encode()).hexdigest()}
        self.document = document
        self.payload_bytes = total

    def read(self, name: str) -> bytes:
        if name not in self.files:
            raise ValueError("File is outside the saved-case allowlist")
        row = self.files[name]
        return file_bytes(self.root, name, row["bytes"], row["sha256"])

    def summary(self):
        return {"status": "verified", "mode": "saved_case_replay", "case_id": CASE_ID,
                "manifest_sha256": self.manifest_sha256, "verified_file_count": len(ALLOWED_FILES),
                "payload_bytes": self.payload_bytes, **{flag: False for flag in FALSE_FLAGS}}


def make_server(bundle: VerifiedBundle, port: int = 0):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, content=b"", content_type="text/plain; charset=utf-8", **headers):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "same-origin")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; "
                             "script-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; "
                             "form-action 'none'; frame-ancestors 'none'")
            for key, value in headers.items():
                self.send_header(key.replace("_", "-"), value)
            self.end_headers()
            if self.command != "HEAD" and content:
                self.wfile.write(content)

        def serve(self):
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if (self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}"
                    or self.headers.get("Origin") not in (None, origin)):
                self.respond(403, b"Same-origin localhost requests only.\n")
                return
            try:
                parsed = urlsplit(self.path)
            except ValueError:
                self.respond(404, b"Asset unavailable.\n")
                return
            if (parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
                    or not parsed.path.startswith("/") or "%" in self.path
                    or "\\" in self.path or "\x00" in self.path):
                self.respond(404, b"Asset unavailable.\n")
                return
            if parsed.path == "/":
                self.respond(302, Location="/report/report.html")
                return
            name = parsed.path[1:]
            if name not in bundle.files:
                self.respond(404, b"Asset unavailable.\n")
                return
            try:
                content = bundle.read(name)
            except (OSError, ValueError):
                self.respond(409, b"Saved artifact no longer matches its verified release checksum.\n")
                return
            content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type == "application/json":
                content_type += "; charset=utf-8"
            self.respond(200, content, content_type)

        do_GET = serve
        do_HEAD = serve

        def do_POST(self):
            self.respond(405, b"This is a saved evidence viewer; no inference endpoint exists.\n", Allow="GET, HEAD")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use an unprivileged localhost port between 1024 and 65535")
    bundle = VerifiedBundle(args.bundle)
    print(json.dumps(bundle.summary(), sort_keys=True), flush=True)
    if args.verify_only:
        return
    server = make_server(bundle, args.port)
    print(f"Open http://127.0.0.1:{server.server_port}/report/report.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
