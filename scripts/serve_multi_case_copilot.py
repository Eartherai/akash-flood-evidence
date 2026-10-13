"""Serve explicit Rasuwa/Kathmandu source clauses and actual local CPU Qwen."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit

from akash_flood.copilot import MODEL, _distinct_json_keys
from akash_flood.multi_case_copilot import MultiCaseFactStore, ask_multi_case_model, default_cases

ROOT = Path(__file__).resolve().parents[1]


def validate_question(body, cases):
    if not isinstance(body, dict) or set(body) != {"case", "question", "language"}:
        raise ValueError("Supply exactly case, question and language")
    if body["case"] not in cases or body["language"] not in ("en", "ne"):
        raise ValueError("Choose a supplied case and language")
    if not isinstance(body["question"], str) or not body["question"].strip() or len(body["question"]) > 1200:
        raise ValueError("Use a nonempty question of at most 1200 characters")
    return body["case"], body["question"], body["language"]


class MultiCaseService:
    def __init__(self, endpoint="http://127.0.0.1:8782", cases=None):
        activation_started = time.perf_counter()
        parsed = urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("Model endpoint must be plain loopback HTTP")
        self.endpoint = endpoint
        self.stores = {slug: MultiCaseFactStore.from_config(config) for slug, config in (cases or default_cases()).items()}
        self.lock = threading.Lock()
        self.html_path = ROOT / "web/multi-case-copilot.html"
        self.html = self.html_path.read_bytes()
        self.code_bindings = [next(iter(self.stores.values()))._helpers.record(path) for path in (
            self.html_path, Path(__file__), ROOT / "src/akash_flood/multi_case_copilot.py",
            ROOT / "src/akash_flood/focused_copilot.py", ROOT / "src/akash_flood/copilot.py")]
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.activation_wall_seconds = time.perf_counter() - activation_started

    def verify_code(self):
        helpers = next(iter(self.stores.values()))._helpers
        if any(helpers.record(Path(item["path"])) != item for item in self.code_bindings):
            raise ValueError("Copilot code changed; restart the source-bound service")

    def metadata(self):
        self.verify_code()
        for store in self.stores.values():
            store.verify_unchanged()
        return {"cases": [{"slug": slug, "label": store.config.label, "case": store.case,
            "facts": [f.as_dict() for f in store.facts.values() if not f.id.startswith("settlement_")],
            "report_url": f"/cases/{slug}/" + ("index.html" if store.config.supplement else "report.html"),
            "source_bindings": store.source_bindings, "verification": store.verification}
            for slug, store in self.stores.items()], "model": MODEL,
            "language_review": "Nepali source clauses are AI-authored; human review has not been performed",
            "evaluation_status": "Known development checks only; no blind relevance, language or hazard validation"}

    def ask(self, slug, question, language):
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("The CPU copilot is answering another question")
        started = time.perf_counter()
        try:
            self.verify_code()
            store = self.stores[slug]
            result = ask_multi_case_model(store, question, language=language, endpoint=self.endpoint)
            self.verify_code()
            result.update(question=question, request_wall_seconds=time.perf_counter() - started,
                created_at=datetime.now(timezone.utc).isoformat(), code_bindings=self.code_bindings)
            return result
        finally:
            self.lock.release()


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(70)

        def log_message(self, *_):
            pass

        def valid_host(self):
            port = self.server.server_address[1]
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}")

        def reply_bytes(self, raw, media, status=200):
            self.send_response(status)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)

        def reply(self, value, status=200):
            self.reply_bytes(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8", status)

        def do_GET(self):
            if not self.valid_host():
                self.reply({"error": "Use a loopback host"}, 403); return
            try:
                service.verify_code()
                if self.path == "/":
                    self.reply_bytes(service.html, "text/html; charset=utf-8"); return
                if self.path == "/api/cases":
                    self.reply(service.metadata()); return
                path = urlsplit(self.path).path.split("/")
                if len(path) == 4 and path[1] == "cases" and path[2] in service.stores:
                    store = service.stores[path[2]]
                    name = path[3]
                    if name not in store.assets:
                        self.reply({"error": "Not found"}, 404); return
                    item = store._bindings[str(store.assets[name].resolve())]
                    if store._helpers.record(store.assets[name]) != item:
                        raise ValueError("Report asset changed")
                    raw = store.assets[name].read_bytes()
                    import hashlib
                    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                        raise ValueError("Report asset changed while read")
                    media = {".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8", ".png": "image/png", ".svg": "image/svg+xml"}
                    self.reply_bytes(raw, media.get(Path(name).suffix, "application/octet-stream")); return
                self.reply({"error": "Not found"}, 404)
            except (ValueError, KeyError, OSError):
                self.reply({"error": "Saved evidence failed verification"}, 503)

        def do_POST(self):
            if not self.valid_host():
                self.reply({"error": "Use a loopback host"}, 403); return
            if self.path != "/api/ask":
                self.reply({"error": "Not found"}, 404); return
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.reply({"error": "Cross-origin requests are disabled"}, 403); return
            if self.headers.get("Transfer-Encoding") or self.headers.get_content_type() != "application/json":
                self.reply({"error": "Use bounded JSON"}, 400); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Request size exceeds bound")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request")
                slug, question, language = validate_question(json.loads(raw, object_pairs_hook=_distinct_json_keys), service.stores)
            except (ValueError, TypeError, KeyError, UnicodeError):
                self.reply({"error": "Invalid case/question/language request"}, 400); return
            try:
                self.reply(service.ask(slug, question, language))
            except BlockingIOError as error:
                self.reply({"error": str(error)}, 429)
            except Exception:
                self.reply({"error": "The local CPU model or pinned case evidence failed. No substitute answer was produced."}, 503)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8797)
    parser.add_argument("--model-endpoint", default="http://127.0.0.1:8782")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use an unprivileged loopback port")
    service = MultiCaseService(args.model_endpoint)
    if args.out:
        if args.out.exists():
            raise FileExistsError("Preserve prior runs; choose a new output directory")
        args.out.mkdir(parents=True)
        (args.out / "index.html").write_bytes(service.html)
        (args.out / "facts.json").write_text(json.dumps(service.metadata(), indent=2, ensure_ascii=False) + "\n")
        (args.out / "activation.json").write_text(json.dumps({"schema": "akash-multi-case-copilot-activation-v1",
            "created_at": service.started_at, "code": service.code_bindings,
            "activation_wall_seconds": service.activation_wall_seconds,
            "case_source_bindings": {slug: s.source_bindings for slug, s in service.stores.items()},
            "model": MODEL, "model_endpoint": args.model_endpoint,
            "cpu_only_launcher": "scripts/run_copilot_model.py --serve", "hazard_inference_run": False,
            "model_invoked_by_activation": False, "reference_labels_loaded": False,
            "language_review": "AI-authored Nepali; no human review"}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"url": f"http://127.0.0.1:{args.port}/", "cases": list(service.stores), "model_invoked": False}), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(service)).serve_forever()


if __name__ == "__main__":
    main()
