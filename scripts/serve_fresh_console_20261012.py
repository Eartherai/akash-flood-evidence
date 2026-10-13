"""Local area/date console. Default bind: http://127.0.0.1:8825/ .

Use the same pinned project environment as run_fresh_case.py. No satellite
acquisition is started by serving this page or creating an offline plan.
"""
from __future__ import annotations

import argparse
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import signal
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from akash_flood.fresh_console_20261012 import (ConsoleService, OUTPUT, execute_worker, unique_pairs)

WEB = ROOT / "web/fresh-console-20261012"
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8")}
CSP = "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *_):
            pass

        def host_ok(self):
            port = self.server.server_address[1]
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def site_ok(self):
            return self.headers.get("Sec-Fetch-Site", "none") in ("none", "same-origin")

        def reply(self, body, status=200, media="application/json; charset=utf-8", report=False):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline' 'self'; img-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'" if report else CSP)
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def parsed_path(self):
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
                raise ValueError("Use a plain local path without URL parameters")
            return parsed.path

        def do_GET(self):
            if not self.host_ok() or not self.site_ok():
                self.reply({"error": "Loopback Host and same-origin request required"}, 403)
                return
            try:
                path = self.parsed_path()
                if path in ASSETS:
                    filename, media = ASSETS[path]
                    self.reply((WEB / filename).read_bytes(), media=media)
                elif path == "/api/session":
                    self.reply(service.capabilities())
                elif path == "/api/jobs":
                    self.reply(service.listing())
                elif path.startswith("/reports/"):
                    parts = path.split("/", 3)
                    if len(parts) != 4:
                        raise KeyError("Unavailable generated report")
                    media = mimetypes.guess_type(parts[3])[0] or "application/octet-stream"
                    self.reply(service.report_asset(parts[2], parts[3]), media=media, report=True)
                elif path.startswith("/api/jobs/") and path.count("/") == 3:
                    self.reply(service.snapshot(path.rsplit("/", 1)[1]))
                else:
                    self.reply({"error": "Not found"}, 404)
            except KeyError:
                self.reply({"error": "Unknown case ID"}, 404)
            except (ValueError, OSError) as error:
                self.reply({"error": str(error)}, 400)

        def do_POST(self):
            origin = "http://" + self.headers.get("Host", "")
            if (not self.host_ok() or not self.site_ok() or self.headers.get("Origin") != origin
                    or self.headers.get("X-Akash-CSRF") != service.token):
                self.reply({"error": "Same-origin request and local session token required"}, 403)
                return
            try:
                if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                    raise ValueError("Use a bounded JSON request body")
                length = self.headers.get("Content-Length", "")
                if not length.isdigit() or not 1 <= int(length) <= 8192:
                    raise ValueError("JSON request body must be 1–8192 bytes")
                raw = self.rfile.read(int(length))
                if len(raw) != int(length):
                    raise ValueError("Incomplete JSON request")
                body = json.loads(raw, object_pairs_hook=unique_pairs,
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
                path = self.parsed_path()
                if path == "/api/plan":
                    self.reply(service.make_plan(body), 201)
                elif path in ("/api/run", "/api/cancel"):
                    if not isinstance(body, dict) or set(body) != {"id"}:
                        raise ValueError("Supply only a server-generated case ID")
                    result = service.start(body["id"]) if path == "/api/run" else service.cancel(body["id"])
                    self.reply(result, 202)
                else:
                    self.reply({"error": "Not found"}, 404)
            except BlockingIOError as error:
                self.reply({"error": str(error)}, 409)
            except KeyError:
                self.reply({"error": "Unknown case ID"}, 404)
            except (ValueError, OSError, TypeError) as error:
                self.reply({"error": str(error)}, 400)

        def do_OPTIONS(self):
            self.reply({"error": "Cross-origin API access is disabled"}, 403)
    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8825)
    p.add_argument("--source-cache", type=Path, help="Trusted local native-source cache; every selected identity/hash is still verified")
    p.add_argument("--reuse-dem-tile", type=Path, help="Bind the preparer's exact checksum-matched default source tile")
    p.add_argument('--osm',type=Path,help='Trusted pre-event cached OSM; exact historical provenance checked by runner')
    p.add_argument('--osm-provenance',type=Path,help='Required provenance for trusted historical extract; no current-data fallback')
    p.add_argument("--activation", type=Path, help="Trusted proved native activation.json from this source extraction; checksum checked without Java/model execution")
    p.add_argument("--requirements", type=Path, help="Trusted separately supplied organizer text; required with --activation and bound into each plan")
    p.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    # Trusted main-process startup only; no server thread changes cwd.
    os.chdir(ROOT)
    if args.worker_plan:
        # Parent cancellation sends SIGINT. SIGTERM must also unwind the stage
        # runner's finally block rather than abandoning its child process group.
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        execute_worker(args.worker_plan)
        return
    if not 1024 <= args.port <= 65535:
        p.error("Use an unprivileged loopback port")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if OUTPUT.is_symlink():
        raise ValueError("Dedicated output root cannot be a symlink")
    lock = (OUTPUT / ".console.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    trusted_inputs = {}
    if args.source_cache:
        trusted_inputs["source-cache"] = args.source_cache.resolve(strict=True)
    if args.reuse_dem_tile:
        trusted_inputs["reuse-dem-tile"] = args.reuse_dem_tile.resolve(strict=True)
    if bool(args.osm)!=bool(args.osm_provenance):
        p.error('Supply both --osm and --osm-provenance')
    if args.osm:
        trusted_inputs['osm']=args.osm.resolve(strict=True)
        trusted_inputs['osm-provenance']=args.osm_provenance.resolve(strict=True)
    service = ConsoleService(trusted_inputs=trusted_inputs, activation=args.activation,
        requirements=args.requirements)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(service))
    server.daemon_threads = True
    print(f"Akash local execution console: http://127.0.0.1:{args.port}/", flush=True)
    print("Offline planning only until an explicit Run this plan action. Full Track B validation remains false.", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
        server.server_close()
        lock.close()


if __name__ == "__main__":
    main()
