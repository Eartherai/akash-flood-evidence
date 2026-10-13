"""Serve one verified saved case and a real, loopback-only CPU copilot.

This integrates immutable case outputs. It neither reruns hazard inference nor
imports evaluation products. A missing or invalid model returns an error; there
is no template answer substituted for an AI call.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from akash_flood.copilot import MODEL, ask_local_model, _distinct_json_keys
from akash_flood.fresh_copilot import FreshCaseFactStore
from build_fresh_report import bound, read_json, record

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'web'


def validate_question(body):
    if not isinstance(body, dict) or set(body) != {'question', 'language'}:
        raise ValueError('Supply only question and language')
    if (not isinstance(body['question'], str) or not body['question'].strip()
            or len(body['question']) > 1200 or body['language'] not in ('en', 'ne')):
        raise ValueError('Use a question of 1–1200 characters and language en or ne')
    return body['question'], body['language']


class CaseService:
    def __init__(self, *, water, exposure, optical, report_binding, model_endpoint='http://127.0.0.1:8782'):
        self.paths = tuple(Path(p).resolve() if p is not None else None for p in (water, exposure, optical))
        self.store = FreshCaseFactStore.from_paths(*self.paths)
        self._source_inputs = [record(p) for p in self.paths if p is not None]
        self.endpoint = model_endpoint
        parsed = urlsplit(model_endpoint)
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ('', '/')):
            raise ValueError('Model endpoint must be a plain, unauthenticated loopback HTTP address')
        self.lock = threading.Lock()
        report_binding = Path(report_binding).resolve()
        manifest = read_json(report_binding)
        if manifest.get('schema') != 'akash-fresh-report-binding-v1' or manifest.get('status') != 'completed':
            raise ValueError('A completed saved report binding is required')
        actual_sources = [record(p) for p in self.paths if p is not None]
        if manifest['inputs'] != actual_sources:
            raise ValueError('Dashboard report must bind the exact active copilot sources')
        if manifest.get('case') != self.store.case:
            raise ValueError('Dashboard binding case differs from the active case')
        self.assets = {}
        for name, item in manifest['outputs'].items():
            if Path(name).name != name:
                raise ValueError('Report asset names must be simple filenames')
            self.assets[name] = bound(item, report_binding.parent)
        self.assets['source-and-output-manifest.json'] = report_binding
        self.assets['fresh-copilot.js'] = WEB / 'fresh-copilot.js'
        panel_path = WEB / 'fresh-copilot-panel.html'
        self.asset_records = {name: record(path) for name, path in self.assets.items()}
        self.asset_bytes = {name: self.read_pinned(path, self.asset_records[name]) for name, path in self.assets.items()}
        report_facts = json.loads(self.asset_bytes['report.json'], object_pairs_hook=_distinct_json_keys)
        if (report_facts.get('case') != self.store.case
                or report_facts.get('counts') != self.store.manifest['counts']
                or report_facts.get('source_dates') != self.store.manifest['source_dates']):
            raise ValueError('Displayed dashboard facts differ from the active case evidence')
        html = self.asset_bytes['report.html'].decode()
        marker = '<div class="main-grid">'
        if marker not in html or 'id="fresh-copilot"' in html:
            raise ValueError('Expected the preserved case dashboard layout')
        panel_record = record(panel_path)
        panel = self.read_pinned(panel_path, panel_record).decode()
        self.html = html.replace(marker, panel + '\n' + marker, 1).encode()
        self.bindings = list(self.asset_records.values()) + [panel_record] + self.store.source_bindings + [
            record(Path(__file__)), record(ROOT / 'src/akash_flood/copilot.py')]
        self.started_at = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def read_pinned(path, item):
        raw = path.read_bytes()
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('Source changed while its displayed bytes were being read')
        return raw

    def verify_unchanged(self):
        for item in self.bindings:
            if record(Path(item['path'])) != item:
                raise ValueError('A saved dashboard source changed; restart after a verified rebuild')
        for path in self.paths:
            if path is not None:
                expected = next(item for item in self.source_inputs if item['path'] == str(path))
                if record(path) != expected:
                    raise ValueError('A copilot case manifest changed; restart after verification')

    @property
    def source_inputs(self):
        # Store construction has already validated the full case. The report's
        # inputs preserve the exact initial manifest identity for later checks.
        return self._source_inputs

    def ask(self, question, language):
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError('The CPU copilot is answering another question; try again shortly')
        started = time.perf_counter()
        try:
            self.verify_unchanged()
            # Revalidate/recount the immutable input evidence before every
            # model call; changed rasters or source bindings cannot be hidden
            # behind a cached fact store.
            store = FreshCaseFactStore.from_paths(*self.paths)
            answer = ask_local_model(store, question, language=language, endpoint=self.endpoint)
            self.verify_unchanged()
            answer['request_wall_seconds'] = time.perf_counter() - started
            answer['question'] = question
            answer['case_manifest_sources'] = self.source_inputs
            answer['model'] = MODEL
            return answer
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
            host = self.headers.get('Host', '')
            port = self.server.server_address[1]
            return host in (f'127.0.0.1:{port}', f'localhost:{port}', f'[::1]:{port}')

        def reply_bytes(self, body, media, status=200):
            self.send_response(status)
            self.send_header('Content-Type', media)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def reply(self, value, status=200):
            self.reply_bytes(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                             'application/json; charset=utf-8', status)

        def do_GET(self):
            if not self.valid_host():
                self.reply({'error': 'Use a loopback host'}, 403); return
            try:
                service.verify_unchanged()
                if self.path in ('/', '/report.html'):
                    self.reply_bytes(service.html, 'text/html; charset=utf-8'); return
                if self.path == '/api/facts':
                    self.reply({'facts': [f.as_dict() for f in service.store.facts.values()],
                                'sources': service.source_inputs, 'model': MODEL,
                                'language_review': 'Nepali has not received human review'}); return
                name = self.path.removeprefix('/')
                if name not in service.assets:
                    self.reply({'error': 'Not found'}, 404); return
                media = {'png': 'image/png', 'svg': 'image/svg+xml', 'js': 'text/javascript; charset=utf-8',
                         'json': 'application/json; charset=utf-8', 'html': 'text/html; charset=utf-8'}
                self.reply_bytes(service.asset_bytes[name], media.get(Path(name).suffix[1:], 'application/octet-stream'))
            except (OSError, ValueError, KeyError):
                self.reply({'error': 'Saved sources failed verification; no case content was returned'}, 503)

        def do_POST(self):
            if not self.valid_host():
                self.reply({'error': 'Use a loopback host'}, 403); return
            if self.path != '/api/ask':
                self.reply({'error': 'Not found'}, 404); return
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                self.reply({'error': 'Cross-origin requests are disabled'}, 403); return
            if (self.headers.get('Transfer-Encoding') or self.headers.get_content_type() != 'application/json'):
                self.reply({'error': 'Use a bounded JSON request'}, 400); return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8192:
                    raise ValueError('Request size must be 1–8192 bytes')
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError('Incomplete request body')
                question, language = validate_question(json.loads(raw, object_pairs_hook=_distinct_json_keys))
            except (ValueError, KeyError, TypeError, UnicodeError):
                self.reply({'error': 'Invalid question request'}, 400); return
            try:
                self.reply(service.ask(question, language))
            except BlockingIOError as error:
                self.reply({'error': str(error)}, 429)
            except Exception:
                self.reply({'error': 'The local model or saved evidence failed verification. Start the pinned CPU model and verify the case files.'}, 503)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--water-manifest', type=Path, required=True)
    parser.add_argument('--exposure-manifest', type=Path)
    parser.add_argument('--optical-manifest', type=Path)
    parser.add_argument('--report-binding', type=Path, required=True)
    parser.add_argument('--model-endpoint', default='http://127.0.0.1:8782')
    parser.add_argument('--port', type=int, default=8789)
    parser.add_argument('--out', type=Path, help='Save a fresh activation record and exact served dashboard')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Use an unprivileged loopback port')
    service = CaseService(water=args.water_manifest, exposure=args.exposure_manifest,
                          optical=args.optical_manifest, report_binding=args.report_binding,
                          model_endpoint=args.model_endpoint)
    if args.out is not None:
        if args.out.exists():
            raise FileExistsError('Use a fresh activation output path')
        args.out.mkdir(parents=True)
        (args.out / 'report.html').write_bytes(service.html)
        (args.out / 'facts.json').write_text(json.dumps({'case': service.store.case,
            'facts': [f.as_dict() for f in service.store.facts.values()]}, indent=2, ensure_ascii=False) + '\n')
        (args.out / 'activation.json').write_text(json.dumps({
            'schema': 'akash-fresh-copilot-activation-v1', 'created_at': service.started_at,
            'source_inputs': service.source_inputs, 'verified_case_bindings': service.store.source_bindings,
            'dashboard_bindings': service.bindings,
            'code': [record(Path(__file__)), record(ROOT / 'src/akash_flood/copilot.py')],
            'rendered_dashboard': record(args.out / 'report.html'),
            'facts': record(args.out / 'facts.json'), 'model': MODEL,
            'model_endpoint': args.model_endpoint, 'listening_url': f'http://127.0.0.1:{args.port}/',
            'model_invoked_by_activation': False, 'hazard_inference_run': False,
            'reference_products_loaded': False,
        }, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'url': f'http://127.0.0.1:{args.port}/', 'source_inputs': service.source_inputs,
                      'model_training_run': False, 'hazard_inference_run': False,
                      'evaluation_products_loaded': False}, indent=2), flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(service)).serve_forever()


if __name__ == '__main__':
    main()
