"""HTTP/source boundaries of the latest-case copilot; not model or hazard QA."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('fresh_copilot_server', SCRIPTS / 'serve_fresh_copilot.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


@pytest.fixture
def case_service(tmp_path, monkeypatch):
    water = tmp_path / 'water.json'; water.write_text('{"case":"synthetic_test_only"}')
    side = tmp_path / 'case-data.json'; side.write_text('{"count":3}')
    report = tmp_path / 'report.html'
    report.write_text('<html><main><div class="main-grid">Synthetic engineering test only</div></main></html>')
    image = tmp_path / 'optical-overview.png'; image.write_bytes(b'test-image')
    case = {'case_id': 'synthetic_test_only'}
    report_facts = tmp_path / 'report.json'
    report_facts.write_text(json.dumps({'case': case, 'counts': {'aoi_pixels': 1}, 'source_dates': {}}))
    web = tmp_path / 'web'; web.mkdir()
    (web / 'fresh-copilot-panel.html').write_text('<section id="fresh-copilot">Test copilot</section>')
    (web / 'fresh-copilot.js').write_text('/* test only */')
    binding = tmp_path / 'source-and-output-manifest.json'
    binding.write_text(json.dumps({'schema': 'akash-fresh-report-binding-v1', 'status': 'completed',
                                  'case': case,
                                  'inputs': [server.record(water)],
                                  'outputs': {'report.html': server.record(report),
                                              'report.json': server.record(report_facts),
                                              'optical-overview.png': server.record(image)}}))
    class Fact:
        def as_dict(self):
            return {'id': 'test_scope', 'en': 'Synthetic HTTP test only'}
    class FakeStore:
        facts = {'test_scope': Fact()}
        manifest = {'counts': {'aoi_pixels': 1}, 'source_dates': {}}
        source_bindings = [server.record(water), server.record(side)]
        @classmethod
        def from_paths(cls, *args):
            assert args == (water, None, None)
            return cls()
    FakeStore.case = case
    monkeypatch.setattr(server, 'FreshCaseFactStore', FakeStore)
    monkeypatch.setattr(server, 'WEB', web)
    service = server.CaseService(water=water, exposure=None, optical=None, report_binding=binding)
    return service, {'water': water, 'side': side, 'report': report, 'facts': report_facts, 'binding': binding, 'web': web}


@pytest.fixture
def endpoint(case_service):
    service, paths = case_service
    http = ThreadingHTTPServer(('127.0.0.1', 0), server.handler_for(service))
    thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
    try:
        yield 'http://127.0.0.1:' + str(http.server_address[1]), service, paths
    finally:
        http.shutdown(); http.server_close(); thread.join(timeout=2)


def fetch(endpoint, path='/', body=None, headers=None):
    request_headers = {'Content-Type': 'application/json'}
    request_headers.update(headers or {})
    raw = body.encode() if isinstance(body, str) else json.dumps(body).encode() if body is not None else None
    request = Request(endpoint + path, data=raw, headers=request_headers)
    try:
        with urlopen(request, timeout=3) as response:
            return response.status, response.read(), response.headers
    except HTTPError as response:
        return response.code, response.read(), response.headers


def test_current_report_and_asset_allowlist_do_not_call_model(endpoint, monkeypatch):
    base, _, _ = endpoint
    def forbidden(*args, **kwargs):
        pytest.fail('Viewing saved evidence cannot invoke the model')
    monkeypatch.setattr(server, 'ask_local_model', forbidden)
    status, body, headers = fetch(base)
    assert status == 200 and b'id="fresh-copilot"' in body and b'Synthetic engineering test' in body
    assert headers['Cache-Control'] == 'no-store'
    assert fetch(base, '/optical-overview.png')[1] == b'test-image'
    assert json.loads(fetch(base, '/api/facts')[1])['facts'][0]['id'] == 'test_scope'
    for path in ('/../../water.json', '/water.json', '/%2e%2e/side.json', '/report.html?x=1'):
        assert fetch(base, path)[0] == 404


def test_valid_question_uses_active_source_and_local_model(endpoint, monkeypatch):
    base, service, _ = endpoint
    calls = []
    def model(store, question, **options):
        calls.append((question, options))
        return {'status': 'answer', 'language': 'ne', 'answer': 'test', 'claims': []}
    monkeypatch.setattr(server, 'ask_local_model', model)
    status, raw, _ = fetch(base, '/api/ask', {'question': 'test source', 'language': 'ne'})
    answer = json.loads(raw)
    assert status == 200 and calls == [('test source', {'language': 'ne', 'endpoint': 'http://127.0.0.1:8782'})]
    assert answer['case_manifest_sources'] == service.source_inputs and answer['request_wall_seconds'] >= 0


@pytest.mark.parametrize('body', [
    {'question': ' ', 'language': 'en'}, {'question': 'x' * 1201, 'language': 'en'},
    {'question': 'q', 'language': 'hi'}, {'question': 'q', 'language': 'en', 'observations': {}},
    {'question': 123, 'language': 'en'},
    '{"question":"q","question":"other","language":"en"}',
])
def test_invalid_requests_never_call_model(endpoint, monkeypatch, body):
    base, _, _ = endpoint
    monkeypatch.setattr(server, 'ask_local_model', lambda *a, **k: pytest.fail('Invalid request reached model'))
    assert fetch(base, '/api/ask', body)[0] == 400


def test_cross_origin_nonloopback_and_oversized_requests_fail(endpoint, monkeypatch):
    base, _, _ = endpoint
    monkeypatch.setattr(server, 'ask_local_model', lambda *a, **k: pytest.fail('Rejected request reached model'))
    body = {'question': 'q', 'language': 'en'}
    assert fetch(base, '/api/ask', body, {'Origin': 'https://example.com'})[0] == 403
    assert fetch(base, '/api/ask', body, {'Host': 'example.com'})[0] == 403
    assert fetch(base, '/', headers={'Host': 'example.com'})[0] == 403
    assert fetch(base, '/api/ask', 'x' * 8193)[0] == 400
    assert fetch(base, '/api/ask', body, {'Content-Type': 'text/plain'})[0] == 400


@pytest.mark.parametrize('source', ['water', 'side', 'report'])
def test_changed_case_sources_disable_cached_content_and_answers(endpoint, monkeypatch, source):
    base, _, paths = endpoint
    paths[source].write_text('changed')
    monkeypatch.setattr(server, 'ask_local_model', lambda *a, **k: pytest.fail('Changed source reached model'))
    assert fetch(base, '/api/facts')[0] == 503
    assert fetch(base, '/api/ask', {'question': 'q', 'language': 'en'})[0] == 503


def test_source_change_during_model_call_does_not_display_stale_answer(endpoint, monkeypatch):
    base, _, paths = endpoint
    def model(*args, **kwargs):
        paths['side'].write_text('changed while model answered')
        return {'answer': 'must not be displayed'}
    monkeypatch.setattr(server, 'ask_local_model', model)
    status, raw, _ = fetch(base, '/api/ask', {'question': 'q', 'language': 'en'})
    assert status == 503 and 'answer' not in json.loads(raw)


def test_busy_and_missing_model_have_no_substitute_answers(endpoint, monkeypatch):
    base, service, _ = endpoint
    service.lock.acquire()
    try:
        assert fetch(base, '/api/ask', {'question': 'q', 'language': 'en'})[0] == 429
    finally:
        service.lock.release()
    def unavailable(*args, **kwargs):
        raise ConnectionError('missing model')
    monkeypatch.setattr(server, 'ask_local_model', unavailable)
    status, raw, _ = fetch(base, '/api/ask', {'question': 'q', 'language': 'en'})
    assert status == 503 and set(json.loads(raw)) == {'error'}


def test_report_must_bind_exact_current_manifests(case_service):
    _, paths = case_service
    manifest = json.loads(paths['binding'].read_text()); manifest['inputs'][0]['sha256'] = '0' * 64
    paths['binding'].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='exact active'):
        server.CaseService(water=paths['water'], exposure=None, optical=None, report_binding=paths['binding'])


@pytest.mark.parametrize('field', ['case', 'counts', 'source_dates'])
def test_displayed_report_facts_cannot_mix_cases_or_counts(case_service, field):
    _, paths = case_service
    facts = json.loads(paths['facts'].read_text()); facts[field] = {'changed': True}
    paths['facts'].write_text(json.dumps(facts))
    binding = json.loads(paths['binding'].read_text())
    binding['outputs']['report.json'] = server.record(paths['facts'])
    paths['binding'].write_text(json.dumps(binding))
    with pytest.raises(ValueError, match='dashboard facts'):
        server.CaseService(water=paths['water'], exposure=None, optical=None, report_binding=paths['binding'])


def test_report_binding_case_must_match(case_service):
    _, paths = case_service
    binding = json.loads(paths['binding'].read_text()); binding['case'] = {'case_id': 'another_case'}
    paths['binding'].write_text(json.dumps(binding))
    with pytest.raises(ValueError, match='binding case'):
        server.CaseService(water=paths['water'], exposure=None, optical=None, report_binding=paths['binding'])


@pytest.mark.parametrize('url', ['https://127.0.0.1:8782', 'http://example.com:8782',
                               'http://user@localhost:8782', 'http://localhost:8782/x'])
def test_remote_or_credentialed_model_endpoints_are_rejected(case_service, url):
    _, paths = case_service
    with pytest.raises(ValueError, match='loopback'):
        server.CaseService(water=paths['water'], exposure=None, optical=None, report_binding=paths['binding'], model_endpoint=url)
