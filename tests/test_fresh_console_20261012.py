"""Real fixture subprocesses exercise input, origin and lifecycle boundaries.

These do not execute remote discovery, radar inference or any saved test split.
"""
from __future__ import annotations

import copy
import http.client
from http.server import ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest

from akash_flood.fresh_console_20261012 import ConsoleService, DEFAULTS, checked_payload, record

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fresh_console_server_test", ROOT / "scripts/serve_fresh_console_20261012.py")
server_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(server_module)


def payload(**extra):
    return copy.deepcopy(DEFAULTS) | extra


@pytest.mark.parametrize("change", [
    {"bbox": [85, 27, 84, 28]}, {"bbox": [85, 27, float('nan'), 28]},
    {"bbox": [True, 27, 85.4, 27.1]}, {"bbox": [85, 27, 86, 28]},
    {"bbox": [83.98, 27, 84.02, 27.04]}, {"bbox": [85, -.02, 85.1, .02]},
    {"event_date": "2026-02-30"}, {"osm_snapshot": "2026-08-26"},
    {"native_budget_mb": True}, {"native_budget_mb": 4101},
    {"optical_budget_mb": 101}, {"max_total_seconds": 7201},
    {"workflow": "../../scripts/arbitrary.py"}, {"output": "/tmp/escape"},
    {"event_date": "2026-08-26; touch /tmp/nope"},
])
def test_refuse_unbounded_ambiguous_or_injected_case_fields(change):
    with pytest.raises(ValueError):
        checked_payload(payload(**change))


@pytest.fixture
def fixture_service(tmp_path, monkeypatch):
    monkeypatch.setattr("akash_flood.fresh_console_20261012.importlib.metadata.version", lambda name: "1.7.2")
    prepared = tmp_path / "prepared.txt"; prepared.write_text("source fixture, no imagery\n")
    runner = tmp_path / "fixture_runner.py"
    runner.write_text('''import argparse, json, hashlib
from pathlib import Path

def rec(p):
 p=Path(p).resolve();return {"path":str(p),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
def parser():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--bbox',nargs=4);p.add_argument('--event-date');p.add_argument('--osm-snapshot');p.parse_args_orig=p.parse_args
 p.parse_args=lambda argv:p.parse_known_args(argv)[0]
 return p
def build_plan(args):
 return {"schema":"fixture-only","out":str(args.out),"source_runner":rec(__file__),"external_inputs":{"prepared":rec(Path(__file__).parent/'prepared.txt')},"case":{"bbox":args.bbox},"limits":{"max_total_seconds":20},"blockers":[],"stages":[{"name":"fixture_stage","network_scope":"none; fixture process only","source_code":[rec(__file__)],"artifacts":[]}],"full_track_b_requirements_satisfied":False}
''')
    worker = tmp_path / "fixture_worker.py"
    worker.write_text('''import json, pathlib, sys, time, subprocess, os
p=pathlib.Path(sys.argv[-1]);plan=json.loads(p.read_text());out=pathlib.Path(plan['out']);out.mkdir()
mode=(p.parent.parent.parent/'mode.txt').read_text() if (p.parent.parent.parent/'mode.txt').exists() else 'success'
print('fixture real worker started',flush=True)
if mode=='cancel':
 child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
 (out/'child-pid.txt').write_text(str(child.pid));(out/'ready.txt').write_text('ready')
 time.sleep(60)
if mode=='cancel-stubborn':
 child_code="import os,pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(os.environ['FIXTURE_GRANDPID']).write_text(str(os.getpid())); time.sleep(60)"
 stage_code="import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"+repr(child_code)+"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); time.sleep(60)"
 env=os.environ.copy();env['FIXTURE_GRANDPID']=str(out/'grand-pid.txt')
 child=subprocess.Popen([sys.executable,'-c',stage_code],env=env,start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 (out/'child-pid.txt').write_text(str(child.pid))
 while not (out/'grand-pid.txt').exists():time.sleep(.01)
 (out/'ready.txt').write_text('ready');time.sleep(60)
if mode=='failure':
 (out/'execution-receipt.json').write_text(json.dumps({'status':'failed','stopped_at_stage':'fixture_stage','stages':[{'name':'fixture_stage','status':'failed','failure_reason':'fixture stage deliberate error'}]}))
 print('actual fixture failure',file=sys.stderr,flush=True);sys.exit(7)
if mode=='logcap':
 print('x'*11000000,flush=True);time.sleep(10)
(out/'execution-receipt.json').write_text(json.dumps({'status':'completed','stages':[{'name':'fixture_stage','status':'completed','artifacts':[]}],'full_track_b_requirements_satisfied':False}))
''')
    service = ConsoleService(output=tmp_path / "output", runners={"water":runner,"multimodal":runner}, worker=worker, python=sys.executable)
    yield service, prepared, tmp_path
    service.close()


def wait_terminal(service, job_id):
    deadline = time.monotonic()+15
    while time.monotonic()<deadline:
        job=service.snapshot(job_id)
        if job['status']!='running' and service.active is None:
            return job
        time.sleep(.03)
    raise AssertionError("fixture process did not reach terminal lifecycle state")


def test_plan_is_fresh_and_inert_then_runs_actual_process_once(fixture_service):
    service, _, _ = fixture_service
    job = service.make_plan(payload())
    assert job['status']=='planned' and job['plan_subprocesses_run']==job['plan_network_requests']==0
    assert not Path(job['output_path']).exists()
    assert Path(job['output_path']).parent.parent == service.output
    service.start(job['id'])
    result=wait_terminal(service, job['id'])
    assert result['status']=='completed' and result['exit_code']==0
    assert 'fixture real worker started' in result['logs']['stdout']
    assert result['receipt']['full_track_b_requirements_satisfied'] is False
    with pytest.raises(ValueError, match="at most once"):
        service.start(job['id'])
    assert service.make_plan(payload())['id'] != job['id']


def test_changed_prepared_source_cannot_execute_the_old_plan(fixture_service):
    service, prepared, _ = fixture_service
    job=service.make_plan(payload());prepared.write_text("mutated source fixture")
    with pytest.raises(ValueError, match="changed"):
        service.start(job['id'])
    assert service.snapshot(job['id'])['run_attempts']==0
    assert not Path(job['output_path']).exists()


def test_plan_or_identity_path_escape_is_refused(fixture_service):
    service, _, _ = fixture_service
    with pytest.raises(KeyError):
        service.snapshot('../outside')
    with pytest.raises(ValueError):
        service.make_plan(payload(output='/tmp/escape'))
    job=service.make_plan(payload())
    path=Path(job['console_plan']['path']);value=json.loads(path.read_text());value['out']='/tmp/nope';path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="plan changed"):
        service.start(job['id'])


def test_failure_preserves_actual_exit_receipt_error_and_no_retry(fixture_service):
    service, _, tmp = fixture_service
    (tmp/'mode.txt').write_text('failure')
    job=service.make_plan(payload());service.start(job['id']);result=wait_terminal(service,job['id'])
    assert result['status']=='failed' and result['exit_code']==7
    assert result['receipt']['stopped_at_stage']=='fixture_stage'
    assert 'actual fixture failure' in result['logs']['stderr']
    with pytest.raises(ValueError):service.start(job['id'])


def test_cancel_stops_actual_worker_and_independently_grouped_child(fixture_service):
    service, _, tmp = fixture_service
    (tmp/'mode.txt').write_text('cancel')
    job=service.make_plan(payload());service.start(job['id'])
    deadline=time.monotonic()+5
    while not (Path(job['output_path'])/'ready.txt').exists() and time.monotonic()<deadline:
        time.sleep(.03)
    assert (Path(job['output_path'])/'ready.txt').is_file()
    child=int((Path(job['output_path'])/'child-pid.txt').read_text())
    with pytest.raises(BlockingIOError):service.make_plan(payload())
    with pytest.raises(BlockingIOError):service.start(job['id'])
    service.cancel(job['id']);result=wait_terminal(service,job['id'])
    assert result['status']=='cancelled' and result['exit_code'] != 0
    # A killed but not yet reaped child may appear briefly as a zombie. It must
    # never remain an executing process after cancellation.
    import subprocess
    status=subprocess.run(['ps','-o','stat=','-p',str(child)],capture_output=True,text=True).stdout.strip()
    assert not status or status.startswith('Z')


def test_stdout_cap_stops_process_and_records_truncation(fixture_service):
    service, _, tmp = fixture_service
    (tmp/'mode.txt').write_text('logcap')
    job=service.make_plan(payload());service.start(job['id']);result=wait_terminal(service,job['id'])
    assert result['status']=='failed' and 'cap' in result['failure_reason']
    assert Path(result['console_logs']['stdout']['path']).stat().st_size==10_000_000


def test_loopback_host_origin_csrf_and_json_transport_guard_real_http(fixture_service):
    service, _, _ = fixture_service
    server=ThreadingHTTPServer(('127.0.0.1',0),server_module.handler_for(service))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    port=server.server_address[1]
    def request(method,path,body=None,**extra):
        conn=http.client.HTTPConnection('127.0.0.1',port,timeout=3)
        headers={'Host':f'127.0.0.1:{port}'}|extra
        if body is not None:
            body=json.dumps(body).encode();headers.setdefault('Content-Type','application/json')
        conn.request(method,path,body=body,headers=headers)
        res=conn.getresponse();data=res.read();status=res.status;conn.close();return status,data
    try:
        assert request('GET','/api/session',Host='evil.example')[0]==403
        assert request('GET','/api/session',**{'Sec-Fetch-Site':'cross-site'})[0]==403
        assert request('POST','/api/plan',payload())[0]==403
        valid={'Origin':f'http://127.0.0.1:{port}','X-Akash-CSRF':service.token}
        assert request('POST','/api/plan',payload(),**(valid|{'Origin':'https://evil.example'}))[0]==403
        assert request('POST','/api/plan',payload(),**(valid|{'Content-Type':'text/plain'}))[0]==400
        code,body=request('POST','/api/plan',payload(),**valid)
        assert code==201 and json.loads(body)['status']=='planned'
        assert request('GET','/../../pyproject.toml')[0]==404
        assert request('GET','/reports/../pyproject.toml')[0]==404
        assert service.active is None
    finally:
        server.shutdown();server.server_close();thread.join(timeout=3)


def test_missing_pinned_environment_is_an_offline_blocker(fixture_service, monkeypatch):
    service, _, _ = fixture_service
    monkeypatch.setattr("akash_flood.fresh_console_20261012.importlib.metadata.version", lambda name:"1.8.0")
    job=service.make_plan(payload())
    assert any('scikit-learn==1.7.2' in item for item in job['plan']['blockers'])
    with pytest.raises(ValueError, match="blocked"):
        service.start(job['id'])
    assert job['plan_network_requests']==0 and not Path(job['output_path']).exists()


def test_console_restart_preserves_job_without_resuming(fixture_service):
    service, _, _ = fixture_service
    job=service.make_plan(payload())
    state_path=Path(job['console_plan']['path']).parent/'console-status.json'
    state=json.loads(state_path.read_text());state.update(status='running',run_attempts=1)
    state_path.write_text(json.dumps(state))
    restarted=ConsoleService(output=service.output,runners=service.runners,worker=service.worker)
    assert restarted.active is None
    assert restarted.snapshot(job['id'])['status']=='interrupted'
    assert not Path(job['output_path']).exists()
    with pytest.raises(ValueError):restarted.start(job['id'])


def test_combined_report_requires_completed_stage_and_exact_output_identity(fixture_service):
    service, _, _ = fixture_service
    job=service.make_plan(payload());service.start(job['id']);wait_terminal(service,job['id'])
    out=Path(job['output_path']);report=out/'combined-report';report.mkdir()
    index=report/'index.html';index.write_text('<!doctype html><title>Actual fixture report</title>')
    manifest=report/'manifest.json';manifest.write_text(json.dumps({'schema':'akash-fresh-combined-report-v1','status':'completed','outputs':{'index.html':record(index)}}))
    receipt_path=out/'execution-receipt.json';receipt=json.loads(receipt_path.read_text())
    receipt['stages'].append({'name':'build_combined_case_report','status':'starting','artifacts':[record(manifest)|{'kind':'fresh_combined_report'}]})
    receipt_path.write_text(json.dumps(receipt))
    assert 'combined_report_url' not in service.snapshot(job['id'])
    receipt['stages'][-1]['status']='completed';receipt_path.write_text(json.dumps(receipt))
    assert service.snapshot(job['id'])['combined_report_url'].endswith('/index.html')
    assert service.report_asset(job['id'],'index.html')==index.read_bytes()
    assert service.report_asset(job['id'],'manifest.json')==manifest.read_bytes()
    index.write_text('<!doctype html>changed report')
    with pytest.raises(ValueError, match="changed"):
        service.report_asset(job['id'],'index.html')


def test_completed_report_cannot_expose_an_external_or_symlink_file(fixture_service):
    service, _, tmp = fixture_service
    job=service.make_plan(payload());service.start(job['id']);wait_terminal(service,job['id'])
    out=Path(job['output_path']);report=out/'combined-report';report.mkdir()
    external=tmp/'secret.html';external.write_text('not a report output')
    manifest=report/'manifest.json';manifest.write_text(json.dumps({'schema':'akash-fresh-combined-report-v1','status':'completed','outputs':{'index.html':record(external)}}))
    receipt_path=out/'execution-receipt.json';receipt=json.loads(receipt_path.read_text())
    receipt['stages'].append({'name':'build_combined_case_report','status':'completed','artifacts':[record(manifest)|{'kind':'fresh_combined_report'}]})
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):service.report_asset(job['id'],'index.html')
    external_record=record(external);linked=report/'index.html';linked.symlink_to(external)
    external_record['path']=str(linked)
    manifest.write_text(json.dumps({'schema':'akash-fresh-combined-report-v1','status':'completed','outputs':{'index.html':external_record}}))
    receipt['stages'][-1]['artifacts']=[record(manifest)|{'kind':'fresh_combined_report'}];receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):service.report_asset(job['id'],'index.html')


def test_cancel_cleans_term_ignoring_descendant_after_worker_and_stage_exit(fixture_service):
    service, _, tmp = fixture_service
    (tmp/'mode.txt').write_text('cancel-stubborn')
    job=service.make_plan(payload());service.start(job['id'])
    out=Path(job['output_path']);deadline=time.monotonic()+5
    while not (out/'ready.txt').exists() and time.monotonic()<deadline:time.sleep(.02)
    assert (out/'ready.txt').is_file()
    grand=int((out/'grand-pid.txt').read_text())
    service.cancel(job['id']);result=wait_terminal(service,job['id'])
    assert result['status']=='cancelled'
    import subprocess
    status=subprocess.run(['ps','-o','stat=','-p',str(grand)],capture_output=True,text=True).stdout.strip()
    assert not status or status.startswith('Z')


@pytest.mark.parametrize('failure', ['log_open', 'log_hash', 'status_save'])
def test_io_failure_never_strands_active_job(fixture_service, monkeypatch, failure):
    service, _, _ = fixture_service
    job=service.make_plan(payload())
    if failure=='log_open':
        original=Path.open
        def broken(path,*args,**kwargs):
            if path.name=='console.stdout.log':raise OSError('fixture disk full while opening logs')
            return original(path,*args,**kwargs)
        monkeypatch.setattr(Path,'open',broken)
    elif failure=='log_hash':
        import akash_flood.fresh_console_20261012 as console
        original=console.record
        def broken(path):
            if Path(path).name.startswith('console.') and str(path).endswith('.log'):
                raise OSError('fixture log hashing unavailable')
            return original(path)
        monkeypatch.setattr(console,'record',broken)
    else:
        original=service._save
        def broken(value):
            if value.get('worker_pid'):raise OSError('fixture receipt disk full')
            return original(value)
        monkeypatch.setattr(service,'_save',broken)
    service.start(job['id']);result=wait_terminal(service,job['id'])
    assert service.active is None and result['status']=='failed'
    assert result.get('failure_reason') or result.get('persistence_error')


def test_trusted_native_cache_directory_does_not_replace_runner_identity(fixture_service):
    service, _, temporary = fixture_service
    cache = temporary / 'native-cache'
    cache.mkdir()
    service.trusted_inputs = {'source-cache': cache}
    planned = service.make_plan(payload())
    assert planned['status'] == 'planned'
    assert planned['runner']['path'] == str(Path(service.runners['water']).resolve())
    assert planned['runner']['path'] != str(cache)
    assert planned['trusted_startup_reuse']['source-cache'] == str(cache)
    assert planned['run_attempts'] == 0


def test_completed_large_route_facts_within_report_cap_are_served_exactly(fixture_service):
    service, _, _ = fixture_service
    job = service.make_plan(payload())
    service.start(job['id'])
    wait_terminal(service, job['id'])
    out = Path(job['output_path'])
    report = out / 'combined-report'
    report.mkdir()
    facts = report / 'case-facts.json'
    facts.write_bytes(b'{"source":"fixture"}' + b' ' * 20_000_001)
    manifest = report / 'manifest.json'
    manifest.write_text(json.dumps({'schema':'akash-fresh-combined-report-v1','status':'completed',
        'outputs':{'case-facts.json':record(facts)}}))
    receipt_path = out / 'execution-receipt.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['stages'].append({'name':'build_combined_case_report','status':'completed',
        'artifacts':[record(manifest)|{'kind':'fresh_combined_report'}]})
    receipt_path.write_text(json.dumps(receipt))
    assert service.report_asset(job['id'],'case-facts.json') == facts.read_bytes()
