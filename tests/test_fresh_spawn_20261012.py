"""Actual fork-free fixture processes; no satellite/network/model execution."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import pytest
from akash_flood.fresh_spawn_20261012 import SpawnProcess, capture_command, spawn_run_command, stop_spawn_process

ROOT=Path(__file__).resolve().parents[1]


def test_native_parent_thread_spawn_has_an_independent_session(monkeypatch):
    import rasterio
    from pyproj import Transformer
    assert Transformer.from_crs('EPSG:4326','EPSG:32645',always_xy=True).transform(85.3,27.7)[0]>0
    # A regression must not quietly fall back to Python's fork-based Popen.
    def forbidden(*args,**kwargs):raise AssertionError('fork-based Popen used')
    monkeypatch.setattr('subprocess.Popen',forbidden)
    result={}
    def launch():
        process=SpawnProcess([sys.executable,'-c','import json,os;print(json.dumps([os.getpid(),os.getsid(0),os.getpgrp()]))'],cwd=ROOT)
        try:
            result['ids']=json.loads(process.stdout.read());result['returncode']=process.wait(5)
            result['error']=process.stderr.read().decode()
        finally:process.close_pipes()
    thread=threading.Thread(target=launch);thread.start();thread.join(10)
    assert not thread.is_alive() and result['returncode']==0 and not result['error']
    assert len(set(result['ids']))==1 and result['ids'][0]!=os.getpgrp()
    # Process-tree inspection must use the same fork-free path after warming.
    assert 'pid' not in capture_command(['/bin/ps','-axo','pid=,pgid=,stat=']).stdout[:3]


def test_explicit_native_inheritable_fd_is_closed_in_child_only(tmp_path):
    path=tmp_path/'private-input.txt';path.write_text('parent-only handle')
    with path.open('rb') as original:
        duplicate=fcntl.fcntl(original.fileno(),fcntl.F_DUPFD,80)
        os.set_inheritable(duplicate,True)
        try:
            code='import os,sys;\ntry:os.fstat(int(sys.argv[1]));print("LEAK")\nexcept OSError:print("CLOSED")'
            process=SpawnProcess([sys.executable,'-c',code,str(duplicate)])
            try:assert process.stdout.read()==b'CLOSED\n' and process.wait(3)==0
            finally:process.close_pipes()
            assert os.fstat(duplicate).st_size==len('parent-only handle')
        finally:os.close(duplicate)


def test_arguments_remain_literal_and_stdin_is_closed():
    literal='$(echo forbidden); arbitrary input'
    process=SpawnProcess([sys.executable,'-c','import json,sys;print(json.dumps([sys.argv[1],sys.stdin.read()]))',literal])
    try:assert json.loads(process.stdout.read())==[literal,''] and process.wait(3)==0
    finally:process.close_pipes()


def test_cwd_mismatch_is_refused_without_global_directory_change(tmp_path):
    original=Path.cwd()
    with pytest.raises(ValueError,match='working directory'):
        SpawnProcess([sys.executable,'-c','print(1)'],cwd=tmp_path)
    assert Path.cwd()==original


def test_stage_runner_preserves_actual_nonzero_exit_and_bounded_logs(tmp_path):
    stdout=tmp_path/'stdout.log';stderr=tmp_path/'stderr.log'
    result=spawn_run_command([sys.executable,'-c','import sys;print("real output");print("actual failure",file=sys.stderr);sys.exit(7)'],cwd=ROOT,environment={},stdout_path=stdout,stderr_path=stderr,timeout_seconds=3,log_cap=1000)
    assert result['exit_code']==7 and result['spawn_method']=='posix_spawn_setsid'
    assert stdout.read_text()=='real output\n' and stderr.read_text()=='actual failure\n'
    assert result['stdout_bytes']==len(stdout.read_bytes())


def test_stage_stdout_cap_stops_the_real_process(tmp_path):
    stdout=tmp_path/'stdout.log';stderr=tmp_path/'stderr.log'
    result=spawn_run_command([sys.executable,'-c','import time;print("x"*10000,flush=True);time.sleep(30)'],cwd=ROOT,environment={},stdout_path=stdout,stderr_path=stderr,timeout_seconds=3,log_cap=1000)
    assert result['status']=='log_limit' and stdout.stat().st_size==1000
    assert result['exit_code']!=0 and result['process_cleanup']['cleanup_confirmed']


def test_stage_timeout_preserves_reason_and_stops_process(tmp_path):
    result=spawn_run_command([sys.executable,'-c','import time;time.sleep(30)'],cwd=ROOT,environment={},stdout_path=tmp_path/'out.log',stderr_path=tmp_path/'err.log',timeout_seconds=.1,log_cap=1000)
    assert result['status']=='timeout' and 'deadline' in result['failure_reason']
    assert result['process_cleanup']['cleanup_confirmed'] and result['exit_code']!=0


def test_cleanup_kills_term_resistant_redirected_descendant_after_leader_exit(tmp_path):
    ready=tmp_path/'child-pid.txt'
    child='import pathlib,signal,time,os;signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path('+repr(str(ready))+').write_text(str(os.getpid()));time.sleep(30)'
    parent='import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",'+repr(child)+'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);time.sleep(30)'
    process=SpawnProcess([sys.executable,'-c',parent])
    deadline=time.monotonic()+5
    try:
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert ready.is_file()
        receipt=stop_spawn_process(process)
        assert receipt['cleanup_confirmed'] and process.returncode!=0
        status=capture_command(['/bin/ps','-axo','pid=,stat=']).stdout
        child_id=int(ready.read_text())
        matching=[line.split()[1] for line in status.splitlines() if len(line.split())==2 and int(line.split()[0])==child_id]
        assert not matching or matching[0].startswith('Z')
    finally:
        if process.poll() is None:stop_spawn_process(process)
        process.close_pipes()
