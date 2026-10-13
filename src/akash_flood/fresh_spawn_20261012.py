"""Fork-free bounded process launch for the new local Akash workflow.

Darwin PROJ/SQLite atfork handlers can crash a multithreaded parent before exec.
This adapter uses os.posix_spawn directly, including setsid and explicit file
closure, instead of changing frozen inference code or suppressing native errors.
"""
from __future__ import annotations

import os
from pathlib import Path
import selectors
import signal
import subprocess
import threading
import time
from types import SimpleNamespace


def _open_fds():
    """Snapshot descriptors for explicit child-only close actions.

    Python-created descriptors are also CLOEXEC. During the console's launch
    there is no concurrent native planning, so any concurrently created HTTP
    descriptors retain their normal Python CLOEXEC protection.
    """
    directory = "/dev/fd" if Path("/dev/fd").is_dir() else "/proc/self/fd"
    if not Path(directory).is_dir():
        raise RuntimeError("Fork-free spawn requires an open-descriptor inventory")
    result = set()
    for name in os.listdir(directory):
        if name.isdigit():
            fd = int(name)
            try:
                os.fstat(fd)
            except OSError:
                continue
            result.add(fd)
    return result


class SpawnProcess:
    """Popen-like handles, with one independently signallable POSIX session."""
    def __init__(self, argv, *, env=None, cwd=None):
        if not isinstance(argv, (list, tuple)) or not argv or any(not isinstance(arg, str) or "\x00" in arg for arg in argv):
            raise ValueError("Spawn requires a nonempty string argument vector")
        if not Path(argv[0]).is_absolute():
            raise ValueError("Spawn executable must have an absolute trusted path")
        if cwd is not None and Path(cwd).resolve() != Path.cwd().resolve():
            raise ValueError("Spawn cannot change a server thread's working directory; start the trusted worker at its declared cwd")
        self.args = list(argv)
        self.returncode = None
        self._wait_lock = threading.Lock()
        self.stdout = self.stderr = None
        opened = []
        try:
            null = os.open(os.devnull, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)); opened.append(null)
            rout, wout = os.pipe(); opened += [rout, wout]
            rerr, werr = os.pipe(); opened += [rerr, werr]
            for fd in opened:
                os.set_inheritable(fd, False)
            actions = [(os.POSIX_SPAWN_DUP2, null, 0), (os.POSIX_SPAWN_DUP2, wout, 1), (os.POSIX_SPAWN_DUP2, werr, 2)]
            # CLOEXEC already protects all noninheritable descriptors. Only
            # native inheritable handles and our stable private pipe originals
            # need explicit CLOSE actions. Closing a concurrently retired
            # Python HTTP/file descriptor can make Darwin spawn fail EBADF.
            inherited = set()
            for fd in _open_fds():
                if fd > 2:
                    try:
                        if os.get_inheritable(fd): inherited.add(fd)
                    except OSError:
                        continue
            actions += [(os.POSIX_SPAWN_CLOSE, fd) for fd in sorted(inherited | set(opened)) if fd > 2]
            resets = tuple(getattr(signal, name) for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ") if hasattr(signal, name))
            self.pid = os.posix_spawn(argv[0], list(argv), dict(os.environ if env is None else env), file_actions=actions, setsid=True, setsigmask=(), setsigdef=resets)
            os.close(null); opened.remove(null)
            os.close(wout); opened.remove(wout)
            os.close(werr); opened.remove(werr)
            self.stdout = os.fdopen(rout, "rb", buffering=0); opened.remove(rout)
            self.stderr = os.fdopen(rerr, "rb", buffering=0); opened.remove(rerr)
        except BaseException:
            for fd in opened:
                try: os.close(fd)
                except OSError: pass
            if hasattr(self, "pid"):
                try: os.killpg(self.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                try: os.waitpid(self.pid, 0)
                except ChildProcessError: pass
            raise

    def poll(self):
        with self._wait_lock:
            if self.returncode is None:
                pid, status = os.waitpid(self.pid, os.WNOHANG)
                if pid:
                    self.returncode = os.waitstatus_to_exitcode(status)
            return self.returncode

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic()+timeout
        while True:
            result = self.poll()
            if result is not None:
                return result
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.args, timeout)
            time.sleep(.01)

    def close_pipes(self):
        for pipe in (self.stdout, self.stderr):
            if pipe:
                pipe.close()


def capture_command(argv, *, timeout=2, cap=2_000_000):
    """Small trusted probes such as absolute /bin/ps, using no fork or shell."""
    process = SpawnProcess(argv)
    selector = selectors.DefaultSelector()
    data = {"stdout": bytearray(), "stderr": bytearray()}
    started = time.monotonic()
    try:
        for name in data:
            pipe = getattr(process, name); os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        while selector.get_map() or process.poll() is None:
            if time.monotonic()-started > timeout:
                raise subprocess.TimeoutExpired(argv, timeout)
            for key, _ in selector.select(.02):
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    selector.unregister(key.fileobj); continue
                if len(data[key.data])+len(chunk) > cap:
                    raise ValueError("Trusted probe exceeded its output limit")
                data[key.data].extend(chunk)
        result = SimpleNamespace(returncode=process.wait(timeout=.1), stdout=data["stdout"].decode(errors="replace"), stderr=data["stderr"].decode(errors="replace"))
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, argv, result.stdout, result.stderr)
        return result
    finally:
        if process.poll() is None:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait(timeout=2)
        selector.close(); process.close_pipes()


def _groups(pid):
    try:
        raw = capture_command(["/bin/ps", "-axo", "pid=,ppid=,pgid="]).stdout
        rows = [tuple(map(int, line.split())) for line in raw.splitlines() if len(line.split()) == 3]
        descendants = {pid}
        while True:
            children = {item for item, parent, _ in rows if parent in descendants}
            if children <= descendants: break
            descendants.update(children)
        return {group for item, _, group in rows if item in descendants and group != os.getpgrp()} | {pid}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {pid}


def _signal(groups, sig):
    for group in groups:
        if group == os.getpgrp(): continue
        try: os.killpg(group, sig)
        except (ProcessLookupError, PermissionError): pass


def _executing(groups):
    raw = capture_command(["/bin/ps", "-axo", "pid=,pgid=,stat="]).stdout
    return [{"pid": int(parts[0]), "process_group": int(parts[1]), "state": parts[2]} for line in raw.splitlines() if len(parts := line.split()) == 3 and int(parts[1]) in groups and not parts[2].startswith("Z")]


def stop_spawn_process(process):
    """Finish descendant termination even if its stage leader exits first."""
    groups = _groups(process.pid)
    _signal(groups, signal.SIGTERM)
    try: process.wait(timeout=1)
    except subprocess.TimeoutExpired: pass
    _signal(groups, signal.SIGKILL)
    try: process.wait(timeout=2)
    except subprocess.TimeoutExpired: pass
    deadline = time.monotonic()+2
    try:
        members = _executing(groups)
        while members and time.monotonic()<deadline:
            time.sleep(.03); members = _executing(groups)
        return {"spawn_method": "posix_spawn_setsid", "captured_groups": sorted(groups), "cleanup_confirmed": not members, "remaining_executing_members": members}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"spawn_method": "posix_spawn_setsid", "captured_groups": sorted(groups), "cleanup_confirmed": False, "verification_error": str(error)}


def spawn_run_command(argv, *, cwd, environment, stdout_path, stderr_path, timeout_seconds, log_cap=10_000_000):
    """Drop-in bounded stage runner for the NEW wrapper's execute_plan hook."""
    started = time.monotonic()
    env = dict(os.environ); env.update(environment); env["PYTHONUNBUFFERED"] = "1"
    process = None; selector = selectors.DefaultSelector(); files = {}
    sizes = {"stdout": 0, "stderr": 0}; status, reason = "completed", None
    cleanup = None
    try:
        for name, path in (("stdout", stdout_path), ("stderr", stderr_path)):
            files[name] = Path(path).open("xb")
        process = SpawnProcess(list(argv), cwd=cwd, env=env)
        for name in files:
            pipe = getattr(process, name); os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        while selector.get_map() or process.poll() is None:
            if time.monotonic()-started > timeout_seconds:
                status, reason = "timeout", "stage or remaining total deadline exceeded"
                cleanup = stop_spawn_process(process); break
            for key, _ in selector.select(.05):
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    selector.unregister(key.fileobj); continue
                name = key.data; available = max(0, log_cap-sizes[name])
                files[name].write(chunk[:available]); files[name].flush()
                sizes[name] += min(len(chunk), available)
                if len(chunk)>available:
                    status, reason = "log_limit", f"{name} exceeded {log_cap} bytes; retained bounded prefix"
                    cleanup = stop_spawn_process(process); break
            if status != "completed": break
        if process.poll() is None:
            cleanup = stop_spawn_process(process)
        if cleanup and not cleanup["cleanup_confirmed"]:
            status, reason = "cleanup_unconfirmed", "Stage descendant termination could not be confirmed"
        result = {"exit_code": process.returncode, "status": status, "failure_reason": reason, "elapsed_seconds": time.monotonic()-started, "stdout_bytes": sizes["stdout"], "stderr_bytes": sizes["stderr"], "spawn_method": "posix_spawn_setsid"}
        if cleanup: result["process_cleanup"] = cleanup
        return result
    finally:
        if process and process.poll() is None:
            stop_spawn_process(process)
        selector.close()
        for stream in files.values(): stream.close()
        if process: process.close_pipes()
