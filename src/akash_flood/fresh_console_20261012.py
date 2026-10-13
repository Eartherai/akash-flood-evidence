"""Loopback execution console, additive to the immutable fresh-case runner.

Only validated area/date/budget fields cross the HTTP boundary. Offline plans
freeze the actual runner inputs and stages; a separate explicit action launches
that exact plan. No cached result is substituted for a run.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import secrets
import selectors
import signal
import subprocess
import sys
import threading
import time

from akash_flood.fresh_spawn_20261012 import SpawnProcess, capture_command, spawn_run_command
from akash_flood.console_activation_20261012 import resolve_startup

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/fresh-console-2026-10-12"
BASE_RUNNER = ROOT / "scripts/run_fresh_case.py"
COMBINED_RUNNER = ROOT / "scripts/run_multimodal_fresh_20261012.py"
WORKER = ROOT / "scripts/serve_fresh_console_20261012.py"
JOB_ID = re.compile(r"case-[0-9a-f]{24}\Z")
LOG_CAP = 10_000_000
JSON_CAP = 16_000_000
DEFAULTS = {"bbox": [85.32, 27.70, 85.38, 27.76], "event_date": "2026-08-26", "osm_snapshot": "2026-07-27", "workflow": "water", "max_days": 24, "native_budget_mb": 3100, "dem_budget_mb": 200, "optical_budget_mb": 60, "stage_timeout_seconds": 1200, "max_total_seconds": 3600, "optical_timeout_seconds": 300}


def now():
    return datetime.now(timezone.utc).isoformat()


def hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hash_file(path)}


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    if Path(path).stat().st_size > JSON_CAP:
        raise ValueError("Receipt exceeds the console JSON limit")
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def write_json(path, value):
    # Atomic replacement keeps live polling from interpreting partial writes.
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def checked_payload(value):
    if not isinstance(value, dict) or set(value) != set(DEFAULTS):
        raise ValueError("Supply only the area, dates, workflow and displayed budget fields")
    result = dict(value)
    bbox = value["bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in bbox):
        raise ValueError("Area needs four finite numeric coordinates: west, south, east, north")
    w, s, e, n = bbox
    if not (-180 < w < e < 180 and -60 <= s < n <= 80):
        raise ValueError("Area must be nonwrapping and within longitude ±180°, latitude −60° to 80°")
    # The console has a deliberately smaller envelope than the CLI runner.
    context = [w-.03, s-.03, e+.03, n+.03]
    if (context[2]-context[0])*(context[3]-context[1]) > .20:
        raise ValueError("Area plus 0.03° context exceeds this console's 0.20-square-degree limit")
    if context[0] <= -180 or context[2] >= 180 or context[1] < -60 or context[3] > 80:
        raise ValueError("The 0.03° processing context exceeds the supported coordinate range")
    zone = lambda longitude: min(60, int((longitude+180)//6)+1)
    if zone(context[0]) != zone(context[2]) or context[1] <= 0 <= context[3]:
        raise ValueError("This local processor needs one UTM zone and one hemisphere, including context")
    for key in ("event_date", "osm_snapshot"):
        if not isinstance(value[key], str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value[key]):
            raise ValueError(f"{key} needs an ISO date, YYYY-MM-DD")
        try:
            date.fromisoformat(value[key])
        except ValueError as error:
            raise ValueError(f"{key} is not a calendar date") from error
    if date.fromisoformat(value["osm_snapshot"]) >= date.fromisoformat(value["event_date"]):
        raise ValueError("Historical OSM cutoff must precede the event day (UTC)")
    if value["workflow"] not in ("water", "multimodal"):
        raise ValueError("Choose water or multimodal workflow")
    bounds = {"max_days": (1, 90), "native_budget_mb": (1, 4100), "dem_budget_mb": (1, 500), "optical_budget_mb": (1, 100), "stage_timeout_seconds": (1, 3600), "max_total_seconds": (1, 7200), "optical_timeout_seconds": (1, 600)}
    for key, (low, high) in bounds.items():
        if isinstance(value[key], bool) or not isinstance(value[key], int) or not low <= value[key] <= high:
            raise ValueError(f"{key} must be a whole number from {low} to {high}")
    result["bbox"] = [float(x) for x in bbox]
    return result


def load_runner(path):
    path = Path(path).resolve(strict=True)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("akash_console_runner_" + hashlib.sha256(str(path).encode()).hexdigest()[:12], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan_argv(value, out):
    args = ["--bbox", *map(str, value["bbox"]), "--event-date", value["event_date"], "--osm-snapshot", value["osm_snapshot"], "--context-buffer-deg", ".03", "--max-days", str(value["max_days"]), "--native-budget-bytes", str(value["native_budget_mb"]*1_000_000), "--dem-budget-bytes", str(value["dem_budget_mb"]*1_000_000), "--stage-timeout-seconds", str(value["stage_timeout_seconds"]), "--max-total-seconds", str(value["max_total_seconds"]), "--with-report", "--out", str(out)]
    if value["workflow"] == "multimodal":
        args += ["--with-optical", "--with-observation-access", "--optical-budget-bytes", str(value["optical_budget_mb"]*1_000_000), "--optical-timeout-seconds", str(value["optical_timeout_seconds"])]
    return args


def bound_path(path, parent):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Console outputs cannot be symlinks")
    absolute = path.resolve()
    absolute.relative_to(Path(parent).resolve())
    # Refuse symlinks at every intermediate component, even those pointing in.
    cursor = path.absolute()
    while cursor != Path(parent).absolute():
        if cursor.is_symlink():
            raise ValueError("Console output ancestors cannot be symlinks")
        if cursor == cursor.parent:
            raise ValueError("Console output is outside its dedicated root")
        cursor = cursor.parent
    return absolute


def walk_records(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            yield value
        for child in value.values():
            yield from walk_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_records(child)


def verify_plan(plan, plan_path, expected):
    if record(plan_path) != expected:
        raise ValueError("Offline execution plan changed; make a new plan")
    records = [plan["source_runner"]]
    records += list(walk_records(plan.get("external_inputs", {})))
    records += [item for stage in plan["stages"] for item in stage.get("source_code", [])]
    for item in records:
        if item.get("missing") or record(item["path"]) != {key: item[key] for key in ("path", "bytes", "sha256")}:
            raise ValueError("A planned source or prepared input changed; make a new plan")
    if plan.get("blockers"):
        raise ValueError("Execution remains blocked: " + "; ".join(plan["blockers"]))


def descendant_groups(pid):
    """Record only groups belonging to this launched process tree, never names."""
    try:
        raw = capture_command(["/bin/ps", "-axo", "pid=,ppid=,pgid="], timeout=2).stdout
        rows = [tuple(map(int, line.split())) for line in raw.splitlines() if len(line.split()) == 3]
        descendants = {pid}
        while True:
            children = {item for item, parent, _ in rows if parent in descendants}
            if children <= descendants:
                break
            descendants.update(children)
        return {group for item, _, group in rows if item in descendants and group != os.getpgrp()}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {pid}


def signal_groups(groups, sig):
    for group in groups:
        if group == os.getpgrp():
            continue
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            # On macOS a reparented, already-terminal group can reject another
            # signal. Terminal cleanup below checks actual executing members;
            # it never treats a rejected signal as proof of termination.
            pass


def executing_group_members(groups):
    raw = capture_command(["/bin/ps", "-axo", "pid=,pgid=,stat="], timeout=2).stdout
    result = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) == 3 and int(parts[1]) in groups and not parts[2].startswith("Z"):
            result.append({"pid": int(parts[0]), "process_group": int(parts[1]), "state": parts[2]})
    return result


def finish_group_cleanup(groups):
    signal_groups(groups, signal.SIGKILL)
    deadline = time.monotonic()+2
    try:
        members = executing_group_members(groups)
        while members and time.monotonic() < deadline:
            time.sleep(.03)
            members = executing_group_members(groups)
        return {"captured_groups": sorted(groups), "cleanup_confirmed": not members,
                "remaining_executing_members": members}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"captured_groups": sorted(groups), "cleanup_confirmed": False,
                "verification_error": str(error)}


class ConsoleService:
    def __init__(self, *, output=OUTPUT, runners=None, worker=WORKER, python=sys.executable, trusted_inputs=None,
                 activation=None, requirements=None, source_root=ROOT):
        self.output = Path(output).absolute()
        if self.output.is_symlink():
            raise ValueError("Dedicated output root cannot be a symlink")
        self.output.mkdir(parents=True, exist_ok=True)
        self.output = self.output.resolve()
        self.runners = runners or {"water": BASE_RUNNER, "multimodal": COMBINED_RUNNER}
        self.trusted_inputs = trusted_inputs or {}
        if set(self.trusted_inputs) - {"source-cache", "reuse-dem-tile", "osm", "osm-provenance"}:
            raise ValueError("Unsupported trusted startup input")
        if bool(self.trusted_inputs.get('osm'))!=bool(self.trusted_inputs.get('osm-provenance')):
            raise ValueError('Trusted historical extract requires both data and provenance')
        self.runtime_inputs, self.runtime_receipt = resolve_startup(source_root,
            activation=activation, requirements=requirements)
        self.worker = Path(worker).resolve()
        self.python = str(Path(python).absolute())
        self.lock = threading.RLock()
        self.jobs = {}
        self.active = None
        self.token = secrets.token_urlsafe(32)
        self._cancel = threading.Event()
        self._process = None
        for directory in sorted(self.output.glob("case-*"))[-100:]:
            if not JOB_ID.fullmatch(directory.name) or directory.is_symlink():
                continue
            try:
                value = read_json(bound_path(directory / "console-status.json", self.output))
                if value["id"] != directory.name:
                    continue
                if value["status"] == "running":
                    value.update(status="interrupted", failure_reason="Console restarted before this run had a terminal console receipt. No automatic resume or retry.")
                    write_json(directory / "console-status.json", value)
                self.jobs[value["id"]] = value
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def capabilities(self):
        return {"defaults": DEFAULTS, "workflows": {key: {"available": Path(path).is_file(), "entry_point": Path(path).name} for key, path in self.runners.items()}, "output_scope": "outputs/fresh-console-2026-10-12", "area_limit_square_degrees": .20, "context_buffer_degrees": .03, "full_track_b_requirements_satisfied": False, "fresh_case_unseen_validation": False, "explicit_source_reuse": {key: str(value) for key, value in self.trusted_inputs.items()}, "native_startup": {"activation_supplied": self.runtime_receipt is not None and "activation" in self.runtime_receipt, "runner_inputs": {key: str(value) for key, value in self.runtime_inputs.items()}, "raw_satellite_replayed": False}, "csrf_token": self.token}

    def _save(self, job):
        write_json(self.output / job["id"] / "console-status.json", job)

    def make_plan(self, payload):
        value = checked_payload(payload)
        with self.lock:
            if self.active:
                raise BlockingIOError("One case is already running; cancel or wait before planning another")
            if len(self.jobs) >= 100:
                raise ValueError("This local session has reached its 100-plan retention limit")
            path = Path(self.runners[value["workflow"]])
            if not path.is_file():
                raise ValueError("Selected workflow source is not installed; select the available workflow")
            module = load_runner(path)
            job_id = "case-" + secrets.token_hex(12)
            directory = bound_path(self.output / job_id, self.output)
            directory.mkdir(exist_ok=False)
            out = directory / "execution"
            try:
                argv = plan_argv(value, out)
                for key, input_path in self.trusted_inputs.items():
                    argv += ["--" + key, str(input_path)]
                for key, input_path in self.runtime_inputs.items():
                    argv += ["--" + key, str(input_path)]
                args = module.parser().parse_args(argv)
                plan = module.build_plan(args)
                plan["external_inputs"]["fork_free_spawn_source"] = record(ROOT / "src/akash_flood/fresh_spawn_20261012.py")
                if self.runtime_receipt:
                    plan["external_inputs"]["console_native_startup"] = self.runtime_receipt
                try:
                    version = importlib.metadata.version("scikit-learn")
                except importlib.metadata.PackageNotFoundError:
                    version = None
                plan["console_environment_check"] = {"scikit_learn": version, "required_scikit_learn": "1.7.2", "python_executable": self.python}
                if version != "1.7.2":
                    plan["blockers"].append("Start this console in the pinned scikit-learn==1.7.2 environment; current version: " + str(version))
                if Path(plan["out"]).resolve() != out.resolve():
                    raise ValueError("Selected runner planned outside the server-generated case output")
                plan_path = directory / "console-plan.json"
                write_json(plan_path, plan)
                job = {"schema": "akash-fresh-console-job-v1", "id": job_id, "created_at": now(), "status": "planned", "parameters": value, "runner": record(path), "console_plan": record(plan_path), "output_path": str(out), "run_attempts": 0, "plan_subprocesses_run": 0, "plan_network_requests": 0, "trusted_startup_reuse": {key: str(value) for key, value in self.trusted_inputs.items()}, "console_sources": [record(Path(__file__)), record(self.worker), record(ROOT / "src/akash_flood/fresh_spawn_20261012.py"), record(ROOT / "src/akash_flood/console_activation_20261012.py")], "full_track_b_requirements_satisfied": False}
            except Exception as error:
                job = {"schema": "akash-fresh-console-job-v1", "id": job_id, "created_at": now(), "status": "plan_failed", "parameters": value, "failure_type": type(error).__name__, "failure_reason": str(error), "run_attempts": 0, "full_track_b_requirements_satisfied": False}
                self.jobs[job_id] = job
                self._save(job)
                raise
            self.jobs[job_id] = job
            self._save(job)
            return self.snapshot(job_id)

    def start(self, job_id):
        with self.lock:
            job = self._job(job_id)
            if self.active:
                raise BlockingIOError("Only one local case can run at a time")
            if job["status"] != "planned" or job["run_attempts"] != 0:
                raise ValueError("Plans run at most once; failed and completed cases are preserved")
            plan_path = bound_path(self.output / job_id / "console-plan.json", self.output)
            plan = read_json(plan_path)
            verify_plan(plan, plan_path, job["console_plan"])
            if any(record(item["path"]) != item for item in job.get("console_sources", [])):
                raise ValueError("Console code changed after planning; inspect a new plan")
            if record(self.runners[job["parameters"]["workflow"]]) != job["runner"]:
                raise ValueError("Selected runner changed after planning")
            out = bound_path(job["output_path"], self.output)
            if out.exists():
                raise ValueError("Fresh execution output already exists")
            job.update(status="running", run_attempts=1, started_at=now(), worker_argv=[self.python, str(self.worker), "--worker-plan", str(plan_path)], worker_source=record(self.worker), spawn_method="posix_spawn_setsid")
            try:
                self._save(job)
            except OSError as error:
                job.update(status="failed", failure_type=type(error).__name__, failure_reason="Could not persist the explicit run action; no worker was launched", persistence_error=str(error))
                raise
            self.active = job_id
            self._cancel.clear()
            thread = threading.Thread(target=self._watch, args=(job_id,), daemon=True)
            thread.start()
            return self.snapshot(job_id)

    def _job(self, job_id):
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id) or job_id not in self.jobs:
            raise KeyError("Unknown case ID")
        return self.jobs[job_id]

    def cancel(self, job_id):
        with self.lock:
            job = self._job(job_id)
            if self.active != job_id or job["status"] != "running":
                raise ValueError("Only the active real process can be cancelled")
            self._cancel.set()
            job["cancel_requested_at"] = now()
            self._save(job)
            return {"id": job_id, "status": "cancellation_requested"}

    def _watch(self, job_id):
        job = self.jobs[job_id]
        directory = self.output / job_id
        streams = {}
        sizes = {"stdout": 0, "stderr": 0}
        selector = selectors.DefaultSelector()
        started = time.monotonic()
        stopped = None
        groups = set()
        failure = None
        process = None
        try:
            for name in ("stdout", "stderr"):
                streams[name] = (directory / f"console.{name}.log").open("wb")
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
            process = SpawnProcess(job["worker_argv"], cwd=ROOT, env=env)
            with self.lock:
                self._process = process
                job["worker_pid"] = process.pid
                self._save(job)
            for name in streams:
                pipe = getattr(process, name)
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, name)
            deadline = job["parameters"]["max_total_seconds"] + 15
            while selector.get_map() or process.poll() is None:
                if stopped is None and (self._cancel.is_set() or time.monotonic()-started > deadline or failure):
                    groups = descendant_groups(process.pid)
                    # Interrupt the Python worker, letting the runner's finally
                    # cleanup stop its independently grouped stage processes.
                    signal_groups(groups - {process.pid}, signal.SIGTERM)
                    signal_groups({process.pid}, signal.SIGINT)
                    stopped = time.monotonic()
                    if time.monotonic()-started > deadline:
                        failure = "Console total execution deadline exceeded"
                if stopped is not None and time.monotonic()-stopped > 8:
                    signal_groups(groups, signal.SIGKILL)
                if stopped is not None and time.monotonic()-stopped > 12:
                    failure = failure or "Cancellation did not terminate the worker within its bounded cleanup window"
                    break
                for key, _ in selector.select(.05):
                    data = os.read(key.fileobj.fileno(), 8192)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    name = key.data
                    remaining = LOG_CAP-sizes[name]
                    streams[name].write(data[:max(0, remaining)])
                    streams[name].flush()
                    sizes[name] += min(len(data), max(0, remaining))
                    if len(data) > remaining:
                        failure = f"Console {name} exceeded its {LOG_CAP}-byte cap; saved log was truncated"
            exit_code = process.wait(timeout=2)
            # Stage leaders may exit on TERM while a Java/grandchild ignores
            # it with redirected logs. Clean every captured descendant group
            # even when the worker/pipes have already reached EOF.
            if stopped is not None:
                job["process_cleanup"] = finish_group_cleanup(groups)
                if not job["process_cleanup"]["cleanup_confirmed"]:
                    failure = "Process termination could not be confirmed; inspect console cleanup receipt"
            if self._cancel.is_set() and not failure:
                status = "cancelled"
            elif failure or exit_code != 0:
                status = "failed"
            else:
                receipt_path = directory / "execution/execution-receipt.json"
                receipt = read_json(receipt_path)
                status = "completed" if receipt.get("status") == "completed" else "failed"
                if status == "failed":
                    failure = "Runner exited successfully without a completed execution receipt"
            with self.lock:
                job.update(status=status, completed_at=now(), elapsed_seconds=round(time.monotonic()-started, 3), exit_code=exit_code)
                if failure:
                    job["failure_reason"] = failure
                elif status == "failed":
                    job["failure_reason"] = f"Worker exited with {exit_code}; inspect actual stage receipt and error log"
        except Exception as error:
            with self.lock:
                job.update(status="failed", completed_at=now(), failure_type=type(error).__name__, failure_reason=str(error))
        finally:
            # Disk-full, log-open/hash and persistence failures must not strand
            # self.active or skip real-process cleanup.
            try:
                if process and process.poll() is None:
                    groups |= descendant_groups(process.pid)
                if groups:
                    job["process_cleanup"] = finish_group_cleanup(groups)
                    if not job["process_cleanup"]["cleanup_confirmed"]:
                        job.update(status="failed", failure_reason="Executing descendant termination could not be confirmed")
                if process and process.poll() is None:
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        job.update(status="failed", failure_reason="Worker remains executing after bounded cancellation")
            except Exception as error:
                job.update(status="failed", cleanup_error=str(error))
            finally:
                try:
                    selector.close()
                except OSError as error:
                    job["selector_close_error"] = str(error)
                for stream in streams.values():
                    try:
                        stream.close()
                    except OSError as error:
                        job["log_close_error"] = str(error)
                if process:
                    for pipe in (process.stdout, process.stderr):
                        if pipe:
                            try:
                                pipe.close()
                            except OSError as error:
                                job["pipe_close_error"] = str(error)
                with self.lock:
                    try:
                        job["console_logs"] = {name: record(directory / f"console.{name}.log") for name in streams}
                        self._save(job)
                    except Exception as error:
                        job["persistence_error"] = str(error)
                        job["status"] = "failed"
                    finally:
                        self.active = None
                        self._process = None

    def snapshot(self, job_id):
        with self.lock:
            job = dict(self._job(job_id))
            directory = bound_path(self.output / job_id, self.output)
            plan_path = directory / "console-plan.json"
            if plan_path.is_file():
                job["plan"] = read_json(plan_path)
            receipt = directory / "execution/execution-receipt.json"
            if receipt.is_file():
                try:
                    job["receipt"] = read_json(bound_path(receipt, directory))
                except (ValueError, json.JSONDecodeError):
                    job["receipt_read_pending"] = True
            job["logs"] = {}
            for name in ("stdout", "stderr"):
                path = directory / f"console.{name}.log"
                if path.is_file():
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size-12_000))
                        job["logs"][name] = stream.read(12_000).decode("utf-8", errors="replace")
            actual = job.get("receipt", {}).get("stages", [])
            if job["status"] == "running":
                finished = {stage["name"] for stage in actual if stage.get("status") == "completed"}
                for stage in job.get("plan", {}).get("stages", []):
                    name = stage["name"]
                    if name not in finished and re.fullmatch(r"[A-Za-z0-9_-]+", name) and (directory / "execution/logs" / (name + ".stdout.log")).is_file():
                        job["active_stage_observation"] = {"name": name, "evidence": "actual stage log files exist; completion remains governed by the runner receipt"}
                        actual = actual + [{"name": name}]
                        break
            # Receipt stage logs belong to this execution only, never old cases.
            for stage in reversed(actual):
                for name in ("stdout", "stderr"):
                    relative = f"logs/{stage.get('name','')}.{name}.log"
                    path = directory / "execution" / relative
                    if re.fullmatch(r"[A-Za-z0-9_-]+", stage.get("name", "")) and path.is_file():
                        bound_path(path, directory)
                        with path.open("rb") as stream:
                            stream.seek(max(0, path.stat().st_size-12_000))
                            job["logs"]["stage_"+name] = stream.read(12_000).decode("utf-8", errors="replace")
                break
            report_files = self.report_files(job_id)
            if "index.html" in report_files:
                job["combined_report_url"] = f"/reports/{job_id}/index.html"
            return job

    def report_files(self, job_id):
        job = self._job(job_id)
        directory = self.output / job_id / "execution"
        receipt_path = directory / "execution-receipt.json"
        if not receipt_path.is_file():
            return {}
        receipt = read_json(bound_path(receipt_path, self.output))
        stage = next((item for item in receipt.get("stages", []) if item.get("name") == "build_combined_case_report" and item.get("status") == "completed"), None)
        if not stage:
            return {}
        artifact = next((item for item in stage.get("artifacts", []) if item.get("kind") == "fresh_combined_report"), None)
        if not artifact:
            return {}
        manifest_path = bound_path(artifact["path"], directory / "combined-report")
        if record(manifest_path) != {key: artifact[key] for key in ("path", "bytes", "sha256")}:
            raise ValueError("Completed report manifest no longer matches its stage receipt")
        manifest = read_json(manifest_path)
        if manifest.get("schema") != "akash-fresh-combined-report-v1" or manifest.get("status") != "completed":
            raise ValueError("Expected the actual completed combined report")
        result = {"manifest.json": manifest_path}
        for name, item in manifest.get("outputs", {}).items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
                raise ValueError("Invalid generated report asset name")
            path = bound_path(item["path"], directory / "combined-report")
            if path.relative_to((directory / "combined-report").resolve()).as_posix() != name:
                raise ValueError("Report asset name differs from its output path")
            if record(path) != {key: item[key] for key in ("path", "bytes", "sha256")}:
                raise ValueError("Generated report output changed after its completed receipt")
            # Source-bound route facts may exceed20MB; keep a64MB per-output
            # hard cap while verifying the exact completed manifest identity.
            if path.suffix.lower() in {".html", ".json", ".md", ".png", ".jpg", ".jpeg", ".svg"} and path.stat().st_size <= 64_000_000:
                result[name] = path
        return result

    def report_asset(self, job_id, name):
        with self.lock:
            files = self.report_files(job_id)
            if name not in files:
                raise KeyError("Generated report asset is unavailable")
            return files[name].read_bytes()

    def listing(self):
        with self.lock:
            return {"active": self.active, "jobs": [{key: job.get(key) for key in ("id", "created_at", "status", "parameters", "failure_reason")} for job in reversed(list(self.jobs.values()))]}

    def close(self):
        if self.active:
            with self.lock:
                job = self.jobs[self.active]
                job["shutdown_cancel_requested_at"] = now()
                try:
                    self._save(job)
                except OSError as error:
                    job["persistence_error"] = str(error)
            self._cancel.set()
            deadline = time.monotonic()+20
            while self.active and time.monotonic() < deadline:
                time.sleep(.05)


def execute_worker(plan_path):
    plan_path = Path(plan_path).absolute()
    bound_path(plan_path, OUTPUT)
    if plan_path.name != "console-plan.json" or not JOB_ID.fullmatch(plan_path.parent.name):
        raise ValueError("Worker accepts only a server-generated console plan")
    status = read_json(plan_path.parent / "console-status.json")
    if status.get("status") != "running" or status.get("run_attempts") != 1:
        raise ValueError("Worker needs an explicit active console run")
    if any(record(item["path"]) != item for item in status.get("console_sources", [])):
        raise ValueError("Console code changed between planning and execution")
    selected = BASE_RUNNER if status["parameters"]["workflow"] == "water" else COMBINED_RUNNER
    if record(selected) != status["runner"]:
        raise ValueError("Runner changed between planning and execution")
    plan = read_json(plan_path)
    verify_plan(plan, plan_path, status["console_plan"])
    bound_path(plan["out"], plan_path.parent)
    if Path(plan["out"]).name != "execution":
        raise ValueError("Worker output must be its dedicated execution directory")
    module = load_runner(selected)
    result = module.execute_plan(plan, command_runner=spawn_run_command)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return result
