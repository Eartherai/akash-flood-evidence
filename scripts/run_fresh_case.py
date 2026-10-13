"""Bounded local raw-to-water stage launcher; offline execution plan by default.

No subprocess or network runs without --execute. Checkpoints are never trained
or retuned here. Existing case directories are refused, including failed runs.
Optional exposure/report stages retain provisional water-state limitations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from akash_flood.catalog import validate_bbox
from akash_flood.fresh_case import (
    event_boundary, osm_plan, parse_discovery, reject_source_path,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "akash-fresh-case-runner-v1"
OSM_ENDPOINT = "https://overpass-api.de/api/interpreter"
OSM_BYTES = 30_000_000
OSM_DEADLINE_SECONDS = 120
LOG_BYTES = 10_000_000


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Pipeline JSON artifact must be an object")
    return value


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def verify_record(record):
    path = Path(record["path"])
    reject_source_path(path)
    if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"Source bytes changed from recorded identity: {path}")
    return path


def finite_bbox(values, label):
    bbox = validate_bbox(values)
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError(f"{label} contains nonfinite coordinates")
    if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > 1:
        raise ValueError(f"{label} exceeds the local one-square-degree launcher bound")
    return bbox


def context_for(args):
    bbox = finite_bbox(args.bbox, "AOI")
    if not math.isfinite(args.context_buffer_deg) or not 0 <= args.context_buffer_deg <= .25:
        raise ValueError("Context buffer must be finite, 0 to 0.25 degrees")
    context = finite_bbox(args.context_bbox or [bbox[0]-args.context_buffer_deg, bbox[1]-args.context_buffer_deg, bbox[2]+args.context_buffer_deg, bbox[3]+args.context_buffer_deg], "context")
    if not (context[0] <= bbox[0] < bbox[2] <= context[2] and context[1] <= bbox[1] < bbox[3] <= context[3]):
        raise ValueError("Context bbox must completely cover the AOI")
    return bbox, context


def bounded_int(value, low, high, label):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{label} must be {low} to {high}")
    return value


def external_inputs(args, root, bbox, context):
    """Read-only checksum/source checks; no subprocess, API or checkpoint load."""
    event_start, _, snapshot = event_boundary(args.event_date, args.osm_snapshot)
    result = {}
    if bool(args.osm) != bool(args.osm_provenance):
        raise ValueError("Cached OSM requires both --osm and --osm-provenance")
    if args.omit_osm and (args.osm or args.osm_provenance):
        raise ValueError("--omit-osm conflicts with cached OSM inputs")
    if args.catalog:
        reject_source_path(args.catalog)
        catalog = read_json(args.catalog)
        groups, incomplete = parse_discovery(catalog)
        result["catalog"] = file_record(args.catalog) | {
            "source_scope": "saved public catalogue metadata; no new discovery or imagery acquisition claim",
            "source_family_counts": {key: len(value) for key, value in groups.items()},
            "catalogue_incomplete": incomplete,
        }
    if args.osm:
        # Validate complete historical-query identity rather than only hashing it.
        checked = osm_plan(snapshot, context, osm_path=args.osm, provenance_path=args.osm_provenance)
        result["osm"] = file_record(args.osm)
        result["osm_provenance"] = file_record(args.osm_provenance)
        result["historic_osm_check"] = {"status": checked["status"], "snapshot": checked["snapshot"], "bbox": checked["bbox"], "scope": "saved historic extract checked; no new OSM retrieval"}
    elif not args.omit_osm and snapshot.time() != datetime.min.time():
        raise ValueError("Bounded OSM retrieval currently supports date or UTC-midnight snapshots; supply checked cached OSM for an exact nonmidnight cutoff")
    if args.source_cache:
        cache = args.source_cache.resolve(strict=True)
        result["native_source_cache_index"] = file_record(cache / "derived-wrapper-index.json")
        result["native_source_cache"] = {"path": str(cache), "scope": "explicit prior-source import; acquire stage must independently verify all selected-item/source hashes; no new remote download claim"}
    runtime = args.snap_runtime.resolve()
    supported = (root / "data/snap-runtime/portable").resolve()
    if runtime != supported:
        raise ValueError("Current SNAP processor supports only its explicit prepared portable runtime; alternate --snap-runtime path cannot be silently ignored")
    result["snap_runtime"] = {"path": str(runtime), "override_supported": False, "scope": "existing local runtime; launcher does not install/download SNAP"}
    runtime_manifest = runtime.parent / "runtime-download-manifest.json"
    if runtime_manifest.is_file():
        result["snap_runtime_manifest"] = file_record(runtime_manifest)
    for name, path in (("dem_reader_patch", args.dem_reader_patch), ("dem_reader_proof", args.dem_reader_proof), ("model_freeze", args.model_freeze), ("geoid", args.geoid), ("requirements", args.requirements)):
        reject_source_path(path)
        if path.is_file():
            result[name] = file_record(path)
        else:
            result[name] = {"path": str(path.resolve()), "missing": True}
    if args.reuse_dem_tile:
        # prepare_fresh_dem currently has one hash-bound implicit source cache.
        # Do not pretend an arbitrary supplied tile can redirect that interface.
        manifest_path = root / "data/copdem-ellipsoid/manifest.json"
        manifest = read_json(manifest_path)
        prior = manifest.get("downloads", {}).get("dem_orthometric", {})
        candidate = args.reuse_dem_tile.resolve(strict=True)
        if (Path(prior.get("path", "")).resolve() != candidate or not candidate.is_file()
                or candidate.stat().st_size != prior.get("bytes") or sha256(candidate) != prior.get("sha256")):
            raise ValueError("--reuse-dem-tile must match the current preparer's exact checksum-bound default source tile; alternate reuse interface is not supported")
        result["reuse_dem_tile"] = file_record(candidate) | {"provider_url": prior["url"], "source_manifest": file_record(manifest_path), "scope": "explicit binding to preparer's existing verified source-cache behavior; reuse occurs only if required provider URL matches"}
    return result


def build_plan(args, *, root=ROOT):
    root = Path(root).resolve()
    output = args.out.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new case root; completed and failed runs are never automatically repeated")
    bbox, context = context_for(args)
    bounded_int(args.max_days, 1, 90, "date window")
    bounded_int(args.native_budget_bytes, 1, 4_100_000_000, "native network budget")
    bounded_int(args.dem_budget_bytes, 1, 500_000_000, "DEM network budget")
    bounded_int(args.stage_timeout_seconds, 1, 3600, "stage timeout")
    bounded_int(args.max_total_seconds, 1, 7200, "total timeout")
    inputs = external_inputs(args, root, bbox, context)
    scripts = root / "scripts"
    python = sys.executable
    script = lambda name: str(scripts / name)
    stages = []
    def add(name, argv, artifacts, modules=(), network="none", required=True):
        code = [Path(argv[1]), *[root / item for item in modules]]
        stages.append({"name": name, "argv": [str(value) for value in argv], "cwd": str(root), "network_scope": network, "required": required, "source_code": [file_record(path) if path.is_file() else {"path": str(path), "missing": True} for path in code], "artifacts": [{"path": str(path), "kind": kind} for path, kind in artifacts], "timeout_seconds": args.stage_timeout_seconds})
    plan_path = output / "execution-plan.json"
    add("validate_sources", [python, script("run_fresh_case.py"), "--_validate-inputs", "--input-spec", plan_path, "--out", output / "source-validation"], [(output / "source-validation/receipt.json", "source_validation")])
    if args.catalog:
        catalog_path = args.catalog.resolve()
    else:
        catalog_path = output / "discovery/catalog.json"
        add("discover", [python, script("discover_fresh_case.py"), "--bbox", *context, "--context-buffer-deg", 0, "--event-date", args.event_date, "--max-days", args.max_days, "--out", output / "discovery"], [(catalog_path, "catalog")], ("src/akash_flood/public_discovery.py",), network="new public EarthSearch queries; no image pixels; bounded existing discovery client")
    if args.osm:
        osm, osm_provenance = args.osm.resolve(), args.osm_provenance.resolve()
    elif args.omit_osm:
        osm = osm_provenance = None
    else:
        osm, osm_provenance = output / "historical-osm/osm.json", output / "historical-osm/provenance.json"
        add("retrieve_historic_osm", [python, script("run_fresh_case.py"), "--_fetch-historical-osm", "--bbox", *context, "--event-date", args.event_date, "--osm-snapshot", args.osm_snapshot, "--out", output / "historical-osm"], [(osm, "osm"), (osm_provenance, "osm_provenance"), (output / "historical-osm/network-receipt.json", "network_receipt")], ("src/akash_flood/osm.py",), network="new historical Overpass retrieval; one HTTPS request, 30 MB response-body cap, 120-second deadline; no retry/current-data fallback")
    case_plan = output / "planning/plan.json"
    argv = [python, script("plan_fresh_case.py"), "--bbox", *bbox, "--context-bbox", *context, "--event-date", args.event_date, "--osm-snapshot", args.osm_snapshot, "--catalog", catalog_path, "--max-days", args.max_days, "--requirements", args.requirements, "--out", output / "planning"]
    if osm:
        argv += ["--osm", osm, "--osm-provenance", osm_provenance]
    add("plan_case", argv, [(case_plan, "case_plan")], ("src/akash_flood/fresh_case.py",))
    native_index = output / "native/derived-wrapper-index.json"
    argv = [python, script("acquire_fresh_native.py"), "--plan", case_plan, "--execute", "--output-dir", output / "native", "--cache-dir", output / "native-cache", "--network-budget-bytes", args.native_budget_bytes]
    if args.source_cache:
        argv += ["--source-cache", args.source_cache.resolve()]
    add("acquire_native", argv, [(native_index, "native_index")], ("src/akash_flood/fresh_native.py",), network="explicit verified local import when possible; otherwise bounded anonymous public Sentinel-1 HTTPS acquisition")
    dem_manifest = output / "dem/manifest.json"
    add("prepare_dem", [python, script("prepare_fresh_dem.py"), "--bbox", *context, "--aoi-bbox", *bbox, "--budget-bytes", args.dem_budget_bytes, "--geoid", args.geoid, "--out", output / "dem"], [(dem_manifest, "dem_manifest")], ("src/akash_flood/fresh_dem.py", "scripts/prepare_copdem_ellipsoid.py"), network="verified cached eligible tile/geoid reuse where exact source matches; otherwise bounded Copernicus DEM/NGA geoid downloads")
    processing = output / "processing/run-metadata.json"
    add("process_s1", [python, script("process_fresh_s1.py"), "--plan", case_plan, "--source-index", native_index, "--dem-manifest", dem_manifest, "--dem-reader-patch", args.dem_reader_patch, "--dem-reader-proof", args.dem_reader_proof, "--execute", "--out", output / "processing"], [(processing, "processing_record")], ("src/akash_flood/snap_products.py", "scripts/process_s1_copdem.py", "scripts/snap_gpt.py"))
    water = output / "water/manifest.json"
    add("infer_water", [python, script("infer_fresh_water.py"), "--processing-record", processing, "--model-freeze", args.model_freeze, "--out", output / "water"], [(water, "water_manifest")], ("src/akash_flood/flood_model.py", "src/akash_flood/water_inference.py", "scripts/infer_trained_water.py"))
    with_exposure = args.with_exposure or args.with_report
    omissions = [{"step": "training", "reason": "not included; unchanged frozen checkpoint only"}, {"step": "debris_detection", "reason": "not implemented by the water-state pipeline"}, {"step": "local/unseen_accuracy_validation", "reason": "not established by launch or successful stage execution"}, {"step": "optical_analysis", "reason": "catalogued discovery only; no S2 reflectance or usable-AOI claims"}, {"step": "connectivity_and_observed_road_closures", "reason": "not inferred from candidate pixels"}]
    blockers = []
    if args.omit_osm:
        omissions.append({"step": "historic_osm", "reason": "explicit --omit-osm; exposure/report cannot run"})
    if with_exposure and not osm:
        blockers.append("Exposure/report was requested without historic OSM; supply cached inputs or enable retrieval")
    elif with_exposure:
        exposure = output / "exposure/source-and-output-manifest.json"
        add("map_exposure", [python, script("map_fresh_exposure.py"), "--water-manifest", water, "--plan", case_plan, "--out", output / "exposure"], [(output / "exposure/summary.json", "exposure_summary"), (exposure, "exposure_manifest")])
        if args.with_report:
            add("build_report", [python, script("build_fresh_report.py"), "--water-manifest", water, "--exposure-manifest", exposure, "--out", output / "report"], [(output / "report/report.json", "report")])
    else:
        omissions.append({"step": "exposure", "reason": "optional stage not requested"})
    if not args.with_report:
        omissions.append({"step": "report", "reason": "optional stage not requested"})
    for stage in stages:
        for record in stage["source_code"]:
            if record.get("missing"):
                blockers.append(f"Missing stage source: {record['path']}")
    for name in ("dem_reader_patch", "dem_reader_proof", "model_freeze", "requirements"):
        if inputs[name].get("missing"):
            blockers.append(f"Missing required prepared input: {name}")
    if not (args.snap_runtime / "jre/Contents/Home/bin/java").is_file():
        blockers.append("Prepared local SNAP Java runtime is missing")
    return {"schema": SCHEMA, "created_at": now(), "status": "execution_plan_only", "out": str(output), "source_runner": file_record(scripts / "run_fresh_case.py"), "python_executable": python, "case": {"bbox": bbox, "context_bbox": context, "event_date": args.event_date, "osm_snapshot": args.osm_snapshot}, "external_inputs": inputs, "environment": {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}, "limits": {"stage_timeout_seconds": args.stage_timeout_seconds, "max_total_seconds": args.max_total_seconds, "log_bytes_per_stream": LOG_BYTES, "native_response_body_budget_bytes": args.native_budget_bytes, "dem_response_body_budget_bytes": args.dem_budget_bytes, "osm_response_body_budget_bytes": OSM_BYTES, "osm_requests": 1}, "stages": stages, "omitted_steps": omissions, "blockers": blockers, "execution": {"subprocesses_run": 0, "network_requests_made_by_launcher": 0, "network_scope": "none; offline plan only; future stage behavior is declared above"}, "claim_scope": "water-state engineering pipeline only; optional exposure/report remains provisional", "full_track_b_requirements_satisfied": False, "fresh_case_unseen_validation": False, "reference_labels_loaded": False, "gpu_used": False, "retraining_or_retuning": False}


def walk_bound_records(value):
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= value.keys():
            yield value
        for child in value.values():
            yield from walk_bound_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_bound_records(child)


def validate_saved_inputs(plan, *, require_model_environment=True):
    if plan.get("schema") != SCHEMA or plan.get("reference_labels_loaded") is not False:
        raise ValueError("Expected the source-bound runner execution plan")
    for record in walk_bound_records(plan.get("external_inputs", {})):
        verify_record(record)
    verify_record(plan["source_runner"])
    freeze_record = plan["external_inputs"].get("model_freeze")
    if not freeze_record or freeze_record.get("missing"):
        raise ValueError("Frozen checkpoint metadata is absent")
    freeze = read_json(verify_record(freeze_record))
    for key in ("model", "protocol"):
        verify_record(freeze[key])
    for item in freeze.get("code", []):
        verify_record(item)
    if freeze.get("threshold") != .5:
        raise ValueError("Runner cannot retune the frozen checkpoint threshold")
    if require_model_environment and importlib.metadata.version("scikit-learn") != "1.7.2":
        raise ValueError("Run in one uv environment with scikit-learn==1.7.2; no automatic install/retraining")
    runtime = Path(plan["external_inputs"]["snap_runtime"]["path"])
    if not (runtime / "jre/Contents/Home/bin/java").is_file():
        raise ValueError("Prepared local SNAP runtime is absent")
    return {"schema": "akash-fresh-source-validation-v1", "status": "checked", "checked_at": now(), "external_file_records_checked": len(list(walk_bound_records(plan["external_inputs"]))), "model_sha256": freeze["model"]["sha256"], "model_threshold": .5, "scikit_learn_contract": "1.7.2", "model_loaded": False, "subprocesses_run_by_validation": 0, "network_requests": 0, "reference_labels_loaded": False, "gpu_used": False}


def fetch_bounded_historic_osm(bbox, snapshot, event_date, out, *, session=None):
    """Run the existing helper through a one-request bounded local transport.

    The helper's transport binding is replaced only inside this dedicated stage
    process and restored in finally. No global requests patch or retry occurs.
    """
    from akash_flood import osm as osm_module
    import requests
    bbox = finite_bbox(bbox, "OSM extent")
    _, _, cutoff = event_boundary(event_date, snapshot)
    if cutoff.time() != datetime.min.time():
        raise ValueError("New OSM retrieval currently requires a UTC-midnight snapshot")
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError("Historical OSM retrieval needs a new output directory")
    client = session if session is not None else requests.Session()
    client.trust_env = False
    if getattr(client, "auth", None) or any(key.casefold() in {"authorization", "proxy-authorization"} for key in getattr(client, "headers", {})):
        raise ValueError("Historical map retrieval does not accept credentials")
    started = time.monotonic()
    transport = {"requests": 0, "received_body_bytes": 0, "endpoint": OSM_ENDPOINT, "maximum_body_bytes": OSM_BYTES, "maximum_requests": 1, "deadline_seconds": OSM_DEADLINE_SECONDS, "redirects_followed": False, "tls_verified": True}
    class BoundedResponse:
        def __init__(self, value):
            self.value = value
        def raise_for_status(self):
            return None
        def json(self):
            return self.value
    class BoundedTransport:
        @staticmethod
        def get(url, *, params, timeout, headers):
            if url != OSM_ENDPOINT or transport["requests"]:
                raise ValueError("Only one request to the declared historical OSM HTTPS endpoint is permitted")
            transport["requests"] += 1
            response = client.get(url, params=params, timeout=(15, 20), headers=headers, stream=True, allow_redirects=False, verify=True)
            try:
                if response.status_code != 200:
                    raise ValueError(f"Historical OSM returned {response.status_code}; no retries/current-data fallback")
                length = response.headers.get("Content-Length")
                if length and (not length.isdigit() or int(length) > OSM_BYTES):
                    raise ValueError("Historical OSM advertised response exceeds the byte cap")
                chunks = []
                for chunk in response.iter_content(65536):
                    if time.monotonic() - started > OSM_DEADLINE_SECONDS:
                        raise TimeoutError("Historical OSM wall deadline exceeded")
                    if chunk:
                        transport["received_body_bytes"] += len(chunk)
                        if transport["received_body_bytes"] > OSM_BYTES:
                            raise ValueError("Historical OSM response exceeds the byte cap")
                        chunks.append(chunk)
                if length and int(length) != transport["received_body_bytes"]:
                    raise ValueError("Historical OSM response is incomplete")
                raw = b"".join(chunks)
                transport["raw_response_sha256"] = hashlib.sha256(raw).hexdigest()
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("Historical OSM response must be a JSON object")
                return BoundedResponse(value)
            finally:
                response.close()
    original_transport = osm_module.requests
    osm_module.requests = BoundedTransport()
    try:
        data, provenance = osm_module.fetch_historical(bbox, cutoff.date().isoformat(), event_date, endpoint=OSM_ENDPOINT)
    finally:
        osm_module.requests = original_transport
    out.mkdir(parents=True)
    write_json(out / "osm.json", data)
    write_json(out / "provenance.json", provenance)
    transport.update(status="completed", completed_at=now(), elapsed_seconds=time.monotonic()-started, osm=file_record(out / "osm.json"), provenance=file_record(out / "provenance.json"), query=provenance["query"], scope="new server-executed historical body query; no current map fallback; per-element edit timestamps not requested")
    write_json(out / "network-receipt.json", transport)
    return transport


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)


def run_command(argv, *, cwd, environment, stdout_path, stderr_path, timeout_seconds, log_cap=LOG_BYTES):
    """No shell; cap both output streams and terminate the child process group."""
    started = time.monotonic()
    env = dict(os.environ)
    env.update(environment)
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    status, reason = "completed", None
    sizes = {"stdout": 0, "stderr": 0}
    selector = selectors.DefaultSelector()
    files = {}
    try:
        for name, pipe, path in (("stdout", process.stdout, stdout_path), ("stderr", process.stderr, stderr_path)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
            files[name] = Path(path).open("xb")
        while selector.get_map():
            if time.monotonic() - started > timeout_seconds:
                status, reason = "timeout", "stage or remaining total deadline exceeded"
                stop_process(process)
                break
            for key, _ in selector.select(timeout=.25):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                name = key.data
                available = max(0, log_cap - sizes[name])
                files[name].write(chunk[:available])
                sizes[name] += min(len(chunk), available)
                if len(chunk) > available:
                    status, reason = "log_limit", f"{name} exceeded {log_cap} bytes; retained bounded prefix"
                    stop_process(process)
                    break
            if status != "completed":
                break
        if process.poll() is None:
            try:
                process.wait(timeout=max(.01, timeout_seconds - (time.monotonic()-started)))
            except subprocess.TimeoutExpired:
                status, reason = "timeout", "stage or remaining total deadline exceeded"
                stop_process(process)
        return {"exit_code": process.returncode, "status": status, "failure_reason": reason, "elapsed_seconds": time.monotonic()-started, "stdout_bytes": sizes["stdout"], "stderr_bytes": sizes["stderr"]}
    finally:
        if process.poll() is None:
            stop_process(process)
        selector.close()
        for stream in files.values():
            stream.close()
        for pipe in (process.stdout, process.stderr):
            if pipe:
                pipe.close()


def validate_artifact(expected):
    path = Path(expected["path"])
    result = file_record(path) | {"kind": expected["kind"]}
    kind = expected["kind"]
    if kind == "file":
        return result
    value = read_json(path)
    if value.get("reference_labels_loaded") is True:
        raise ValueError("Pipeline result unexpectedly consumed reference labels")
    if kind == "source_validation" and value.get("status") != "checked":
        raise ValueError("Source validation did not pass")
    elif kind == "catalog":
        parse_discovery(value)
    elif kind == "case_plan":
        from akash_flood.fresh_native import scenes_from_plan
        scenes_from_plan(path)
    elif kind == "native_index" and (value.get("schema") != "akash-fresh-native-acquisition-v1" or value.get("status") != "completed"):
        raise ValueError("Native acquisition did not complete")
    elif kind == "dem_manifest" and (value.get("schema") != "akash-fresh-dem-v1" or not value.get("completed_at")):
        raise ValueError("DEM preparation did not complete")
    elif kind == "processing_record" and (value.get("schema") != "akash-fresh-s1-processing-v1" or value.get("status") != "completed"):
        raise ValueError("SNAP processing did not complete")
    elif kind == "water_manifest" and (value.get("schema") != "akash-fresh-water-v1" or value.get("available") is not True):
        raise ValueError("Water-state inference did not complete")
    elif kind == "exposure_summary" and (value.get("schema") != "akash-fresh-exposure-v1" or value.get("status") != "completed"):
        raise ValueError("Fresh exposure analysis did not complete")
    elif kind == "exposure_manifest" and (value.get("schema") != "akash-fresh-exposure-binding-v1" or value.get("status") != "completed"):
        raise ValueError("Fresh exposure bindings did not complete")
    elif kind == "report" and value.get("schema") != "akash-fresh-report-v1":
        raise ValueError("Fresh source-bound report was not generated")
    elif kind == "network_receipt" and value.get("status") != "completed":
        raise ValueError("Historical OSM retrieval did not complete")
    if kind not in {"file", "source_validation", "catalog", "case_plan", "native_index", "dem_manifest", "processing_record", "water_manifest", "exposure_summary", "exposure_manifest", "report", "network_receipt", "osm", "osm_provenance"}:
        raise ValueError("Unrecognized required stage artifact kind: " + kind)
    return result


def execute_plan(plan, *, command_runner=run_command):
    output = Path(plan["out"])
    if output.exists() or output.is_symlink():
        raise FileExistsError("Fresh execution root must not already exist")
    if plan["blockers"]:
        raise ValueError("Execution plan is blocked: " + "; ".join(plan["blockers"]))
    # No network subprocess starts until every external supplied input and stage
    # code identity has been checked against the newly constructed plan.
    for record in walk_bound_records(plan.get("external_inputs", {})):
        verify_record(record)
    for stage in plan["stages"]:
        for record in stage["source_code"]:
            verify_record(record)
    output.mkdir(parents=True)
    write_json(output / "execution-plan.json", plan)
    state = {"schema": SCHEMA, "status": "running", "started_at": now(), "execution_plan": file_record(output / "execution-plan.json"), "case": plan["case"], "environment": plan["environment"], "stages": [], "omitted_steps": plan["omitted_steps"], "full_track_b_requirements_satisfied": False, "fresh_case_unseen_validation": False, "reference_labels_loaded": False, "gpu_used": False, "retraining_or_retuning": False, "claim_scope": "water-only pipeline until required actual water-stage artifacts pass; optional exposure/report never establish debris or unseen local accuracy"}
    receipt = output / "execution-receipt.json"
    write_json(receipt, state)
    start = time.monotonic()
    for stage in plan["stages"]:
        item = {"name": stage["name"], "argv": stage["argv"], "cwd": stage["cwd"], "network_scope": stage["network_scope"], "status": "starting", "started_at": now(), "source_code_before_run": []}
        state["stages"].append(item)
        try:
            for record in stage["source_code"]:
                item["source_code_before_run"].append(file_record(verify_record(record)))
            for record in walk_bound_records(plan.get("external_inputs", {})):
                verify_record(record)
            remaining = plan["limits"]["max_total_seconds"] - (time.monotonic()-start)
            if remaining <= 0:
                raise TimeoutError("Total launcher deadline exhausted before next stage")
            logs = output / "logs"; logs.mkdir(exist_ok=True)
            stdout, stderr = logs / (stage["name"] + ".stdout.log"), logs / (stage["name"] + ".stderr.log")
            result = command_runner(stage["argv"], cwd=stage["cwd"], environment=plan["environment"], stdout_path=stdout, stderr_path=stderr, timeout_seconds=min(remaining, stage["timeout_seconds"]), log_cap=plan["limits"]["log_bytes_per_stream"])
            item.update(result, stdout=file_record(stdout), stderr=file_record(stderr))
            if result["status"] != "completed" or result["exit_code"] != 0:
                raise RuntimeError(result.get("failure_reason") or f"Stage exited with {result['exit_code']}")
            item["artifacts"] = [validate_artifact(artifact) for artifact in stage["artifacts"]]
            # Freeze receipt identities only after the required artifacts passed.
            item.update(status="completed", completed_at=now())
            write_json(receipt, state, replace=True)
            print(stage["name"], "completed", round(result["elapsed_seconds"], 3), "seconds", flush=True)
        except Exception as error:
            item.update(status="failed", failure_type=type(error).__name__, failure_reason=str(error), failed_at=now())
            state.update(status="failed", stopped_at_stage=stage["name"], completed_at=now(), elapsed_seconds=time.monotonic()-start, omitted_remaining_stages=[remaining_stage["name"] for remaining_stage in plan["stages"][len(state["stages"]):]], no_automatic_retry=True)
            write_json(receipt, state, replace=True)
            raise
    completed = {stage["name"] for stage in state["stages"] if stage["status"] == "completed"}
    scope = "completed water-state engineering pipeline"
    if "map_exposure" in completed:
        scope += " with source-bound provisional mapped-feature overlaps"
    if "build_report" in completed:
        scope += " and generated report"
    state.update(status="completed", completed_at=now(), elapsed_seconds=time.monotonic()-start, claim_scope=scope + "; no debris, operational passability, full Track B or unseen local accuracy validation")
    write_json(receipt, state, replace=True)
    return state


def parser(root=ROOT):
    root = Path(root)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bbox", type=float, nargs=4)
    p.add_argument("--context-bbox", type=float, nargs=4)
    p.add_argument("--context-buffer-deg", type=float, default=.03)
    p.add_argument("--event-date")
    p.add_argument("--osm-snapshot")
    p.add_argument("--catalog", type=Path)
    p.add_argument("--osm", type=Path)
    p.add_argument("--osm-provenance", type=Path)
    p.add_argument("--omit-osm", action="store_true")
    p.add_argument("--source-cache", type=Path)
    p.add_argument("--reuse-dem-tile", type=Path)
    p.add_argument("--snap-runtime", type=Path, default=root / "data/snap-runtime/portable")
    p.add_argument("--dem-reader-patch", type=Path, default=root / "data/snap-external-dem-pixel-correction/manifest.json")
    p.add_argument("--dem-reader-proof", type=Path, default=root / "outputs/snap-dem-center-proof/proof.json")
    p.add_argument("--model-freeze", type=Path, default=root / "outputs/sen1-cpu-v1/model-evaluation-freeze.json")
    p.add_argument("--geoid", type=Path, default=root / "data/copdem-ellipsoid/us_nga_egm08_25.tif")
    p.add_argument("--requirements", type=Path, default=root.parent / "research/source-extracts/Track B.txt")
    p.add_argument("--max-days", type=int, default=24)
    p.add_argument("--native-budget-bytes", type=int, default=4_100_000_000)
    p.add_argument("--dem-budget-bytes", type=int, default=400_000_000)
    p.add_argument("--stage-timeout-seconds", type=int, default=1800)
    p.add_argument("--max-total-seconds", type=int, default=3600)
    p.add_argument("--with-exposure", action="store_true")
    p.add_argument("--with-report", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--_fetch-historical-osm", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_validate-inputs", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--input-spec", type=Path, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args._fetch_historical_osm:
        if not args.bbox or not args.event_date or not args.osm_snapshot:
            p.error("Internal OSM stage needs bbox/event-date/osm-snapshot")
        result = fetch_bounded_historic_osm(args.bbox, args.osm_snapshot, args.event_date, args.out)
    elif args._validate_inputs:
        if not args.input_spec:
            p.error("Internal source validation needs --input-spec")
        if args.out.exists():
            raise FileExistsError("Use fresh source validation output")
        result = validate_saved_inputs(read_json(args.input_spec))
        args.out.mkdir(parents=True)
        result["execution_plan"] = file_record(args.input_spec)
        write_json(args.out / "receipt.json", result)
    else:
        if not args.bbox or not args.event_date or not args.osm_snapshot:
            p.error("Supply bbox/event-date/osm-snapshot")
        plan = build_plan(args)
        if args.execute:
            result = execute_plan(plan)
        else:
            args.out.mkdir(parents=True)
            write_json(args.out / "execution-plan.json", plan)
            result = {"status": plan["status"], "execution_plan": file_record(args.out / "execution-plan.json"), "stage_names": [stage["name"] for stage in plan["stages"]], "blockers": plan["blockers"], "subprocesses_run": 0, "network_requests": 0, "full_track_b_requirements_satisfied": False, "fresh_case_unseen_validation": False}
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
