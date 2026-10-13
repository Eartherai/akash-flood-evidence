"""Verify offline planning, ordered failure, source receipts and bounded stages."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fresh_case_runner", ROOT / "scripts/run_fresh_case.py")
launch = importlib.util.module_from_spec(spec); spec.loader.exec_module(launch)

BBOX = [85.30, 28.12, 85.43, 28.32]
CONTEXT = [85.27, 28.09, 85.46, 28.35]


def basic_args(tmp_path, *extra):
    catalog = tmp_path / "catalog.json"
    if not catalog.exists():
        catalog.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    return launch.parser().parse_args(["--bbox", *map(str, BBOX), "--event-date", "2026-08-26", "--osm-snapshot", "2026-07-27", "--catalog", str(catalog), "--omit-osm", "--out", str(tmp_path / "case"), *map(str, extra)])


def test_default_only_writes_plan_and_never_starts_subprocess_or_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline planning attempted execution")
    monkeypatch.setattr(launch.subprocess, "Popen", forbidden)
    import requests
    monkeypatch.setattr(requests, "Session", forbidden)
    args = basic_args(tmp_path)
    command = ["--bbox", *map(str, BBOX), "--event-date", args.event_date, "--osm-snapshot", args.osm_snapshot, "--catalog", str(args.catalog), "--omit-osm", "--out", str(args.out)]
    result = launch.main(command)
    assert result["subprocesses_run"] == result["network_requests"] == 0
    assert list(args.out.iterdir()) == [args.out / "execution-plan.json"]
    plan = launch.read_json(args.out / "execution-plan.json")
    names = [stage["name"] for stage in plan["stages"]]
    assert names == ["validate_sources", "plan_case", "acquire_native", "prepare_dem", "process_s1", "infer_water"]
    assert plan["fresh_case_unseen_validation"] is False
    assert plan["full_track_b_requirements_satisfied"] is False
    assert all(stage["argv"][0] == sys.executable for stage in plan["stages"])
    assert all(isinstance(argument, str) for stage in plan["stages"] for argument in stage["argv"])
    assert plan["environment"]["OMP_NUM_THREADS"] == "4"
    with pytest.raises(FileExistsError):
        launch.main(command + ["--execute"])


def test_discovery_and_historical_osm_precede_planner_with_explicit_context(tmp_path):
    args = launch.parser().parse_args(["--bbox", *map(str, BBOX), "--context-bbox", *map(str, CONTEXT), "--event-date", "2026-08-26", "--osm-snapshot", "2026-07-27", "--out", str(tmp_path / "case")])
    plan = launch.build_plan(args)
    names = [stage["name"] for stage in plan["stages"]]
    assert names[:4] == ["validate_sources", "discover", "retrieve_historic_osm", "plan_case"]
    discover = plan["stages"][1]["argv"]
    bbox_offset = discover.index("--bbox") + 1
    assert list(map(float, discover[bbox_offset:bbox_offset+4])) == CONTEXT
    assert discover[discover.index("--context-buffer-deg") + 1] == "0"
    osm = plan["stages"][2]
    assert "one HTTPS request" in osm["network_scope"]
    planner = plan["stages"][3]["argv"]
    assert "--osm" in planner and "--osm-provenance" in planner


def test_input_and_resource_overrides_cannot_be_silently_ignored(tmp_path):
    args = basic_args(tmp_path, "--snap-runtime", tmp_path / "different-runtime")
    with pytest.raises(ValueError, match="cannot be silently ignored"):
        launch.build_plan(args)
    args = basic_args(tmp_path, "--with-report")
    plan = launch.build_plan(args)
    assert any("historic OSM" in reason for reason in plan["blockers"])
    assert not any(stage["name"] == "build_report" for stage in plan["stages"])
    args = basic_args(tmp_path, "--native-budget-bytes", "4100000001")
    with pytest.raises(ValueError, match="native network budget"):
        launch.build_plan(args)
    args = basic_args(tmp_path, "--context-bbox", "85.31", "28.13", "85.42", "28.31")
    with pytest.raises(ValueError, match="completely cover"):
        launch.build_plan(args)


def synthetic_plan(tmp_path, count=3):
    stages = []
    for index in range(count):
        source = tmp_path / f"stage-{index}.py"; source.write_text("# synthetic fixture\n")
        artifact = tmp_path / "run" / f"stage-{index}" / "artifact.json"
        stages.append({"name": f"stage_{index}", "argv": [sys.executable, str(source)], "cwd": str(tmp_path), "network_scope": "none; synthetic fixture", "source_code": [launch.file_record(source)], "timeout_seconds": 2, "artifacts": [{"path": str(artifact), "kind": "water_manifest"}]})
    return {"schema": launch.SCHEMA, "out": str(tmp_path / "run"), "case": {"bbox": BBOX}, "source_runner": launch.file_record(ROOT / "scripts/run_fresh_case.py"), "external_inputs": {}, "environment": {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}, "limits": {"max_total_seconds": 30, "log_bytes_per_stream": 1000}, "stages": stages, "omitted_steps": [{"step": "debris_detection", "reason": "not implemented"}], "blockers": []}


def fake_runner(plan, *, fail_at=None, mutate_at=None, invalid_artifact_at=None):
    calls = []
    def run(argv, **kwargs):
        index = len(calls); calls.append(argv)
        Path(kwargs["stdout_path"]).write_text(f"stage {index} fixture\n")
        Path(kwargs["stderr_path"]).write_text("")
        if index == mutate_at:
            Path(plan["stages"][index+1]["source_code"][0]["path"]).write_text("# changed after plan\n")
        if index != fail_at:
            path = Path(plan["stages"][index]["artifacts"][0]["path"])
            path.parent.mkdir(parents=True)
            launch.write_json(path, {"schema": "akash-fresh-water-v1", "available": True, "reference_labels_loaded": index == invalid_artifact_at})
        return {"status": "completed", "exit_code": 7 if index == fail_at else 0, "elapsed_seconds": .001, "failure_reason": None}
    return run, calls


def test_failed_stage_stops_without_retry_and_preserves_source_log_receipts(tmp_path):
    plan = synthetic_plan(tmp_path)
    runner, calls = fake_runner(plan, fail_at=1)
    with pytest.raises(RuntimeError, match="exited with 7"):
        launch.execute_plan(plan, command_runner=runner)
    assert len(calls) == 2
    receipt = launch.read_json(tmp_path / "run/execution-receipt.json")
    assert receipt["status"] == "failed" and receipt["stopped_at_stage"] == "stage_1"
    assert receipt["no_automatic_retry"] is True
    assert receipt["omitted_remaining_stages"] == ["stage_2"]
    failed = receipt["stages"][1]
    assert failed["exit_code"] == 7
    assert launch.sha256(failed["stdout"]["path"]) == failed["stdout"]["sha256"]
    assert failed["source_code_before_run"][0]["sha256"] == plan["stages"][1]["source_code"][0]["sha256"]
    with pytest.raises(FileExistsError):
        launch.execute_plan(plan, command_runner=runner)


def test_source_mutation_between_stages_fails_before_next_command(tmp_path):
    plan = synthetic_plan(tmp_path)
    runner, calls = fake_runner(plan, mutate_at=0)
    with pytest.raises(ValueError, match="Source bytes changed"):
        launch.execute_plan(plan, command_runner=runner)
    assert len(calls) == 1
    receipt = launch.read_json(tmp_path / "run/execution-receipt.json")
    assert receipt["stopped_at_stage"] == "stage_1"
    assert receipt["stages"][0]["status"] == "completed"


def test_result_artifacts_are_required_and_reference_inputs_are_rejected(tmp_path):
    plan = synthetic_plan(tmp_path)
    runner, calls = fake_runner(plan, invalid_artifact_at=0)
    with pytest.raises(ValueError, match="consumed reference"):
        launch.execute_plan(plan, command_runner=runner)
    assert len(calls) == 1
    receipt = launch.read_json(tmp_path / "run/execution-receipt.json")
    assert receipt["status"] == "failed"
    assert receipt["full_track_b_requirements_satisfied"] is False


def test_success_keeps_water_only_and_false_unseen_full_track_claims(tmp_path):
    plan = synthetic_plan(tmp_path, count=1)
    runner, calls = fake_runner(plan)
    receipt = launch.execute_plan(plan, command_runner=runner)
    assert len(calls) == 1
    assert receipt["status"] == "completed"
    assert receipt["full_track_b_requirements_satisfied"] is False
    assert receipt["fresh_case_unseen_validation"] is False
    assert "no debris" in receipt["claim_scope"]
    assert receipt["stages"][0]["artifacts"][0]["sha256"] == launch.sha256(plan["stages"][0]["artifacts"][0]["path"])


@pytest.mark.parametrize('kind', ['exposure_summary', 'exposure_manifest', 'report', 'unsupported_future_stage'])
def test_optional_output_cannot_pass_with_unrelated_success_json(tmp_path, kind):
    p=tmp_path/'artifact.json';p.write_text('{"status":"completed","schema":"unrelated"}')
    with pytest.raises(ValueError):
        launch.validate_artifact({'path':str(p),'kind':kind})


class FakeResponse:
    def __init__(self, raw, status=200):
        self.raw, self.status_code = raw, status
        self.headers = {"Content-Length": str(len(raw))}
        self.closed = False
    def iter_content(self, chunk_size):
        yield self.raw
    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, response):
        self.response, self.calls = response, []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_historical_osm_uses_one_bounded_tls_request_and_preserves_query(tmp_path):
    data = {"elements": [{"type": "node", "id": 1, "lat": 28.2, "lon": 85.35}]}
    session = FakeSession(FakeResponse(json.dumps(data).encode()))
    result = launch.fetch_bounded_historic_osm(CONTEXT, "2026-07-27", "2026-08-26", tmp_path / "osm", session=session)
    assert result["requests"] == 1
    assert result["received_body_bytes"] > 0
    assert session.calls[0][1]["verify"] is True and session.calls[0][1]["allow_redirects"] is False
    assert session.trust_env is False
    assert '[date:"2026-07-27T00:00:00Z"]' in result["query"]
    assert "no current map fallback" in result["scope"]
    assert launch.sha256(result["osm"]["path"]) == result["osm"]["sha256"]


def test_oversized_or_failed_osm_never_falls_back_or_leaves_success_artifact(tmp_path, monkeypatch):
    from akash_flood import osm
    original = osm.requests
    monkeypatch.setattr(launch, "OSM_BYTES", 10)
    session = FakeSession(FakeResponse(b"12345678901"))
    with pytest.raises(ValueError, match="exceeds the byte cap"):
        launch.fetch_bounded_historic_osm(CONTEXT, "2026-07-27", "2026-08-26", tmp_path / "too-big", session=session)
    assert len(session.calls) == 1 and not (tmp_path / "too-big").exists()
    assert osm.requests is original
    session = FakeSession(FakeResponse(b"{}", status=503))
    with pytest.raises(ValueError, match="no retries/current-data fallback"):
        launch.fetch_bounded_historic_osm(CONTEXT, "2026-07-27", "2026-08-26", tmp_path / "failed", session=session)
    assert len(session.calls) == 1 and not (tmp_path / "failed").exists()


def test_subprocess_output_is_bounded_without_shell(tmp_path):
    stdout, stderr = tmp_path / "stdout.log", tmp_path / "stderr.log"
    result = launch.run_command([sys.executable, "-c", "import sys;sys.stdout.write('x'*10000)"], cwd=str(tmp_path), environment={"OMP_NUM_THREADS": "4"}, stdout_path=stdout, stderr_path=stderr, timeout_seconds=2, log_cap=100)
    assert result["status"] == "log_limit"
    assert stdout.stat().st_size == 100
    assert stderr.stat().st_size <= 100


def test_execution_timeout_terminates_local_child_group(tmp_path):
    result = launch.run_command([sys.executable, "-c", "import time;time.sleep(10)"], cwd=str(tmp_path), environment={}, stdout_path=tmp_path / "stdout.log", stderr_path=tmp_path / "stderr.log", timeout_seconds=.05, log_cap=100)
    assert result["status"] == "timeout"
    assert result["exit_code"] is not None
