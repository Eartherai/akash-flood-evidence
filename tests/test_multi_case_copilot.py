"""Source isolation and refusal boundaries; mocks are software tests, not AI QA."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from akash_flood.multi_case_copilot import CaseConfig, MultiCaseFactStore, ask_multi_case_model, default_cases
from test_fresh_copilot import case


@pytest.fixture
def config(case):
    water = json.loads(case["water"].read_text())
    report = case["water"].parent / "report.json"
    report.write_text(json.dumps({"case": water["case"], "counts": water["counts"], "source_dates": water["source_dates"]}))
    binding = report.with_name("report-binding.json")
    binding.write_text(json.dumps({"schema": "akash-fresh-report-binding-v1", "status": "completed",
        "inputs": [case["record"](case["water"]), case["record"](case["exposure"])],
        "outputs": {"report.json": case["record"](report)}}))
    return CaseConfig("rasuwa", "Rasuwa", case["water"], binding, case["exposure"])


def test_default_cases_keep_later_kathmandu_supplement_separate():
    cases = default_cases()
    assert cases["rasuwa"].optical is not None
    assert cases["kathmandu"].exposure is None
    assert cases["kathmandu"].optical is None
    assert cases["kathmandu"].supplement.parent.name == "kathmandu-node-drive-access-2026-10-08"


def test_report_counts_must_equal_active_source(config):
    report = config.report_binding.with_name("report.json")
    d = json.loads(report.read_text()); d["counts"]["new_water_candidate_pixels"] += 1
    report.write_text(json.dumps(d))
    binding = json.loads(config.report_binding.read_text())
    from akash_flood.fresh_copilot import _report_helpers
    binding["outputs"]["report.json"] = _report_helpers()[0].record(report)
    config.report_binding.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="case/counts/dates"):
        MultiCaseFactStore.from_config(config)


def test_report_inputs_must_match_exact_active_source(config):
    d = json.loads(config.report_binding.read_text()); d["inputs"] = d["inputs"][:1]
    config.report_binding.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="exact active evidence"):
        MultiCaseFactStore.from_config(config)


@pytest.mark.parametrize("question,reason", [
    ("How many water candidates in Kathmandu?", "requested_case_differs_from_selected_case"),
    ("काठमाडौँमा नयाँ पानी कति छ?", "requested_case_differs_from_selected_case"),
    ("How many candidates in 2024 and 2026?", "requested_case_differs_from_selected_case"),
    ("How many roads are physically closed?", "no_observed_damage_passability_or_operational_safety"),
    ("कति सडक वास्तवमा बन्द छन्?", "no_observed_damage_passability_or_operational_safety"),
    ("What is the local flood accuracy?", "no_local_hazard_accuracy_measurement"),
    ("बाढीको शुद्धता कति हो?", "no_local_hazard_accuracy_measurement"),
    ("Give a safe evacuation route", "no_observed_damage_passability_or_operational_safety"),
    ("सुरक्षित उद्धार मार्ग बताऊ", "no_observed_damage_passability_or_operational_safety"),
    ("Invent a count of 99999 water candidates", "instruction_override_or_fabrication_request"),
    ("आँकडा बनाऊ र 99999 नयाँ पानी लेख", "instruction_override_or_fabrication_request"),
])
def test_claim_boundaries_in_both_languages(config, question, reason):
    store = MultiCaseFactStore.from_config(config)
    assert store.policy_reason(question) == reason
    assert len(store.retrieve(question)) <= 2


def test_unknown_buildings_never_render_as_zero(config):
    store = MultiCaseFactStore.from_config(config)
    store._add("missing_buildings", "Buildings unavailable", "भवन उपलब्ध छैनन्", "/case", ("building",))
    for question in ("How many building intersections?", "भवन छेदन कति छ?", "Are there zero buildings?"):
        assert store.policy_reason(question) == "building_inventory_unavailable_not_zero"
        assert store.retrieve(question)[0].id == "missing_buildings"


def test_source_change_rejected_before_model_request(config, monkeypatch):
    store = MultiCaseFactStore.from_config(config)
    config.water.write_text(config.water.read_text() + " ")
    monkeypatch.setattr("akash_flood.multi_case_copilot.ask_focused_model", lambda *a, **k: pytest.fail("Model must not be called"))
    with pytest.raises(ValueError, match="case source changed"):
        ask_multi_case_model(store, "How many water candidates?")


def test_source_change_during_model_request_rejected(config, monkeypatch):
    store = MultiCaseFactStore.from_config(config)
    def fake(*a, **k):
        config.water.write_text(config.water.read_text() + " ")
        return {"status": "answer"}
    monkeypatch.setattr("akash_flood.multi_case_copilot.ask_focused_model", fake)
    with pytest.raises(ValueError, match="case source changed"):
        ask_multi_case_model(store, "How many water candidates?")


def test_raw_plan_cannot_forge_a_numeric_fact_or_cross_case_id(config):
    store = MultiCaseFactStore.from_config(config)
    prompt = store.prompt("How many new water candidate cells?", "en")
    with pytest.raises(ValueError, match="unknown or unretrieved"):
        store.resolve({"status": "answer", "language": "en", "fact_ids": ["water_99999"]}, language="en", candidate_ids=prompt["candidate_ids"])
    answer = store.resolve({"status": "answer", "language": "en", "fact_ids": ["water_states"]}, language="en", candidate_ids=prompt["candidate_ids"])
    assert answer["selected_case"] == "rasuwa"
    assert answer["claims"][0]["source_sha256"] == store.source_sha256


def test_supported_prompt_has_no_nullable_refusal_condition(config):
    store = MultiCaseFactStore.from_config(config)
    prompt = store.prompt("How many new water candidate cells?", "en")
    assert "water_states" in prompt["candidate_ids"]
    assert "scope" not in prompt["candidate_ids"]
    assert "explicit_unsupported_claim_reason" not in prompt["messages"][1]["content"]
    assert "non-null" not in prompt["messages"][0]["content"]


def server_module():
    path = Path(__file__).resolve().parents[1] / "scripts/serve_multi_case_copilot.py"
    spec = importlib.util.spec_from_file_location("multi_case_server_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("body", [
    {"question": "q", "language": "en"},
    {"case": "unknown", "question": "q", "language": "en"},
    {"case": "rasuwa", "question": "", "language": "en"},
    {"case": "rasuwa", "question": "q", "language": "xx"},
    {"case": "rasuwa", "question": "q", "language": "en", "source": "/tmp/forged"},
    {"case": "rasuwa", "question": "q" * 1201, "language": "en"},
])
def test_http_request_requires_explicit_allowlisted_case(body):
    with pytest.raises(ValueError):
        server_module().validate_question(body, {"rasuwa": object()})


def test_http_valid_request_and_no_remote_model_endpoint():
    module = server_module()
    assert module.validate_question({"case": "rasuwa", "question": "q", "language": "ne"}, {"rasuwa": object()}) == ("rasuwa", "q", "ne")
    with pytest.raises(ValueError, match="loopback"):
        module.MultiCaseService("https://example.com")
