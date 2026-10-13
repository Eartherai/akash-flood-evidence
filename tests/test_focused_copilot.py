"""Focused retrieval is a development heuristic, not measured model fluency."""
import json
import io
from pathlib import Path

import pytest

from test_fresh_copilot import case
from akash_flood.fresh_copilot import FreshFact, FreshCaseFactStore
from akash_flood.focused_copilot import FocusedCaseFactStore
import akash_flood.focused_copilot as module


def store(case, *, exposure=True, optical=False):
    value = FocusedCaseFactStore.from_paths(case['water'], case['exposure'] if exposure else None)
    if optical:
        # Synthetic lexical fixture only; it does not create optical evidence.
        for key, tags in {'optical_coverage': ('optical', 'coverage', 'assessed', 'unknown'),
                          'optical_candidates': ('optical', 'surface', 'candidate', 'count', 'change'),
                          'optical_limits': ('optical', 'limits', 'accuracy'),
                          'optical_acquisition': ('optical', 'date', 'before', 'after')}.items():
            value.facts[key] = FreshFact(key, 'Synthetic optical lexical test only', 'synthetic', '/synthetic',
                                         tags, value.source_sha256, value.source_path)
        value.facts.pop('missing_optical')
    return value


def ids(value, question):
    return [f.id for f in value.ranked_facts(question)]


@pytest.mark.parametrize('question,expected', [
    ('How many new water candidate cells and lost water cells are there?', 'water_states'),
    ('How much water grid coverage is assessed or unknown?', 'water_coverage'),
    ('What are the before and after Sentinel-1 acquisition dates?', 'water_acquisition'),
    ('नयाँ पानी र हराएको पानी उम्मेदवारको गणना कति छ?', 'water_states'),
    ('पानीको मूल्याङ्कन गरिएको र अज्ञात क्षेत्र कति छ?', 'water_coverage'),
])
def test_water_questions_do_not_retrieve_other_layers(case, question, expected):
    value = store(case, optical=True); selected = ids(value, question)
    assert expected in selected
    assert all(f.startswith('water_') or f == 'scope' for f in selected)


def test_explicit_optical_domain_keeps_own_counts_and_limits(case):
    value = store(case, optical=True)
    selected = ids(value, 'How many optical surface candidates and what are their limits?')
    assert {'optical_candidates', 'optical_limits'} <= set(selected)
    assert not any(f.startswith('water_') or f.startswith('exposure_') for f in selected)


def test_cross_layer_question_preserves_both_coverage_domains(case):
    value = store(case, optical=True)
    selected = ids(value, 'Compare water and optical assessed coverage and unknown cells.')
    assert {'water_coverage', 'optical_coverage'} <= set(selected)


def test_one_layer_multi_part_question_preserves_coverage_and_candidates(case):
    value = store(case, optical=True)
    selected = ids(value, 'For the optical layer, how much coverage is assessed and how many surface-change candidate cells are flagged?')
    assert {'optical_coverage', 'optical_candidates'} <= set(selected)


def test_named_status_plus_destination_retains_both_explicit_facets(case):
    value = store(case)
    selected = ids(value, 'गोठेनको मोडेल गरिएको पहुँच अवस्था र निश्चित नक्सागत गन्तव्य बताउनुहोस्।')
    assert {'settlement_6473257a6cf5', 'settlement_6473257a6cf5_destination'} <= set(selected)


def test_feature_overlap_mentions_water_mask_without_grid_totals(case):
    selected = ids(store(case), 'How many road corridors and building footprints overlap new-water candidates?')
    assert 'exposure_intersections' in selected
    assert not any(f.startswith('water_') for f in selected)


def test_named_status_and_destination_are_distinct_and_keep_assumptions(case):
    value = store(case)
    status = ids(value, 'गोठेनको पहुँच अवस्था बताउनुहोस्।')
    assert 'settlement_6473257a6cf5' in status and 'access_assumptions' in status
    assert 'settlement_6473257a6cf5_destination' not in status
    destination = ids(value, 'What is the nearest mapped destination for गोठेन and how is nearest defined?')
    assert {'settlement_6473257a6cf5_destination', 'access_method', 'access_assumptions'} <= set(destination)
    assert 'access_counts' not in destination


def test_unknown_named_id_does_not_become_arbitrary_named_fact(case):
    value = store(case)
    assert value.policy_reason('node/2695690340 status') == 'requested_settlement_not_in_bound_source'
    assert not any(f.startswith('settlement_') for f in ids(value, 'node/2695690340 status'))


def test_historical_map_query_retrieves_its_source_not_network_totals(case):
    selected = ids(store(case), 'What historical OpenStreetMap snapshot underlies roads and destinations?')
    assert selected == ['osm_snapshot', 'scope']


def test_missing_optional_sources_remain_unavailable(case):
    value = store(case, exposure=False)
    assert value.policy_reason('What is optical assessed coverage?') == 'no_bound_optical_artifact'
    assert value.policy_reason('Which settlement is accessible?') == 'no_bound_historical_exposure_or_access_artifact'
    assert 'missing_optical' in ids(value, 'What optical coverage exists?')


def test_unrelated_request_has_no_arbitrary_case_counts(case):
    value = store(case)
    assert ids(value, 'Write a sorting tutorial and recommend a gaming laptop.') == ['scope']
    assert value.policy_reason('Write a sorting tutorial and recommend a gaming laptop.') == 'question_has_no_bound_case_evidence'


def test_short_prompt_keeps_active_case_and_real_fact_selector_contract(case):
    focused = store(case)
    base = FreshCaseFactStore.from_paths(case['water'], case['exposure'])
    question = 'How many new water candidate cells are there?'
    prompt = focused.prompt(question, 'ne'); original = base.prompt(question, 'ne')
    data = json.loads(prompt['messages'][1]['content'])
    assert data['active_case']['event_date'] == '2026-08-26'
    assert data['active_case']['water_observation_times'] == focused.manifest['source_dates']
    assert len(json.dumps(prompt)) < len(json.dumps(original))
    assert all(set(fact) == {'id', 'claim'} for fact in data['candidate_facts'])
    assert str(Path(__file__).resolve().parents[1] / 'src/akash_flood/focused_copilot.py') in {r['path'] for r in focused.source_bindings}
    plan = {'status': 'answer', 'language': 'ne', 'fact_ids': ['water_states']}
    answer = focused.resolve(plan, language='ne', candidate_ids=prompt['candidate_ids'])
    assert answer['claims'][0]['text'] == focused.facts['water_states'].ne
    assert answer['retrieval_policy']['relative_score_cutoff'] == .6


def test_focused_adapter_preserves_validation_not_just_retrieval(case):
    case['model'].write_bytes(b'changed model source')
    with pytest.raises(ValueError, match='checksum'):
        FocusedCaseFactStore.from_paths(case['water'], case['exposure'])


def model_response(monkeypatch, plan, *, finish='stop', raw_plan=None):
    calls = []
    def response(request, timeout):
        calls.append(json.loads(request.data))
        raw = {'choices': [{'finish_reason': finish, 'message': {'content': raw_plan or json.dumps(plan)}}],
               'usage': {'completion_tokens': 10}}
        return io.BytesIO(json.dumps(raw).encode())
    monkeypatch.setattr(module, 'urlopen', response)
    return calls


def test_guarded_empty_answer_becomes_explicit_refusal_with_raw_plan_preserved(case, monkeypatch):
    value = store(case)
    raw = {'status': 'answer', 'language': 'en', 'fact_ids': []}
    calls = model_response(monkeypatch, raw)
    answer = module.ask_focused_model(value, 'Confirm all bridges are safe for ambulances.')
    assert answer['status'] == 'decline' and answer['claims'] == []
    assert answer['model_plan'] == raw and answer['policy_guard']['corrected_model_plan'] is True
    assert calls[0]['temperature'] == 0 and calls[0]['response_format']['type'] == 'json_schema'


def test_unguarded_empty_answer_remains_invalid(case, monkeypatch):
    model_response(monkeypatch, {'status': 'answer', 'language': 'en', 'fact_ids': []})
    with pytest.raises(ValueError, match='requires a source'):
        module.ask_focused_model(store(case), 'How many new water candidate cells are there?')


@pytest.mark.parametrize('plan', [
    {'status': 'answer', 'language': 'en', 'fact_ids': ['invented_fact']},
    {'status': 'decline', 'language': 'hi', 'fact_ids': []},
    {'status': 'decline', 'language': 'en', 'fact_ids': [], 'free_text': 'invented'},
])
def test_policy_never_hides_invalid_ids_language_or_generated_fields(case, monkeypatch, plan):
    model_response(monkeypatch, plan)
    with pytest.raises(ValueError, match='Invalid raw'):
        module.ask_focused_model(store(case), 'Confirm all bridges are safe for ambulances.')


def test_duplicate_raw_keys_and_truncation_are_not_refusals(case, monkeypatch):
    model_response(monkeypatch, None, raw_plan='{"status":"decline","status":"answer","language":"en","fact_ids":[]}')
    with pytest.raises(ValueError, match='Duplicate'):
        module.ask_focused_model(store(case), 'Safe ambulance route?')
    model_response(monkeypatch, {'status': 'decline', 'language': 'en', 'fact_ids': []}, finish='length')
    with pytest.raises(ValueError, match='Truncated'):
        module.ask_focused_model(store(case), 'Safe ambulance route?')


def test_valid_multi_fact_plan_retains_exact_language_clauses(case, monkeypatch):
    value = store(case)
    plan = {'status': 'answer', 'language': 'ne', 'fact_ids': ['settlement_6473257a6cf5', 'settlement_6473257a6cf5_destination']}
    model_response(monkeypatch, plan)
    answer = module.ask_focused_model(value, 'गोठेनको पहुँच अवस्था र निश्चित गन्तव्य बताउनुहोस्।', language='ne')
    assert answer['status'] == 'answer' and answer['model_plan'] == plan
    assert all(claim['text'] == value.facts[claim['fact_id']].ne for claim in answer['claims'])
