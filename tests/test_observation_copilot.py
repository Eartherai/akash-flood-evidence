"""Synthetic source isolation tests; no graph/model/human validation is claimed."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from akash_flood.multi_case_copilot import MultiCaseFactStore, ask_multi_case_model, default_cases
from akash_flood.observation_copilot import CLAIMS, ObservationCaseFactStore, SUMMARY_RELATIVE, observation_cases
from test_fresh_copilot import case, rebind, save
from test_multi_case_copilot import config


@pytest.fixture
def audit(case, config, tmp_path):
    # Enrich the existing synthetic original proof before activating its store.
    old = json.loads(case['proof'].read_text())
    old['settlements'][0]['snap_node_id'] = 'source-node'
    save(case['proof'], old); rebind(case, 'proof')
    binding = json.loads(config.report_binding.read_text())
    binding['inputs'] = [case['record'](config.water), case['record'](config.exposure)]
    save(config.report_binding, binding)
    original = MultiCaseFactStore.from_config(config)
    root = tmp_path
    def record(path):
        raw = path.read_bytes()
        return {'path':str(path.relative_to(root)), 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}
    protocol = root/'protocol.json'; save(protocol, {'synthetic':True})
    checker = root/'independent.json'; save(checker, {'synthetic':True})
    summary = {'schema':'akash-observation-access-summary-v1', 'protocol':record(protocol),
        'independent_verification':record(checker), 'cases':{}}
    audit_root = root/'outputs'/'audit'
    support = {'projected_centerline_total_m':15, 'lengths_m':{'assessed':10,'unknown_within_aoi':5,'outside_aoi':0},
        'fractions_of_projected_centerline':{'assessed':2/3,'unknown_within_aoi':1/3,'outside_aoi':0}}
    counts = {'ambiguous':1}
    scopes = {'all_source':counts, 'inside_water_aoi':{}, 'inside_extraction_context':counts, 'incidental_outside_context':{}}
    paths = {}
    for slug in ('rasuwa','kathmandu'):
        folder = audit_root/(slug+'-v1'); folder.mkdir(parents=True)
        row = {k:old['settlements'][0][k] for k in ('id','name','snap_node_id','nearest_baseline_destination')}
        row.update(previous_status='ambiguous',status='ambiguous',point_coordinates=[85.29,28.1],
                   inside_water_aoi=False,inside_extraction_context=True,path_support=support)
        expanded = {'schema':'akash-observation-access-bounds-v1','case':original.case,
            'confirmed_closures':0,'certain_closed_edge_ids':[], 'destination_switching_after_closure':False,
            'scenario_policy':{'observed_nonwater_is_passable':False, 'unknown_is_observed_closed':False,
                'positive_area_tolerance_m2':1e-6, 'certain_closures':[]},
            'settlements':[row], 'status_counts':counts, 'scope_status_counts':scopes}
        witnesses = {'schema':'akash-original-witness-support-v1','case':original.case,
            'witness_count':1,'witnesses':[{'id':row['id'],'support':support}]}
        ep=save(folder/'expanded-scenario.json',expanded); wp=save(folder/'original-witness-support.json',witnesses)
        outputs=[record(wp),record(ep)]
        receipt={'schema':'akash-observation-access-receipt-v1','status':'completed','case':original.case,
            'claims':CLAIMS,'all_frozen_inputs_unchanged':True,'protocol':record(protocol),'outputs':outputs,
            'expanded_status_counts':counts,'original_status_counts':counts,'scope_status_counts':scopes,
            'uncertain_physical_group_count':2,'uncertain_directed_edge_count':4}
        rp=save(folder/'receipt.json',receipt)
        summary['cases'][slug]={'receipt':record(rp),'full_evidence':outputs,'expanded_status_counts':counts,
            'original_status_counts':counts,'scope_status_counts':scopes,'original_witness_statistics':{
                'count':1,'traversal_weighted_fractions':support['fractions_of_projected_centerline']},
            'zero_length_graph_witness_ids':[]}
        paths[slug]={'receipt':rp,'expanded':ep,'witness':wp}
    sp=root/SUMMARY_RELATIVE;sp.parent.mkdir(parents=True);save(sp,summary)
    cases={'rasuwa':config,'kathmandu':replace(config,slug='kathmandu',label='Kathmandu')}
    def refresh(slug):
        receipt=json.loads(paths[slug]['receipt'].read_text())
        receipt['outputs']=[record(paths[slug]['witness']),record(paths[slug]['expanded'])]
        save(paths[slug]['receipt'],receipt)
        summary=json.loads(sp.read_text()); item=summary['cases'][slug]
        item['receipt']=record(paths[slug]['receipt']);item['full_evidence']=receipt['outputs']
        save(sp,summary)
    def load():
        configs=observation_cases(audit_root,source_root=root,cases=cases)
        return ObservationCaseFactStore.from_verified_store(original,configs['rasuwa'])
    return dict(root=root,audit_root=audit_root,cases=cases,original=original,paths=paths,
                summary=sp,record=record,refresh=refresh,load=load,case=case)


def test_default_configuration_and_original_store_remain_exact(audit):
    before=[f.as_dict() for f in audit['original'].facts.values()]
    store=audit['load']()
    assert before==[f.as_dict() for f in audit['original'].facts.values()]
    assert not hasattr(default_cases()['rasuwa'],'observation_receipt')
    assert all(f.id.startswith('newscenario_') or f.id in audit['original'].facts for f in store.facts.values())
    assert store.facts['access_counts'].en.startswith('Original candidate-only scenario: ')


def test_both_completed_receipts_and_all_exact_outputs_required(audit):
    audit['paths']['kathmandu']['receipt'].unlink()
    with pytest.raises(FileNotFoundError):audit['load']()


@pytest.mark.parametrize('filename',['expanded','witness','receipt'])
def test_hash_mutation_blocks_activation_for_either_case(audit,filename):
    path=audit['paths']['kathmandu'][filename];path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='hash/size'):audit['load']()


def test_rebound_wrong_case_cannot_supply_observation_facts(audit):
    path=audit['paths']['rasuwa']['expanded'];data=json.loads(path.read_text())
    data['case']['case_id']='other-case';save(path,data);audit['refresh']('rasuwa')
    with pytest.raises(ValueError,match='Cross-case'):audit['load']()


def test_fixed_target_cannot_change_even_with_rebound_hash(audit):
    path=audit['paths']['rasuwa']['expanded'];data=json.loads(path.read_text())
    data['settlements'][0]['nearest_baseline_destination']['destination_id']='another-target'
    save(path,data);audit['refresh']('rasuwa')
    with pytest.raises(ValueError,match='fixed target'):audit['load']()


def test_scope_must_follow_actual_bound_case_coordinates(audit):
    path=audit['paths']['rasuwa']['expanded'];data=json.loads(path.read_text())
    data['settlements'][0]['inside_water_aoi']=True;save(path,data);audit['refresh']('rasuwa')
    with pytest.raises(ValueError,match='AOI scope'):audit['load']()


def test_zero_support_fractions_are_null_never_one(audit):
    ep=audit['paths']['rasuwa']['expanded'];data=json.loads(ep.read_text())
    data['settlements'][0]['path_support']['projected_centerline_total_m']=0
    data['settlements'][0]['path_support']['fractions_of_projected_centerline']={k:None for k in ('assessed','unknown_within_aoi','outside_aoi')}
    save(ep,data);audit['refresh']('rasuwa')
    summary=json.loads(audit['summary'].read_text())
    summary['cases']['rasuwa']['zero_length_graph_witness_ids']=[data['settlements'][0]['id']];save(audit['summary'],summary)
    store=audit['load']();identifier=store.observation_settlement_ids[data['settlements'][0]['id']]+'_support'
    assert 'null support fractions, never 100%' in store.facts[identifier].en
    data['settlements'][0]['path_support']['fractions_of_projected_centerline']['assessed']=1
    save(ep,data);audit['refresh']('rasuwa')
    with pytest.raises(ValueError,match='must remain null'):audit['load']()


def test_unavailable_row_support_is_not_zero_or_full_coverage(audit):
    ep=audit['paths']['rasuwa']['expanded'];data=json.loads(ep.read_text())
    data['settlements'][0]['path_support']=None;save(ep,data);audit['refresh']('rasuwa')
    store=audit['load']();identifier=store.observation_settlement_ids[data['settlements'][0]['id']]+'_support'
    assert 'No graph-path support record is available' in store.facts[identifier].en
    assert '100%' not in store.facts[identifier].en


@pytest.mark.parametrize('question,reason',[
    ('Are roads passable to a functioning hospital?','no_observed_damage_passability_or_operational_safety'),
    ('सञ्चालनमा रहेको अस्पताल बताऊ','no_observed_damage_passability_or_operational_safety'),
    ('Expanded counts for Kathmandu?','requested_case_differs_from_selected_case'),
    ('काठमाडौँको विस्तारित अवस्था कति छ?','requested_case_differs_from_selected_case'),
])
def test_case_and_safety_boundaries(audit,question,reason):
    assert audit['load']().policy_reason(question)==reason


def test_new_named_support_and_comparison_retrieval_are_allowlisted(audit):
    store=audit['load']()
    prompt=store.prompt('Expanded route support for node/269569034','en')
    identifier=store.observation_settlement_ids['node/269569034']+'_support'
    assert identifier in prompt['candidate_ids']
    answer=store.resolve({'status':'answer','language':'en','fact_ids':[identifier]},language='en',candidate_ids=prompt['candidate_ids'])
    assert '66.67% assessed' in answer['answer']
    assert answer['claims'][0]['source_path']==str(audit['paths']['rasuwa']['expanded'])
    with pytest.raises(ValueError,match='retrieved source fact'):
        store.resolve({'status':'answer','language':'en','fact_ids':['newscenario_assumptions']},language='en',candidate_ids=prompt['candidate_ids'])
    with pytest.raises(ValueError,match='unknown or unretrieved'):
        store.resolve({'status':'answer','language':'en','fact_ids':['newscenario_forged_999']},language='en',candidate_ids=prompt['candidate_ids'])
    ordinary=store.retrieve('How many settlements have conditional access?')
    assert ordinary[0].id=='newscenario_counts'
    assert all(f.en.startswith('Original candidate-only scenario: ') for f in ordinary if f.id.startswith('access_'))


def test_changed_bound_evidence_rejected_before_model_call(audit,monkeypatch):
    store=audit['load']();path=audit['paths']['rasuwa']['witness'];path.write_text(path.read_text()+' ')
    monkeypatch.setattr('akash_flood.multi_case_copilot.ask_focused_model',lambda *a,**k:pytest.fail('Model must not be called'))
    with pytest.raises(ValueError,match='case source changed'):ask_multi_case_model(store,'Expanded counts?')


def test_metadata_excludes_named_rows_and_default_mode_is_original(audit,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/serve_observation_copilot_20261011.py'
    spec=importlib.util.spec_from_file_location('observation_server_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module._base.MultiCaseFactStore,'from_config',lambda config:audit['original'])
    original=module.ObservationCaseService(cases=audit['cases'])
    assert original.observation_enabled is False
    assert 'observation_extension' not in original.metadata()
    active=module.ObservationCaseService(cases=audit['cases'],source_root=audit['root'],observation_audit_root=audit['audit_root'])
    meta=active.metadata()
    assert len(meta['cases'])==2 and 'observation_extension' in meta
    assert all(not f['id'].startswith('newscenario_settlement_') for c in meta['cases'] for f in c['facts'])
    module_path=Path(__file__).resolve().parents[1]/'src/akash_flood/observation_copilot.py'
    assert str(module_path) in {r['path'] for r in active.code_bindings}
