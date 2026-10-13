"""Additive bounded raw workflow: radar, optical, observation support and reports.

The original launcher and every prior artifact remain unchanged. Sources are
frozen before each new stage; no reference labels, fitting or retuning occurs.
Planning runs no subprocess/network. Execution preserves failures and budgets.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace
import run_fresh_case as base
from akash_flood.fresh_spawn_20261012 import spawn_run_command

ROOT = base.ROOT
read_json = base.read_json
verify_record = base.verify_record
walk_bound_records = base.walk_bound_records
file_record = base.file_record
WORKFLOW = 'akash-bounded-multimodal-workflow-v1'


def parser(root=ROOT):
    p = base.parser(root)
    p.description = __doc__
    p.add_argument('--with-optical', action='store_true')
    p.add_argument('--with-observation-access', action='store_true')
    p.add_argument('--optical-budget-bytes', type=int, default=60_000_000)
    p.add_argument('--optical-timeout-seconds', type=int, default=180)
    p.add_argument('--osm-provider',choices=('fossgis','private-coffee'),default='private-coffee',
        help='One historical provider selected before execution; no automatic fallback')
    return p


def build_plan(args, *, root=ROOT):
    root = Path(root).resolve()
    values = vars(args).copy()
    # Final reports are scheduled after the additive layers. The base launcher
    # still owns raw input validation, pairing, terrain correction and inference.
    wanted_report = args.with_report
    values['with_report'] = False
    values['with_exposure'] = args.with_exposure or wanted_report or args.with_observation_access
    plan = base.build_plan(SimpleNamespace(**values), root=root)
    base.bounded_int(args.optical_budget_bytes, 1, 100_000_000, 'optical response-body budget')
    base.bounded_int(args.optical_timeout_seconds, 1, 600, 'optical deadline')
    if args.with_optical and args.max_days>24:
        raise ValueError('The fixed optical adapter supports at most 24 days per side; reduce the search window for a multimodal run')
    out = Path(plan['out'])
    python = plan['python_executable']
    scripts = root/'scripts'
    catalog = (Path(plan['external_inputs']['catalog']['path'])
               if 'catalog' in plan['external_inputs'] else out/'discovery/catalog.json')
    case_plan = out/'planning/plan.json'
    wrapper = file_record(Path(__file__).resolve())
    plan['external_inputs']['additive_workflow_source'] = wrapper
    plan['external_inputs']['spawn_helper_source'] = file_record(root/'src/akash_flood/fresh_spawn_20261012.py')
    plan['workflow_schema'] = WORKFLOW
    plan['source_workflow'] = wrapper
    plan['layers_requested'] = {'optical': bool(args.with_optical),
        'observation_access': bool(args.with_observation_access), 'report': bool(wanted_report)}
    plan['historic_osm_provider']=args.osm_provider
    for stage in plan['stages']:
        if stage['name']=='retrieve_historic_osm':
            helper=scripts/'fetch_historic_osm_20261012.py'
            stage['argv']=[str(x) for x in [python,helper,'--bbox',*plan['case']['context_bbox'],
                '--event-date',args.event_date,'--osm-snapshot',args.osm_snapshot,
                '--provider',args.osm_provider,'--out',out/'historical-osm']]
            stage['source_code']=[file_record(helper),file_record(root/'scripts/run_fresh_case.py'),
                file_record(root/'src/akash_flood/osm.py'),wrapper]
            stage['network_scope']='One declared '+args.osm_provider+' historical Overpass request; same pre-event cutoff/body query, 30MB/120s; no retry/current-data fallback'
            stage['timeout_seconds']=min(args.stage_timeout_seconds,120)
    plan['limits'].update(optical_response_body_budget_bytes=args.optical_budget_bytes,
                          optical_deadline_seconds=args.optical_timeout_seconds)

    def add(name, argv, artifacts, modules=(), network='none', timeout=None):
        paths = [Path(argv[1]), Path(__file__).resolve(), root/'src/akash_flood/fresh_spawn_20261012.py', *[root/p for p in modules]]
        records = [file_record(p) if p.is_file() else {'path':str(p),'missing':True} for p in paths]
        plan['stages'].append({'name':name,'argv':[str(x) for x in argv], 'cwd':str(root),
            'network_scope':network,'required':True,'source_code':records,
            'artifacts':[{'path':str(p),'kind':kind} for p,kind in artifacts],
            'timeout_seconds':timeout or args.stage_timeout_seconds})
        for r in records:
            if r.get('missing'):
                plan['blockers'].append('Missing additive stage source: '+r['path'])

    if args.with_optical:
        optical = out/'optical'
        adapter = scripts/'run_fresh_optical_20261012_v3.py'
        add('freeze_optical_sources', [python,adapter,'--prepare-only','--catalog',catalog,
            '--bbox',*plan['case']['bbox'],'--event-date',args.event_date,'--plan',case_plan,
            '--max-days',args.max_days,
            '--budget-bytes',args.optical_budget_bytes,'--timeout-seconds',args.optical_timeout_seconds,
            '--out',optical], [(optical/'protocol.json','fresh_optical_protocol')],
            ('src/akash_flood/optical_composite.py','src/akash_flood/optical_change.py',
             'scripts/build_s2_debris_candidates.py'))
        add('acquire_and_screen_optical', [python,adapter,'--execute','--out',optical],
            [(optical/'manifest.json','fresh_optical_manifest')],
            ('src/akash_flood/optical_composite.py','src/akash_flood/optical_change.py',
             'scripts/build_s2_debris_candidates.py'),
            network='New anonymous source-bound Sentinel-2 COG/SCL range requests; fixed pair, bounded bytes/time, no fallback',
            timeout=min(args.stage_timeout_seconds,args.optical_timeout_seconds+30))
        plan['omitted_steps'] = [r for r in plan['omitted_steps'] if r['step'] not in {'optical_analysis','debris_detection'}]
        plan['omitted_steps'].append({'step':'validated_debris_detection',
            'reason':'Fixed optical surface-change hypotheses and explicit unknowns; local debris accuracy remains unvalidated'})
    if args.with_observation_access:
        adapter=scripts/'run_fresh_observation_access_20261012.py'
        protocol=out/'observation-access-protocol.json'
        add('freeze_observation_scenario', [python,adapter,'freeze','--exposure-manifest',
            out/'exposure/source-and-output-manifest.json','--out',protocol],
            [(protocol,'fresh_observation_protocol')],('src/akash_flood/observation_access.py',))
        add('analyze_observation_access', [python,adapter,'run','--protocol',protocol,
            '--out',out/'observation-access'],
            [(out/'observation-access/receipt.json','fresh_observation_receipt')],
            ('src/akash_flood/observation_access.py',),timeout=min(args.stage_timeout_seconds,330))
        plan['omitted_steps']=[r for r in plan['omitted_steps'] if r['step']!='connectivity_and_observed_road_closures']
        plan['omitted_steps'].append({'step':'observed_road_closures_and_operability',
            'reason':'Conditional fixed-target graph bounds are computed; physical closures, passability and facility operation remain unobserved'})
    if wanted_report:
        report_argv=[python,scripts/'build_fresh_report.py','--water-manifest',out/'water/manifest.json',
            '--exposure-manifest',out/'exposure/source-and-output-manifest.json','--out',out/'report']
        if args.with_optical:
            report_argv += ['--optical-manifest',out/'optical/manifest.json']
        add('build_native_layer_report',report_argv,[(out/'report/report.json','report')])
        summary_argv=[python,scripts/'build_multimodal_case_report_20261012.py',
            '--report-binding',out/'report/source-and-output-manifest.json',
            '--execution-plan',out/'execution-plan.json','--out',out/'combined-report']
        if args.with_observation_access:
            summary_argv += ['--observation-receipt',out/'observation-access/receipt.json']
        add('build_combined_case_report',summary_argv,
            [(out/'combined-report/manifest.json','fresh_combined_report')])
        plan['omitted_steps']=[r for r in plan['omitted_steps'] if r['step']!='report']
    plan['claim_scope']='Bounded raw-source engineering workflow; acquired optical candidates and observation scenario only when their required stages complete. No unrestricted-area, hazard-accuracy or passability guarantee.'
    plan['full_track_b_requirements_satisfied']=False
    return plan


def validate_additive_artifact(expected):
    kind=expected['kind']
    if not kind.startswith('fresh_'):
        return base.validate_artifact(expected)
    p=Path(expected['path']); value=read_json(p)
    if value.get('reference_labels_loaded') is True or value.get('retraining_or_retuning') is True:
        raise ValueError('Unexpected reference or training input in additive stage')
    if kind=='fresh_optical_manifest':
        if value.get('schema')!='akash-s2-surface-change-v1' or value.get('status')!='completed':
            raise ValueError('Optical execution did not produce the compatible completed manifest')
    elif kind in {'fresh_observation_receipt','fresh_combined_report'}:
        schemas={'fresh_observation_receipt':'akash-fresh-observation-access-receipt-v1',
                 'fresh_combined_report':'akash-fresh-combined-report-v1'}
        if value.get('status')!='completed' or value.get('schema')!=schemas[kind]:
            raise ValueError('Required additive stage did not complete')
    elif kind in {'fresh_optical_protocol','fresh_observation_protocol'}:
        schemas={'fresh_optical_protocol':'akash-s2-surface-change-protocol-v1',
                 'fresh_observation_protocol':'akash-fresh-observation-access-protocol-v1'}
        if value.get('schema')!=schemas[kind]:
            raise ValueError('Wrong additive source/scenario protocol schema')
    else:
        raise ValueError('Unknown additive artifact kind')
    return file_record(p)|{'kind':kind}


def execute_plan(plan, *, command_runner=spawn_run_command):
    if plan.get('workflow_schema')!=WORKFLOW:
        raise ValueError('Expected the fixed additive workflow plan')
    verify_record(plan['source_workflow'])
    # Scoped adapter hook: the original module/file is not changed. The worker
    # process runs one plan at a time and restores its helper even on failure.
    original=base.validate_artifact
    def check(expected):
        if expected['kind'].startswith('fresh_'):
            return validate_additive_artifact(expected)
        return original(expected)
    base.validate_artifact=check
    try:
        state=base.execute_plan(plan,command_runner=command_runner)
    except BaseException:
        receipt=Path(plan['out'])/'execution-receipt.json'
        if receipt.is_file():
            failed=read_json(receipt)
            failed['workflow_schema']=WORKFLOW
            failed['layers_requested']=plan['layers_requested']
            failed['claim_scope']='Incomplete bounded multimodal workflow; only recorded completed stages produced checked artifacts. No complete-layer or mapping-accuracy claim.'
            if failed.get('status')=='running':
                failed['status']='interrupted'
                failed['no_automatic_retry']=True
            base.write_json(receipt,failed,replace=True)
        raise
    finally:
        base.validate_artifact=original
    state['workflow_schema']=WORKFLOW
    state['layers_requested']=plan['layers_requested']
    state['claim_scope']=plan['claim_scope']
    state['full_track_b_requirements_satisfied']=False
    base.write_json(Path(plan['out'])/'execution-receipt.json',state,replace=True)
    return state


def main(argv=None):
    import json
    p=parser();args=p.parse_args(argv)
    if not args.bbox or not args.event_date or not args.osm_snapshot:
        p.error('Supply bbox/event-date/osm-snapshot')
    if args._fetch_historical_osm or args._validate_inputs:
        p.error('Internal base-stage commands use the preserved base runner')
    plan=build_plan(args)
    if args.execute:
        result=execute_plan(plan)
    else:
        args.out.mkdir(parents=True)
        base.write_json(args.out/'execution-plan.json',plan)
        result={'status':'execution_plan_only','execution_plan':file_record(args.out/'execution-plan.json'),
            'stages':[s['name'] for s in plan['stages']],'blockers':plan['blockers'],
            'network_requests':0,'full_track_b_requirements_satisfied':False}
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    return result


if __name__=='__main__':
    main()
