"""Lifecycle guarantees for the additive worker; fixtures are not real runs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/'scripts'))
import run_multimodal_fresh_20261012 as worker
import run_fresh_case as base
from build_multimodal_case_report_20261012 import run as build_report


class WorkflowLifecycleTests(unittest.TestCase):
    def test_changed_frozen_workflow_stops_before_execution_root_or_commands(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);code=folder/'workflow.py';code.write_text('# frozen\n')
            frozen=base.file_record(code);code.write_text('# changed\n')
            plan={'workflow_schema':worker.WORKFLOW,'source_workflow':frozen,
                  'out':str(folder/'run')}
            called=[]
            with self.assertRaisesRegex(ValueError,'Source bytes changed'):
                worker.execute_plan(plan,command_runner=lambda *a,**k:called.append(a))
            self.assertEqual(called,[])
            self.assertFalse((folder/'run').exists())

    def test_missing_required_optical_artifact_preserves_failure_and_stops_next_stage(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);code=folder/'workflow.py';code.write_text('# fixture source\n')
            out=folder/'run';source=base.file_record(code)
            stages=[{'name':name,'argv':['fixture',name],'cwd':str(folder),
                'network_scope':'fixture only; no network','required':True,
                'source_code':[source],'artifacts':artifacts,'timeout_seconds':5}
                for name,artifacts in [('optical', [{'path':str(out/'optical/manifest.json'),
                    'kind':'fresh_optical_manifest'}]),('must_not_run',[])]]
            plan={'schema':base.SCHEMA,'workflow_schema':worker.WORKFLOW,
                'source_workflow':source,'source_runner':source,'out':str(out),
                'blockers':[],'external_inputs':{},'case':{},'environment':{},
                'stages':stages,'omitted_steps':[],'layers_requested':{'optical':True},
                'limits':{'max_total_seconds':10,'log_bytes_per_stream':1000},
                'claim_scope':'Synthetic fixture only'}
            seen=[]
            def command(argv,**kwargs):
                seen.append(argv[1])
                Path(kwargs['stdout_path']).write_text('fixture stage exited normally\n')
                Path(kwargs['stderr_path']).write_text('')
                return {'status':'completed','exit_code':0,'elapsed_seconds':.001}
            original=base.validate_artifact
            with self.assertRaises(FileNotFoundError):
                worker.execute_plan(plan,command_runner=command)
            self.assertIs(base.validate_artifact,original)
            self.assertEqual(seen,['optical'])
            receipt=json.loads((out/'execution-receipt.json').read_text())
            self.assertEqual(receipt['status'],'failed')
            self.assertEqual(receipt['stopped_at_stage'],'optical')
            self.assertEqual(receipt['omitted_remaining_stages'],['must_not_run'])
            self.assertFalse(receipt['full_track_b_requirements_satisfied'])
            self.assertEqual(receipt['workflow_schema'],worker.WORKFLOW)
            self.assertEqual(receipt['layers_requested'],{'optical':True})

    def test_report_rejects_case_mismatch_before_creating_output(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);report=folder/'report.json';binding=folder/'binding.json'
            plan=folder/'plan.json';out=folder/'report-output'
            report.write_text(json.dumps({'case':{'bbox':[1,2,3,4],'event_date':'2024-09-30'}}))
            binding.write_text(json.dumps({'schema':'akash-fresh-report-binding-v1','status':'completed',
                'case':{'bbox':[1,2,3,4],'event_date':'2024-09-30'},
                'inputs':[],'outputs':{'report.json':base.file_record(report)}}))
            plan.write_text(json.dumps({'case':{'bbox':[1,2,3,4],'event_date':'2024-10-01'}}))
            with self.assertRaisesRegex(ValueError,'Case mismatch'):
                build_report(binding,plan,out)
            self.assertFalse(out.exists())

    def test_same_aoi_event_cannot_attach_report_from_another_osm_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);report=folder/'report.json';binding=folder/'binding.json'
            plan=folder/'plan.json';out=folder/'report-output'
            case={'bbox':[1,2,3,4],'context_bbox':[.9,1.9,3.1,4.1],'event_date':'2024-09-30'}
            report.write_text(json.dumps({'case':case,'historical_snapshot':'2024-09-01T00:00:00Z'}))
            binding.write_text(json.dumps({'schema':'akash-fresh-report-binding-v1','status':'completed',
                'case':case,'inputs':[],'outputs':{'report.json':base.file_record(report)}}))
            plan.write_text(json.dumps({'case':dict(case,osm_snapshot='2024-09-02')}))
            with self.assertRaisesRegex(ValueError,'Case mismatch'):
                build_report(binding,plan,out)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
