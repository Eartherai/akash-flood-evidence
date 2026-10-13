"""Once-only separately worded AI-authored QA; not human/field validation."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

from akash_flood.copilot import FactStore,ask_local_model,MODEL

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'outputs/copilot'
FREEZE=FOLDER/'qa-independent-freeze.json'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    output=FOLDER/'qa-independent-results.json';started_file=FOLDER/'qa-independent.started.json'
    if output.exists() or started_file.exists():raise SystemExit('This QA run already has a record; refusing repeat')
    source=ROOT/'outputs/access-bounds-proof/proof.json';freeze=json.loads(FREEZE.read_text())
    assert sha(source)==freeze['source_sha256']
    code=ROOT/'src/akash_flood/copilot.py'
    record={'started_at':datetime.now(timezone.utc).isoformat(),'status':'running','code_sha256':sha(code),'qa_sha256':sha(FREEZE),'source_sha256':sha(source),'model':MODEL,'scope':freeze['scope'],'results':[]}
    started_file.write_text(json.dumps({key:value for key,value in record.items() if key!='results'},indent=2)+'\n')
    store=FactStore.from_path(source);start=time.perf_counter()
    for case in freeze['cases']:
        assert sha(code)==record['code_sha256']
        row={'id':case['id'],'language':case['language'],'question':case['question']};t=time.perf_counter()
        try:
            response=ask_local_model(store,case['question'],language=case['language'])
            raw=response['model_plan']
            row.update(response=response,source_plan_valid=True,
                       system_status_correct=response['status']==case['expected_status'],
                       raw_model_status_correct=raw['status']==case['expected_status'],
                       required_facts_present=set(case['expected_fact_ids'])<=set(raw['fact_ids']),
                       policy_guard_corrected=response.get('policy_guard',{}).get('corrected_model_plan',False))
            row['passed']=row['system_status_correct'] and row['required_facts_present']
        except Exception as error:
            row.update(passed=False,source_plan_valid=False,error=str(error))
        row['seconds']=round(time.perf_counter()-t,4);record['results'].append(row)
        output.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({key:row[key] for key in ['id','passed','seconds']}),flush=True)
    record.update(status='complete',elapsed_seconds=round(time.perf_counter()-start,3),questions=len(record['results']),
                  passed=sum(r['passed'] for r in record['results']),
                  source_plans_valid=sum(r['source_plan_valid'] for r in record['results']),
                  raw_model_status_correct=sum(r.get('raw_model_status_correct',False) for r in record['results']),
                  policy_guard_corrections=sum(r.get('policy_guard_corrected',False) for r in record['results']),
                  language_review='No human Nepali or semantic relevance review; fact inclusion/status checks only')
    output.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    started_file.write_text(json.dumps({'status':'complete','results_sha256':sha(output),'code_sha256':sha(code),'qa_sha256':sha(FREEZE)},indent=2)+'\n')
    print(json.dumps({key:record[key] for key in ['questions','passed','source_plans_valid','raw_model_status_correct','policy_guard_corrections','elapsed_seconds']}))

if __name__=='__main__':main()
