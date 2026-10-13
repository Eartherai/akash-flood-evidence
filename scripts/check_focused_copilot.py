"""Check focused prompting in a separately frozen development run.

Source-clause equality is separate from question relevance. Neither this
challenge nor Nepali output is independent human QA or hazard validation.
Earlier first-run outcomes and these cases are known development information;
this is not a new blind holdout. No model or detector training is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

from akash_flood.copilot import MODEL
from akash_flood.focused_copilot import FocusedCaseFactStore, ask_focused_model as ask_local_model
from build_fresh_report import record

ROOT = Path(__file__).resolve().parents[1]


def verify_records(records):
    if any(record(Path(item['path'])) != item for item in records):
        raise ValueError('A frozen source, code, model or challenge file changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--water-manifest', type=Path, required=True)
    parser.add_argument('--exposure-manifest', type=Path)
    parser.add_argument('--optical-manifest', type=Path)
    parser.add_argument('--challenge', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT / 'data/copilot-model' / MODEL['file'])
    parser.add_argument('--prior-development-result', type=Path, action='append', default=[],
                        help='Bind known earlier development results; repeated checks remain nonblind')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output path; earlier responses and failures must be preserved')
    store = FocusedCaseFactStore.from_paths(args.water_manifest, args.exposure_manifest, args.optical_manifest)
    challenge = json.loads(args.challenge.read_text())
    if challenge.get('schema') != 'akash-fresh-copilot-development-challenge-v1':
        raise ValueError('Require the declared nonblind development challenge schema')
    cases = challenge['cases']
    if (not isinstance(cases, list) or not 1 <= len(cases) <= 32
            or len({case['id'] for case in cases}) != len(cases)):
        raise ValueError('Use 1–32 distinct development cases')
    for case in cases:
        if case['language'] not in ('en', 'ne') or case['status'] not in ('answer', 'decline'):
            raise ValueError('Invalid challenge language or status')
        if any(key not in store.facts for key in case['required']):
            raise ValueError('Required challenge fact is unavailable')
    model_record = record(args.model)
    if model_record['bytes'] != MODEL['bytes'] or model_record['sha256'] != MODEL['sha256']:
        raise ValueError('Actual local model must match the pinned official GGUF')
    runtime = shutil.which('llama-server')
    if not runtime:
        raise FileNotFoundError('The official llama.cpp runtime is absent')
    process_text = subprocess.run(['ps', '-axo', 'pid,args'], check=True, text=True,
                                  capture_output=True).stdout
    runtime_lines = [line for line in process_text.splitlines() if line.strip().split(' ', 1)[-1].startswith(runtime + ' ')
                     and '--port 8782' in line]
    if len(runtime_lines) != 1 or any(token not in runtime_lines[0]
                                    for token in ('--device none', '-ngl 0', '--no-op-offload', str(args.model.resolve()))):
        raise ValueError('Cannot establish the actual pinned model CPU-only server process')
    code_paths = [Path(__file__), ROOT / 'src/akash_flood/copilot.py',
                  ROOT / 'src/akash_flood/fresh_copilot.py', ROOT / 'scripts/build_fresh_report.py']
    records = [*store.source_bindings, record(args.challenge), model_record, record(Path(runtime)),
               *[record(p) for p in code_paths], *[record(p) for p in args.prior_development_result]]
    active_bindings = {item['path']: item for item in store.source_bindings}
    for item in challenge.get('source_verification', []):
        path = (ROOT / item['path']).resolve()
        expected = {'path': str(path), 'bytes': item['bytes'], 'sha256': item['sha256']}
        if record(path) != expected or active_bindings.get(str(path)) != expected:
            raise ValueError('Development challenge source identity differs from the active case')
    verify_records(records)
    args.out.mkdir(parents=True)
    freeze = {'schema': 'akash-fresh-copilot-development-freeze-v1',
              'frozen_at': datetime.now(timezone.utc).isoformat(), 'role': challenge['role'],
              'challenge': record(args.challenge), 'case': store.case,
              'adapter': 'FocusedCaseFactStore', 'known_development_outcomes': challenge.get('known_before_authoring', []),
              'known_prior_development_results': [record(p) for p in args.prior_development_result],
              'challenge_questions_known_before_this_run': True,
              'model': MODEL, 'model_record': model_record, 'runtime_record': record(Path(runtime)),
              'actual_runtime_process': runtime_lines[0].strip(), 'model_endpoint': 'http://127.0.0.1:8782',
              'source_and_code_bindings': records, 'cases': cases,
              'model_training_run': False, 'hazard_inference_run': False,
              'reference_products_loaded': False, 'human_nepali_review_completed': False}
    (args.out / 'freeze.json').write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + '\n')
    rows = []
    for case in cases:
        started = time.perf_counter()
        row = {'case': case}
        try:
            verify_records(records)
            answer = ask_local_model(store, case['question'], language=case['language'])
            verify_records(records)
            claims = answer['claims']
            ids = {claim['fact_id'] for claim in claims}
            source_match = all(claim['text'] == getattr(store.facts[claim['fact_id']], case['language'])
                               and claim['source_sha256'] == store.facts[claim['fact_id']].source_sha256
                               and claim['source_pointer'] == store.facts[claim['fact_id']].source_pointer
                               and claim.get('source_references', []) == list(store.facts[claim['fact_id']].source_references)
                               for claim in claims)
            row.update(response=answer, displayed_source_clauses_match=source_match,
                       expected_display_status=answer['status'] == case['status'],
                       required_facts_included=set(case['required']) <= ids,
                       raw_status_expected=answer['model_plan']['status'] == case['status'],
                       guard_corrected_model=answer['policy_guard']['corrected_model_plan'])
            row['only_declared_relevant_ids'] = ids <= set(case.get('allowed_fact_ids', ids))
            row['passes'] = row['expected_display_status'] and row['required_facts_included'] and source_match
        except Exception as error:
            row.update(passes=False, error=type(error).__name__ + ': ' + str(error))
        row['wall_seconds'] = time.perf_counter() - started
        rows.append(row)
        (args.out / (case['id'] + '.json')).write_text(json.dumps(row, indent=2, ensure_ascii=False) + '\n')
        print(case['id'], 'pass' if row['passes'] else 'fail', round(row['wall_seconds'], 2), flush=True)
    verify_records(records)
    summary = {'schema': 'akash-fresh-copilot-development-result-v1',
               'role': challenge['role'], 'finished_at': datetime.now(timezone.utc).isoformat(),
               'freeze': record(args.out / 'freeze.json'), 'cases': len(rows),
               'passes': sum(row['passes'] for row in rows),
               'displayed_source_clause_checks_pass': sum(row.get('displayed_source_clauses_match', False) for row in rows),
               'raw_status_expected': sum(row.get('raw_status_expected', False) for row in rows),
               'guard_corrections': sum(row.get('guard_corrected_model', False) for row in rows),
               'errors': sum('error' in row for row in rows), 'wall_seconds': sum(row['wall_seconds'] for row in rows),
               'only_declared_relevant_ids': sum(row.get('only_declared_relevant_ids', False) for row in rows),
               'adapter': 'FocusedCaseFactStore',
               'human_nepali_review_completed': False, 'hazard_accuracy_validated': False}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
