"""Combine current source-bound layer facts without model inference or resampling."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil
from run_fresh_case import file_record, read_json, verify_record, write_json
from akash_flood.fresh_case import normalized_utc, event_boundary


def checked_outputs(binding):
    return {name:verify_record(record) for name,record in binding['outputs'].items()}


def run(report_binding, execution_plan, out, observation_receipt=None):
    out=Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError('Preserve prior case reports; use a fresh folder')
    plan=read_json(execution_plan)
    source=read_json(report_binding)
    if source.get('schema')!='akash-fresh-report-binding-v1' or source.get('status')!='completed':
        raise ValueError('Require completed native-layer report binding')
    files=checked_outputs(source)
    for record in source.get('inputs',[]):
        verify_record(record)
    report=read_json(files['report.json'])
    if (report['case']!=source['case'] or report['case']['bbox']!=plan['case']['bbox']
            or report['case']['event_date']!=plan['case']['event_date']
            or report['case'].get('context_bbox')!=plan['case'].get('context_bbox')
            or normalized_utc(event_boundary(report['case']['event_date'],report['historical_snapshot'])[2])
               !=normalized_utc(event_boundary(plan['case']['event_date'],plan['case']['osm_snapshot'])[2])):
        raise ValueError('Case mismatch between completed reports and execution plan')
    inputs=[file_record(report_binding),file_record(execution_plan),file_record(files['report.json'])]
    expanded=None
    if observation_receipt:
        receipt=read_json(observation_receipt)
        if receipt.get('status')!='completed' or receipt['case']!=report['case']:
            raise ValueError('Observation-support result is incomplete or belongs to another case')
        rows={Path(r['path']).name:verify_record(r) for r in receipt['outputs']}
        expanded=read_json(rows['expanded-scenario.json'])
        if expanded['case']!=report['case'] or expanded.get('confirmed_closures')!=0 or expanded.get('certain_closed_edge_ids')!=[]:
            raise ValueError('Expanded graph scenario changed its declared scientific scope')
        inputs.extend([file_record(observation_receipt),*map(file_record,rows.values())])
    points=(expanded['settlements'] if expanded else report['settlements'])
    counts=(expanded['scope_status_counts'] if expanded else {'candidate_only':report['access_scenario']['status_counts']})
    facts={'schema':'akash-fresh-combined-case-facts-v1','case':report['case'],
        'radar':{'counts':report['counts'],'source_dates':report['source_dates'],
                 'scope':'Uncalibrated binary-water transfer; new-water pixels are provisional candidates'},
        'optical':report.get('optical'),'historical_snapshot':report['historical_snapshot'],
        'mapped_features':report['mapped_features'],'conditional_scenario_counts':counts,
        'observation_uncertainty_applied':expanded is not None,'settlements':points,
        'confirmed_closures':0,'local_hazard_accuracy_established':False,
        'physical_passability_established':False,'full_track_b_requirements_satisfied':False,
        'reference_labels_loaded':False,'model_invoked_by_reporter':False,
        'native_grids_resampled_or_fused':False,'inputs':inputs}
    c=facts['radar']['counts'];optical=facts['optical']
    status_labels={'robust_connected':'Conditional graph connection','ambiguous':'Undetermined',
        'unmapped':'No mapped association','baseline_disconnected':'Baseline disconnected'}
    table=''.join('<tr><td>'+escape(str(row['name']))+'<small>'+escape(str(row['id']))+'</small></td><td>'+escape(status_labels.get(row['status'],row['status']))+'</td><td>'+escape(str((row.get('nearest_baseline_destination') or {}).get('destination_id','No eligible target')) )+'</td></tr>' for row in points)
    count_text=''.join('<li><b>'+escape(scope.replace('_',' '))+'</b>: '+escape(', '.join(f'{status_labels.get(k,k)} {v}' for k,v in values.items()) or 'No supplied settlement points')+'</li>' for scope,values in counts.items())
    optical_html='<p>No optical layer was requested or completed in this report.</p>'
    if optical:
        oc=optical['counts']
        optical_html=f'''<p><b>{oc['joint_assessed_pixels']:,} / {oc['aoi_pixels']:,}</b> native 20 m cells jointly assessed ({optical['joint_assessed_fraction']*100:.2f}%). <b>{oc['unknown_pixels']:,}</b> unknown; <b>{oc['candidate_pixels']:,}</b> surface-change candidates.</p><p>Before {escape(optical['source_dates']['before'])}; after {escape(optical['source_dates']['after'])}.</p><img src="optical-overview.png" alt="Native optical layer: gray unknown, green assessed without candidate, brown provisional surface-change candidate"><p>Fixed spectral hypotheses. Local debris accuracy and physical road effects are unvalidated; agriculture, riverbank change and vegetation removal can trigger candidates.</p>'''
    year=report['case']['event_date'][:4]
    layer_heading='Two observation layers.' if optical else 'Radar observation evidence.'
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Akash | Completed case evidence</title><style>
    :root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#f4f7f9;color:#15354a;font:16px/1.55 "Avenir Next","Segoe UI",sans-serif}}main{{max-width:1240px;margin:auto;padding:32px}}h1{{font-size:clamp(30px,4vw,48px);line-height:1.12;letter-spacing:-1px}}h2{{font-size:23px}}.muted,small{{color:#526575}}small{{display:block;overflow-wrap:anywhere}}.notice{{padding:18px;border-left:4px solid #875600;background:#f6efdf}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}}section{{background:#fff;border:1px solid #b6c6d1;padding:22px;margin:22px 0;border-radius:7px;min-width:0}}img{{width:100%;height:310px;object-fit:contain;background:#edf2f5}}a{{color:#174c97;overflow-wrap:anywhere}}.table{{overflow:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #cdd8df;vertical-align:top}}details{{padding:16px 0}}summary{{cursor:pointer;min-height:44px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}@media(max-width:720px){{main{{padding:18px}}.grid{{grid-template-columns:1fr}}section{{padding:16px}}}} 
    </style></head><body><main><p class="muted">Akash / completed bounded case</p><h1>{layer_heading}<br>One inspectable case.</h1><p>Event {escape(report['case']['event_date'])}; AOI {escape(str(report['case']['bbox']))}. Historical OSM cutoff {escape(report['historical_snapshot'])}.</p><p class="notice">Engineering execution is complete for the recorded stages. These outputs do not establish damaged roads, safe passage, hospital operation or local flood/debris accuracy. Baseline disconnection is an earlier graph condition, not new flood isolation.</p>
    <div class="grid"><section><h2>Radar water-state candidates</h2><p><b>{c['assessed_pixels']:,} / {c['aoi_pixels']:,}</b> cells assessed; <b>{c['unknown_pixels']:,}</b> unknown. <b>{c['new_water_candidate_pixels']:,}</b> new-water candidate cells.</p><p>Before {escape(report['source_dates']['before'])}; after {escape(report['source_dates']['after'])}.</p><img src="map.png" alt="Native radar-classification map; candidate water and unknown observations, not observed road closures"><p>Frozen CPU checkpoint and threshold unchanged. Permanent water and flooding are not separated by the trained binary-water label.</p></section><section><h2>Optical observation support</h2>{optical_html}<p>The layers retain their separate native grids; no radar-optical pixel fusion or resampling is claimed.</p></section></div>
    <section><h2>Conditional settlement evidence</h2><p>{'Unknown-within-AOI and outside-AOI physical road groups are uncertain in this expanded scenario.' if expanded else 'Only the earlier candidate-intersection scenario is available.'} Destinations stay fixed. A connection is a graph result, not proof of passability.</p><ul>{count_text}</ul><div class="table"><table><thead><tr><th>Historical source point</th><th>Graph status</th><th>Fixed mapped target ID</th></tr></thead><tbody>{table or '<tr><td colspan="3">No supplied settlement points in this source inventory.</td></tr>'}</tbody></table></div><p>Empty graph witnesses retain null observation fractions. Whole-source, inside-AOI and extraction-context denominators are separate. Point counts are not people or confirmed damage.</p></section>
    <section><h2>Source facts and limitations</h2><p><a href="case-facts.json">Exact computed case facts</a> · <a href="situation.md">Situation evidence note</a> · <a href="manifest.json">Current report hashes</a></p><details><summary>Read the source bindings</summary><pre>{escape(json.dumps(inputs,indent=2))}</pre></details><p>Contains modified Copernicus Sentinel data {escape(year)}. © OpenStreetMap contributors; applicable upstream data terms remain separate from original-code MIT. No reference labels, model fitting or outcome-guided threshold changes occurred in this report.</p></section></main></body></html>'''
    note=f'''# Situation evidence - {report['case']['event_date']}

AOI: {report['case']['bbox']}. Historical OSM: {report['historical_snapshot']}.
Radar: {c['assessed_pixels']} of {c['aoi_pixels']} cells assessed; {c['unknown_pixels']} unknown; {c['new_water_candidate_pixels']} new-water candidate cells.
Radar source dates: {report['source_dates']}.
Optical: {json.dumps(optical['counts']) if optical else 'not included'}.
Conditional graph counts by point scope: {json.dumps(counts,ensure_ascii=False)}.

The saved fixed-target graph scenario does not establish observed closures, physical passability, facility entrances or hospital operation. Baseline-disconnected points are not new flood isolation. Unknown observations remain unknown. Optical candidates are fixed hypotheses, not validated debris. This bounded case execution is not an unrestricted-area or mapping-accuracy guarantee.

Exact facts: case-facts.json. Source and output identities: manifest.json.
'''
    for record in inputs: verify_record(record)
    out.mkdir(parents=True)
    write_json(out/'case-facts.json',facts)
    (out/'index.html').write_text(html)
    (out/'situation.md').write_text(note)
    for name in ('map.png','optical-overview.png'):
        if name in files:
            shutil.copyfile(files[name],out/name)
            inputs.append(file_record(files[name]))
    for record in inputs: verify_record(record)
    manifest={'schema':'akash-fresh-combined-report-v1','status':'completed',
        'created_utc':datetime.now(timezone.utc).isoformat(),'case':report['case'],
        'inputs':inputs,'code':[file_record(Path(__file__))],
        'outputs':{p.name:file_record(p) for p in sorted(out.iterdir())},
        'reference_labels_loaded':False,'model_invoked_by_reporter':False,
        'full_track_b_requirements_satisfied':False,'local_hazard_accuracy_established':False,
        'native_grids_resampled_or_fused':False}
    write_json(out/'manifest.json',manifest)
    return {'status':'completed','report':file_record(out/'index.html'),'manifest':file_record(out/'manifest.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-binding',type=Path,required=True)
    p.add_argument('--execution-plan',type=Path,required=True)
    p.add_argument('--observation-receipt',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.report_binding,a.execution_plan,a.out,a.observation_receipt),indent=2))
