"""Prepare compact source-bound graph witnesses for the saved two-case viewer.

No new graph analysis, model call or published checking map enters this render.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'outputs/observation-access-2026-10-11'
RESEARCH = ROOT/'research/observation-access-2026-10-11'
OUT = ROOT/'outputs/observation-access-viewer-2026-10-11'

def record(p):
    p=Path(p);raw=p.read_bytes()
    return {'path':str(p.relative_to(ROOT)), 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}

def bound_json(p, binding):
    if record(p) != binding:
        raise ValueError('Saved evidence changed: '+str(p))
    return json.loads(Path(p).read_text())

def main():
    if OUT.exists():
        raise FileExistsError('Preserve previous viewer; choose a new version')
    protocol_path=RESEARCH/'protocol.frozen.json'
    protocol=json.loads(protocol_path.read_text())
    independent_path=RESEARCH/'independent-verification.json'
    independent=json.loads(independent_path.read_text())
    if independent.get('status')!='passed':
        raise ValueError('Independent graph check must pass')
    sources=[record(protocol_path),record(independent_path),record(Path(__file__)),record(ROOT/'web/observation-access-audit.html')]
    data={'schema':'akash-observation-access-viewer-v1','cases':{},
          'scope':'Conditional graph connectivity, not observed physical closures, field access or a safe-route service.'}
    for slug in ('rasuwa','kathmandu'):
        folder=SOURCE/(slug+'-v1')
        receipt_path=folder/'receipt.json';receipt=json.loads(receipt_path.read_text());sources.append(record(receipt_path))
        if receipt['status']!='completed' or not receipt['all_frozen_inputs_unchanged']:
            raise ValueError('Analysis incomplete or source changed')
        bindings={Path(r['path']).name:r for r in receipt['outputs']}
        previous=bound_json(folder/'original-witness-support.json',bindings['original-witness-support.json'])
        scenario=bound_json(folder/'expanded-scenario.json',bindings['expanded-scenario.json'])
        sources.extend(bindings.values())
        old_by_id={row['id']:row for row in previous['witnesses']}
        segments={}
        def route(support):
            if support is None:return None
            ids=[]
            for segment in support['segments']:
                edge=segment['edge_id'];item={k:segment[k] for k in (
                    'coordinates','centerline_length_m','assessed_length_m','unknown_within_aoi_length_m','outside_aoi_length_m')}
                item['observation_state']='outside' if item['outside_aoi_length_m']>0 else 'unknown' if item['unknown_within_aoi_length_m']>0 else 'assessed'
                if edge in segments and segments[edge]!=item:raise ValueError('Conflicting edge support')
                segments[edge]=item;ids.append(edge)
            return {'edge_ids':ids,'graph_distance_m':support['graph_distance_m'],
                    'projected_centerline_total_m':support['projected_centerline_total_m'],
                    'lengths_m':support['lengths_m'],'fractions':support['fractions_of_projected_centerline']}
        rows=[]
        for row in scenario['settlements']:
            old=old_by_id.get(row['id']);cut=row.get('cut_certificate')
            rows.append({k:row[k] for k in ('id','name','previous_status','status','point_coordinates','inside_water_aoi','inside_extraction_context')} |
                {'destination':row['nearest_baseline_destination'], 'original_route':route(old['support']) if old else None,
                 'expanded_route':route(row.get('path_support')),
                 'expanded_witness_scenario':row['path_witness']['scenario'] if row.get('path_witness') else None,
                 'cut_summary':{'reachable_nodes':len(cut['reachable_node_ids']),'removed_boundary_edges':len(cut['removed_boundary_edge_ids'])} if cut else None})
        if len(rows)!=sum(receipt['expanded_status_counts'].values()):raise ValueError('Settlement count differs')
        data['cases'][slug]={'label':slug.title(),'case':protocol['cases'][slug]['case'],
            'original_counts':receipt['original_status_counts'],'expanded_counts':receipt['expanded_status_counts'],
            'scope_counts':receipt['scope_status_counts'],'uncertain_groups':receipt['uncertain_physical_group_count'],
            'new_uncertain_groups':receipt['new_uncertain_physical_group_count'], 'segments':segments,'settlements':rows,
            'resources':{k:receipt[k] for k in ('wall_seconds','cpu_seconds','peak_process_rss_bytes')}}
    raw=(json.dumps(data,separators=(',',':'),ensure_ascii=False)+'\n').encode()
    if len(raw)>12_000_000:raise ValueError('Compact browser payload exceeds 12 MB bound')
    OUT.mkdir();(OUT/'data.json').write_bytes(raw)
    shutil.copy2(ROOT/'web/observation-access-audit.html',OUT/'index.html')
    (OUT/'receipts').mkdir()
    for slug in ('rasuwa','kathmandu'):
        shutil.copy2(SOURCE/(slug+'-v1')/'receipt.json',OUT/'receipts'/(slug+'.json'))
    for name in ('protocol.frozen.json','independent-verification.json','summary.json'):
        shutil.copy2(RESEARCH/name,OUT/'receipts'/name)
    proof={'schema':'akash-observation-viewer-render-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'source_hashes_and_recounts_passed_browser_review_pending','sources':sources,
        'outputs':{str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.rglob('*') if p.is_file()},
        'graph_analysis_performed':False,'model_inference_performed':False,'reference_maps_loaded':False,'visual_review':'pending'}
    (OUT/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'viewer':str(OUT),'data_bytes':len(raw),'source_bindings':len(sources)},indent=2))

if __name__=='__main__':main()
