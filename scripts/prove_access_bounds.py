"""Bounded CPU proof on real historical roads with hypothetical bridge states.

No flood map, satellite-based closure, damage or real passability is inferred.
All bridge-tagged physical segments are declared uncertain before outcomes are
computed. This is an engineering stress test, not evaluation of the disaster.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

from akash_flood.access import build_road_graph
from akash_flood.access_bounds import analyze_access_bounds
from akash_flood.osm import named_points


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/access-bounds-proof'


def verify_certificates(graph, result):
    """Check submitted paths and cuts directly against source edge identities."""
    index = {data['edge_id']:(u,v,data) for u,v,key,data in graph.edges(keys=True,data=True)}
    certain = set(result['certain_closed_edge_ids'])
    worst = certain | {edge for group in result['uncertain_links'].values() for edge in group}
    counts = {'paths':0, 'cuts':0}
    for row in result['settlements']:
        for key in ['path_witness','optimistic_path_witness']:
            witness = row[key]
            if witness is None:
                continue
            removed = worst if witness['scenario']=='pessimistic' else certain
            nodes,edges = witness['path_node_ids'],witness['path_edge_ids']
            assert len(nodes)==len(edges)+1 and nodes[0]==row['snap_node_id']
            assert nodes[-1]==witness['destination_node_id']
            distance = 0.0
            for u,v,edge in zip(nodes,nodes[1:],edges):
                actual_u,actual_v,data = index[edge]
                assert (u,v)==(actual_u,actual_v) and edge not in removed
                distance += data['length_m']
            assert abs(distance-witness['distance_m'])<1e-8
            counts['paths']+=1
        for key in ['unreachable_certificate','pessimistic_cut_certificate']:
            certificate = row[key]
            if certificate is None:
                continue
            removed = worst if certificate['scenario']=='pessimistic' else certain
            reachable = set(certificate['reachable_node_ids'])
            assert row['snap_node_id'] in reachable
            assert not reachable & set(certificate['mapped_destination_node_ids'])
            boundary = {edge for edge,(u,v,data) in index.items() if u in reachable and v not in reachable}
            assert boundary<=removed and boundary==set(certificate['removed_edge_ids'])
            counts['cuts']+=1
    return counts


def main():
    input_path = ROOT / 'data/rasuwa-pilot/osm.json'
    provenance_path = ROOT / 'data/rasuwa-pilot/osm.provenance.json'
    source = json.loads(input_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    graph = build_road_graph(source)
    settlements,destinations = named_points(source)
    groups = defaultdict(list)
    for u,v,key,data in graph.edges(keys=True,data=True):
        if data['bridge']:
            link = f"way-{data['way_id']}-nodes-{min(u,v)}-{max(u,v)}"
            groups[link].append(data['edge_id'])
    groups = {key:sorted(value) for key,value in sorted(groups.items())}
    # All bridge tags, never selected using favorable settlement outcomes.
    start = time.perf_counter()
    result = analyze_access_bounds(graph,settlements,destinations,uncertain_links=groups,max_snap_m=500)
    analysis_elapsed = time.perf_counter()-start
    start = time.perf_counter()
    certificates = verify_certificates(graph,result)
    verification_elapsed = time.perf_counter()-start
    # Save complete proofs for the highest closed-branch resolving segment.
    # Selection is explicitly by the computed ranking, not a preselected case.
    best = sorted(result['inspection_outcomes'],key=lambda item:(-item['if_closed']['resolved_count'],item['link_id']))[0] if groups else None
    focused = None
    if best:
        start = time.perf_counter()
        link = best['link_id']
        rest = {key:value for key,value in groups.items() if key!=link}
        fixed_open = analyze_access_bounds(graph,settlements,destinations,uncertain_links=rest,max_snap_m=500,rank_inspections=False)
        fixed_closed = analyze_access_bounds(graph,settlements,destinations,certain_closed_edge_ids=groups[link],uncertain_links=rest,max_snap_m=500,rank_inspections=False)
        focused = {'selected_link_id':link,'selection_rule':'largest if_closed resolved count; ties by link ID',
                   'if_open':fixed_open,'if_closed':fixed_closed,
                   'certificate_checks':{'if_open':verify_certificates(graph,fixed_open),'if_closed':verify_certificates(graph,fixed_closed)},
                   'analysis_and_verification_seconds':round(time.perf_counter()-start,6)}
        ambiguous = {row['id'] for row in result['settlements'] if row['status']=='ambiguous'}
        for branch in ['if_open','if_closed']:
            resolved = {row['id'] for row in focused[branch]['settlements'] if row['id'] in ambiguous and row['status']!='ambiguous'}
            assert resolved==set(best[branch]['resolved_settlement_ids'])
    result['provenance'] = {
        'created_at':datetime.now(timezone.utc).isoformat(),
        'osm_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
        'osm_provenance_sha256':hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        'requested_historical_snapshot':provenance['snapshot'],
        'scenario_rule':'Every motor-road bridge-tagged physical segment is uncertain; no fixed closures.',
        'scenario_is_synthetic':True,'hazard_inputs_used':False,
        'n_graph_nodes':len(graph),'n_directed_edges':graph.number_of_edges(),
        'n_settlements':len(settlements),'n_destinations':len(destinations),
        'n_uncertain_physical_segments':len(groups),
        'uncertain_assignments_covered_by_monotone_bounds':2**len(groups),
        'analysis_seconds':round(analysis_elapsed,6),'certificate_verification_seconds':round(verification_elapsed,6),
        'certificate_checks':certificates,'python':platform.python_version(),
        'machine':platform.machine(),'platform':platform.platform(),
        'limitations':[
            'Not observed flood damage or physical passability. No satellite closure evidence is used.',
            'Reachability is to any of the mapped node-tagged destinations, not a verified operative or nearest hospital.',
            'Independent segment states are a stress-test uncertainty set; actual bridge segments and hazards can be correlated.',
            'Nearest-node snapping, missing OSM facilities, network boundary, restrictions and unmapped settlements limit conclusions.',
            'Inspection outcomes are separate open/closed branch counts; no expected gain or branch likelihood is claimed.',
        ],
    }
    OUT.mkdir(parents=True,exist_ok=True)
    output = OUT/'proof.json'
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    if focused:
        (OUT/'inspection-proof.json').write_text(json.dumps(focused,indent=2,ensure_ascii=False)+'\n')
    strongest_open = sorted(result['inspection_outcomes'],key=lambda i:(-i['if_open']['resolved_count'],i['link_id']))[:5]
    strongest_closed = sorted(result['inspection_outcomes'],key=lambda i:(-i['if_closed']['resolved_count'],i['link_id']))[:5]
    summary = {**result['provenance'],'status_counts':result['status_counts'],
               'top_open_branch_inspections':strongest_open,'top_closed_branch_inspections':strongest_closed,
               'proof_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
               'focused_inspection_proof':{key:focused[key] for key in ['selected_link_id','selection_rule','certificate_checks','analysis_and_verification_seconds']} if focused else None}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({key:summary[key] for key in ['n_graph_nodes','n_directed_edges','n_uncertain_physical_segments','status_counts','analysis_seconds','certificate_verification_seconds','certificate_checks','focused_inspection_proof','proof_sha256']},indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
