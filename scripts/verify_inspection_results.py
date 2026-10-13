"""Independent NetworkX reachability checks for every saved policy step.

Uses source graph edge IDs and saved observations rather than the benchmark's
compiled BFS to construct each optimistic/pessimistic graph.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import networkx as nx

from akash_flood.access import _snap,build_road_graph
from akash_flood.inspection import InspectionExperiment
from akash_flood.osm import named_points

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main(region):
    folder=ROOT/'outputs/inspection-benchmark'/region
    result=json.loads((folder/'results.json').read_text())
    trace_data=json.loads((folder/'traces.json').read_text())
    source=ROOT/'data/rasuwa-pilot/osm.json' if region=='rasuwa-development' else ROOT/'data/inspection-benchmark'/region/'osm.json'
    assert sha(source)==result['source_sha256']
    graph=build_road_graph(json.loads(source.read_text()))
    settlements,destinations=named_points(json.loads(source.read_text()))
    groups=defaultdict(list)
    for u,v,key,data in graph.edges(keys=True,data=True):
        if data['bridge']:groups[f"bridge-way-{data['way_id']}"].append(data['edge_id'])
    assert sorted(groups)==trace_data['group_order']
    experiment=InspectionExperiment(graph,settlements,destinations,groups,500)
    index={edge:1<<i for i,key in enumerate(trace_data['group_order']) for edge in groups[key]}
    snaps=[_snap(graph,point,500) for point in settlements]
    targets={snap['node_id'] for point in destinations if (snap:=_snap(graph,point,500))['mapped']}
    # A synthetic sink joins all destinations so the independent NetworkX
    # oracle traverses the network once per state, instead of once per facility.
    oracle_graph=graph.copy();sink=object();oracle_graph.add_node(sink)
    for target in targets:oracle_graph.add_edge(target,sink,edge_id='verification-sink')
    cache={};started=time.perf_counter();steps=0;statement_checks=0
    def oracle(removed):
        if removed not in cache:
            view=nx.subgraph_view(oracle_graph,filter_edge=lambda u,v,key:not (index.get(oracle_graph[u][v][key]['edge_id'],0)&removed))
            reachable=nx.ancestors(view,sink)
            cache[removed]=sum(1<<i for i,snap in enumerate(snaps) if snap['mapped'] and snap['node_id'] in reachable)
        return cache[removed]
    baseline=oracle(0)
    allmask=(1<<len(groups))-1
    for policy,traces in trace_data['traces'].items():
        for scenario,trace in enumerate(traces):
            hidden=trace_data['hidden_closed_masks'][scenario]
            truth=oracle(hidden)
            for step in trace:
                closed,opened=step['known_closed_mask'],step['known_open_mask']
                assert not closed&opened
                assert closed & ~hidden==0 and opened & hidden==0
                lower=oracle(allmask^opened)&baseline
                upper=oracle(closed)&baseline
                unknown=upper&~lower
                assert unknown.bit_count()==step['unresolved_count']
                assert not lower&~truth and not (baseline&~upper)&truth
                compiled=experiment.state(closed,opened)
                assert compiled['ambiguous_bits']==unknown
                assert compiled['connected_bits']==lower
                assert compiled['disconnected_bits']==baseline&~upper
                if 'chosen_group' in step:
                    selected=trace_data['group_order'].index(step['chosen_group'])
                    assert selected in experiment.eligible(compiled)
                    a=oracle(allmask^(opened|1<<selected))&baseline
                    b=oracle(closed|1<<selected)&baseline
                    exact=[(unknown&a).bit_count(),(unknown&~b).bit_count()]
                    assert exact==step['branch_resolutions_open_closed']
                    assert step['observed_closed']==bool(hidden & (1<<selected))
                statement_checks+=(lower | (baseline&~upper)).bit_count();steps+=1
        print(f'{region}: independently verified {policy}',flush=True)
    output={'status':'passed','region':region,'source_sha256':sha(source),
            'results_sha256':sha(folder/'results.json'),'traces_sha256':sha(folder/'traces.json'),
            'saved_steps_checked':steps,'definitive_settlement_statements_checked':statement_checks,
            'distinct_removed_masks_checked':len(cache),'seconds':round(time.perf_counter()-started,3),
            'oracle':'NetworkX ancestors on source-edge-filtered directed multigraph views with a synthetic all-destination sink',
            'scope':'algorithm and observation-contract verification; no actual road-state or flood validation'}
    (folder/'independent-verification.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--region',default='rasuwa-development')
    main(parser.parse_args().region)
