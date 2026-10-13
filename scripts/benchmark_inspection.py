"""Run a once-recorded, predeclared inspection simulation on historical roads.

Freeze protocol before evaluating either new graph region. Road states are
synthetic. Results describe this scenario generator, never observed closures.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import networkx as nx
import numpy as np
import requests

from akash_flood.access import build_road_graph
from akash_flood.inspection import InspectionExperiment
from akash_flood.osm import named_points

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'data/inspection-benchmark/protocol.json'
REGIONS={
    'rasuwa-development':{'bbox':[85.25,28.04,85.48,28.38],'source':'existing pilot; already inspected'},
    'nepal-east-holdout':{'bbox':[85.65,27.75,85.90,28.05],'source':'new prespecified box; not claimed flooded'},
    'nepal-west-holdout':{'bbox':[84.95,27.70,85.15,27.95],'source':'new prespecified box; not claimed flooded'},
}
POLICIES=['maximin','adaptive_route_count','adaptive_betweenness','random_relevant']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def freeze():
    if PROTOCOL.exists():raise SystemExit('Protocol already exists; refusing replacement')
    write(PROTOCOL,{'created_at':datetime.now(timezone.utc).isoformat(),'schema_version':1,
        'regions':REGIONS,'snapshot':'2026-07-27T00:00:00Z','policies':POLICIES,
        'n_scenarios_per_region':100,'scenario_generator':'independent Bernoulli(0.5) closed/open per complete bridge-tagged OSM way; all way segments and directions jointly share the state',
        'scenario_seed':20261006,'random_policy_seed_base':902026,'bootstrap_seed':192026,'bootstrap_samples':2000,
        'max_snap_m':500,'inspection_budget':5,'reported_budgets':[1,3,5],
        'primary_metric':'remaining unresolved / initial ambiguous at budget three; initially ambiguous set is fixed for every policy',
        'eligibility':'identical adaptive filter for every policy: unknown bridge way has an optimistic directed walk from a currently ambiguous source to a mapped destination',
        'candidate':'maximize minimum of exact open/closed resolution counts; tie by maximum branch count descending, then group ID ascending',
        'route_count_baseline':'count currently ambiguous sources whose deterministic optimistic shortest-hop witness traverses a group, once per source; recompute each inspection',
        'betweenness_baseline':'NetworkX directed multigraph edge_betweenness_centrality_subset, sources=current ambiguous snap nodes, targets=all mapped destination nodes, unweighted, normalized=False; sum all directed segments per bridge way; recompute using known closures only',
        'random_baseline':'uniform choice among the same relevant unknown ways without replacement; one fixed random seed per shared scenario',
        'oracle':'exact synthetic queried-way state; perfect observation, no cost differences or sensor errors',
        'missing_baseline':'hazard-score ordering unavailable without SAR evidence; not approximated by invented scores',
        'correctness':'at every step definitive connected/disconnected bits must match fully revealed reachability; no unobserved state reaches choose()',
        'interval_scope':'paired fixed-seed simulation resampling on one fixed graph, excludes geography and graph/source uncertainty',
        'claim_scope':'scenario efficiency only; no flood accuracy, road passability, clinical or operational claim',
        'sources_sha256':{name:sha(ROOT/name) for name in ['src/akash_flood/inspection.py','scripts/benchmark_inspection.py','src/akash_flood/access.py','src/akash_flood/access_bounds.py','uv.lock']},
        'known_prior_results':'Rasuwa hypothetical segment outcomes already inspected. New boxes and grouped-way policy evaluations were not run before this freeze.'})
    print(json.dumps({'protocol':str(PROTOCOL),'sha256':sha(PROTOCOL),'frozen':True}))

def load_protocol():
    p=json.loads(PROTOCOL.read_text())
    for name,digest in p['sources_sha256'].items():
        if sha(ROOT/name)!=digest:raise SystemExit(f'Source changed after protocol freeze: {name}')
    return p

def fetch_region(region):
    p=load_protocol()
    if region=='rasuwa-development':raise SystemExit('Use the existing pilot source')
    folder=PROTOCOL.parent/region
    if (folder/'osm.json').exists():raise SystemExit('Region source already exists; refusing new snapshot')
    west,south,east,north=p['regions'][region]['bbox']
    extent=f'({south},{west},{north},{east})'
    query=(f'[out:json][timeout:90][date:"{p["snapshot"]}"];('
           f'way["highway"]{extent};nwr["place"~"^(city|town|village|hamlet)$"]{extent};'
           f'nwr["amenity"="hospital"]{extent};);(._;>;);out body;')
    endpoint='https://overpass-api.de/api/interpreter'
    response=requests.get(endpoint,params={'data':query},timeout=105,headers={'User-Agent':'AkashFloodResearchPrototype/0.1 (historical-road benchmark)'})
    response.raise_for_status();data=response.json()
    if data.get('remark') or not data.get('elements'):raise ValueError('Incomplete or empty historical source')
    write(folder/'osm.json',data)
    write(folder/'osm.provenance.json',{'endpoint':endpoint,'query':query,'snapshot':p['snapshot'],'bbox':p['regions'][region]['bbox'],
        'retrieved_at':datetime.now(timezone.utc).isoformat(),'raw_file_sha256':sha(folder/'osm.json'),
        'history_evidence':'server-executed historical body query, not per-element edit timestamps',
        'attribution':'© OpenStreetMap contributors; ODbL','element_count':len(data['elements'])})
    print(json.dumps({'region':region,'elements':len(data['elements']),'sha256':sha(folder/'osm.json')}))

def bootstrap_difference(values,seed,n):
    rng=np.random.default_rng(seed)
    values=np.asarray(values,dtype=float)
    means=values[rng.integers(0,len(values),size=(n,len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(means,[.025,.975])]

def run(region):
    p=load_protocol();out=ROOT/'outputs/inspection-benchmark'/region
    if (out/'run.started.json').exists():raise SystemExit('This region already has an evaluation record; refusing repeat')
    path=ROOT/'data/rasuwa-pilot/osm.json' if region=='rasuwa-development' else PROTOCOL.parent/region/'osm.json'
    data=json.loads(path.read_text());graph=build_road_graph(data);settlements,destinations=named_points(data)
    groups=defaultdict(list)
    for u,v,key,edge in graph.edges(keys=True,data=True):
        if edge['bridge']:groups[f"bridge-way-{edge['way_id']}"].append(edge['edge_id'])
    groups={key:sorted(value) for key,value in sorted(groups.items())}
    setup_start=time.perf_counter()
    reference=InspectionExperiment(graph,settlements,destinations,groups,p['max_snap_m'])
    initial=reference.initial_ambiguous_bits.bit_count()
    baseline_mapped=reference.baseline_bits.bit_count()
    baseline_disconnected=sum(snap['mapped'] and snap['node_id'] not in reference.baseline_nodes for snap in reference.snaps) if reference.targets else 0
    unmapped=len(settlements)-baseline_mapped-baseline_disconnected
    if not initial or not groups:
        write(out/'run.started.json',{'status':'uninformative','reason':'no initial ambiguity or bridge ways','n_settlements':len(settlements),'n_bridge_ways':len(groups),'initial_ambiguous':initial})
        print(json.dumps({'region':region,'status':'uninformative','initial_ambiguous':initial,'settlements':len(settlements),'bridge_ways':len(groups)}));return
    if region!='rasuwa-development':
        pilot=build_road_graph(json.loads((ROOT/'data/rasuwa-pilot/osm.json').read_text()))
        overlap=set(graph)&set(pilot)
        if overlap:raise ValueError(f'New box shares {len(overlap)} graph nodes with pilot; independence check failed')
    protocol_hash=sha(PROTOCOL)
    scenario_rng=random.Random(p['scenario_seed'])
    hidden=[sum(1<<i for i in range(len(groups)) if scenario_rng.random()<.5) for _ in range(p['n_scenarios_per_region'])]
    write(out/'run.started.json',{'status':'running','protocol_sha256':protocol_hash,'source_sha256':sha(path),'started_at':datetime.now(timezone.utc).isoformat()})
    policies={};traces={};timings={}
    for policy in POLICIES:
        experiment=InspectionExperiment(graph,settlements,destinations,groups,p['max_snap_m'])
        started=time.perf_counter();rows=[]
        for scenario,state in enumerate(hidden):
            trace=experiment.run(state,policy,p['inspection_budget'],p['random_policy_seed_base']+scenario)
            rows.append(trace)
            if (scenario+1)%25==0:print(f'{region}: {policy} {scenario+1}/100',flush=True)
        timings[policy]=round(time.perf_counter()-started,3)
        traces[policy]=rows
        policies[policy]={str(budget):{'mean_unresolved_count':float(np.mean([row[budget]['unresolved_count'] for row in rows])),
            'mean_unresolved_fraction':float(np.mean([row[budget]['unresolved_count']/initial for row in rows])),
            'mean_actual_inspections':float(np.mean([sum('chosen_group' in step for step in row[:budget]) for row in rows]))}
            for budget in p['reported_budgets']}
    comparisons={}
    for baseline in POLICIES[1:]:
        comparisons[baseline]={}
        for budget in p['reported_budgets']:
            differences=[b[budget]['unresolved_count']-a[budget]['unresolved_count'] for a,b in zip(traces['maximin'],traces[baseline])]
            base=policies[baseline][str(budget)]['mean_unresolved_count'];candidate=policies['maximin'][str(budget)]['mean_unresolved_count']
            comparisons[baseline][str(budget)]={'mean_fewer_unresolved':float(np.mean(differences)),
                'paired_95_interval_conditional':bootstrap_difference(differences,p['bootstrap_seed'],p['bootstrap_samples']),
                'relative_reduction':float((base-candidate)/base) if base else None,
                'scenario_wins':sum(value>0 for value in differences),'ties':sum(value==0 for value in differences),'losses':sum(value<0 for value in differences)}
    result={'region':region,'scope':'synthetic hidden bridge states on real historical OSM; scenario efficiency only',
        'protocol_sha256':protocol_hash,'source_sha256':sha(path),'grouping':'joint complete bridge-tagged OSM way',
        'graph_nodes':len(graph),'directed_edges':graph.number_of_edges(),'bridge_ways':len(groups),'settlements':len(settlements),'mapped_destinations':len(reference.targets),
        'initial_ambiguous':initial,'baseline_connected':baseline_mapped,'baseline_disconnected':baseline_disconnected,'unmapped':unmapped,
        'n_scenarios':len(hidden),'policies':policies,'paired_comparisons':comparisons,'policy_wall_seconds':timings,
        'setup_wall_seconds':round(time.perf_counter()-setup_start-sum(timings.values()),3),
        'all_definitive_conclusions_matched_fully_revealed_states':True,
        'hidden_states_used_for_selection':False,'hazard_score_baseline_available':False,'satellite_pixels_used':False,'gpu_used':False,
        'python':platform.python_version(),'networkx':nx.__version__,'machine':platform.machine(),
        'limitations':p['interval_scope']+'; perfect synthetic inspection, joint OSM-way grouping, any mapped destination, incomplete historic OSM and boundary/snap assumptions.'}
    write(out/'results.json',result)
    write(out/'traces.json',{'group_order':list(groups),'hidden_closed_masks':hidden,'traces':traces})
    write(out/'run.started.json',{'status':'complete','protocol_sha256':protocol_hash,'source_sha256':sha(path),'results_sha256':sha(out/'results.json'),'traces_sha256':sha(out/'traces.json')})
    print(json.dumps({'region':region,'initial_ambiguous':initial,'budget3':{key:value['3'] for key,value in policies.items()},'comparisons_budget3':{key:value['3'] for key,value in comparisons.items()},'seconds':timings},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','fetch','run'])
    parser.add_argument('--region',choices=list(REGIONS),default='rasuwa-development')
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='fetch':fetch_region(args.region)
    else:run(args.region)
