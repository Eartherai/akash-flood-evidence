from itertools import product
import random

import networkx as nx
import pytest

from akash_flood.inspection import InspectionExperiment


def setup():
    graph=nx.MultiDiGraph()
    for node in range(4):graph.add_node(node,x=node*.001,y=0)
    groups={}
    for way,(u,v) in enumerate([(0,1),(1,3),(0,2),(2,3)]):
        edges=[]
        for a,b,direction in [(u,v,'f'),(v,u,'r')]:
            edge=f'{way}:0:{direction}';edges.append(edge)
            graph.add_edge(a,b,key=edge,edge_id=edge,way_id=way,length_m=10.)
        groups[f'way-{way}']=edges
    points=[{'id':f's-{n}','lon':n*.001,'lat':0} for n in [0,1,2]]
    targets=[{'id':'hospital','lon':.003,'lat':0}]
    return graph,groups,InspectionExperiment(graph,points,targets,groups,max_snap_m=1)


def test_all_partial_states_equal_independent_directed_reachability():
    graph,groups,experiment=setup()
    for states in product(['unknown','open','closed'],repeat=4):
        closed=sum(1<<i for i,value in enumerate(states) if value=='closed')
        opened=sum(1<<i for i,value in enumerate(states) if value=='open')
        result=experiment.state(closed,opened)
        for worst,key in [(False,'disconnected_bits'),(True,'connected_bits')]:
            removed={edge for i,value in enumerate(states) if value=='closed' or (worst and value=='unknown') for edge in groups[experiment.keys[i]]}
            actual=graph.copy()
            actual.remove_edges_from([(u,v,k) for u,v,k,data in actual.edges(keys=True,data=True) if data['edge_id'] in removed])
            for i,source in enumerate([0,1,2]):
                reachable=nx.has_path(actual,source,3)
                assert bool(result[key] & (1<<i))==(reachable if worst else not reachable)


def test_policy_actions_do_not_depend_on_unobserved_hidden_states():
    _,_,experiment=setup()
    for policy in ['maximin','adaptive_route_count','adaptive_betweenness','random_relevant']:
        first=[experiment.run(hidden,policy,5,random_seed=2026)[0]['chosen_group'] for hidden in range(16)]
        assert len(set(first))==1
        for hidden in range(16):
            trace=experiment.run(hidden,policy,5,random_seed=2026)
            assert trace[-1]['unresolved_count']==0
            assert all(b['unresolved_count']<=a['unresolved_count'] for a,b in zip(trace,trace[1:]))
            chosen=[step['chosen_group'] for step in trace if 'chosen_group' in step]
            assert len(chosen)==len(set(chosen))


def test_joint_way_group_is_a_single_inspection_and_irrelevant_links_are_filtered():
    graph,groups,_=setup()
    graph.add_node(4,x=.004,y=0);graph.add_node(5,x=.005,y=0)
    edge='irrelevant:0:f';graph.add_edge(4,5,key=edge,edge_id=edge,way_id=99,length_m=10.)
    groups={'joint':groups['way-0']+groups['way-1'],'other':groups['way-2']+groups['way-3'],'irrelevant':[edge]}
    experiment=InspectionExperiment(graph,[{'id':'s','lon':0,'lat':0}],[{'id':'d','lon':.003,'lat':0}],groups,max_snap_m=1)
    state=experiment.state(0,0)
    assert 'irrelevant' not in [experiment.keys[i] for i in experiment.eligible(state)]
    assert experiment.branch_resolutions(state,experiment.keys.index('joint'))==(1,0)


def test_unknown_or_duplicate_groups_and_inconsistent_observations_rejected():
    graph,groups,experiment=setup()
    with pytest.raises(ValueError):InspectionExperiment(graph,[],[],{'x':['missing']})
    with pytest.raises(ValueError):InspectionExperiment(graph,[],[],{'x':groups['way-0'],'y':groups['way-0']})
    with pytest.raises(ValueError):experiment.state(1,1)
    with pytest.raises(ValueError):experiment.run(1<<7,'maximin')
    with pytest.raises(ValueError):experiment.choose(experiment.state(0,0),'fake')
