"""Independent enumeration of uncertain states, rather than mirrored BFS tests."""
from copy import deepcopy
from itertools import product
import random

import networkx as nx
import pytest

from akash_flood.access_bounds import analyze_access_bounds


def road_graph(n):
    graph = nx.MultiDiGraph()
    for node in range(n):
        graph.add_node(node, x=node*.001, y=0)
    return graph


def segment(graph, way, u, v, both=True):
    result = []
    for a, b, direction in [(u, v, 'f')] + ([(v, u, 'r')] if both else []):
        edge_id = f'{way}:0:{direction}'
        graph.add_edge(a, b, key=edge_id, edge_id=edge_id, way_id=way, length_m=10.0)
        result.append(edge_id)
    return result


def point(node, identifier=None):
    return {'id': identifier or f'point-{node}', 'lon': node*.001, 'lat': 0}


def retained_graph(graph, removed):
    copy = graph.copy()
    copy.remove_edges_from([(u,v,key) for u,v,key,data in copy.edges(keys=True,data=True) if data['edge_id'] in removed])
    return copy


def verify_path(graph, witness, removed):
    nodes, edges = witness['path_node_ids'], witness['path_edge_ids']
    assert len(nodes) == len(edges) + 1
    assert nodes[0] == witness['source_node_id']
    assert nodes[-1] == witness['destination_node_id']
    distance = 0
    for u,v,edge in zip(nodes,nodes[1:],edges):
        assert edge not in removed
        data = graph[u][v][edge]
        assert data['edge_id'] == edge
        distance += data['length_m']
    assert witness['distance_m'] == pytest.approx(distance)


def verify_cut(graph, certificate, removed):
    reachable = set(certificate['reachable_node_ids'])
    targets = set(certificate['mapped_destination_node_ids'])
    assert certificate['source_node_id'] in reachable
    assert not reachable & targets
    boundary = {data['edge_id'] for u,v,key,data in graph.edges(keys=True,data=True) if u in reachable and v not in reachable}
    assert boundary <= removed
    assert boundary == set(certificate['removed_edge_ids'])
    assert boundary == {edge['edge_id'] for edge in certificate['outgoing_cut_boundary']}


def test_all_assignments_and_inspection_branches_match_independent_enumeration():
    # One-way paths, parallel ways, fixed closures, alternatives and disconnected
    # components vary across seeded small graphs. Exactly 16 states per graph.
    for seed in range(20):
        rng = random.Random(seed)
        graph = road_graph(6)
        physical = []
        for way in range(9):
            u,v = rng.sample(range(6),2)
            physical.append(segment(graph, way, u, v, both=rng.choice([True,False])))
        uncertain = {f'link-{i}': physical[i] for i in range(4)}
        certain = set(physical[4])
        settlements = [point(node) for node in range(5)]
        result = analyze_access_bounds(graph,settlements,[point(5,'destination')],certain,uncertain,max_snap_m=1)
        states = list(product([False,True], repeat=4))
        outcomes = {}
        for state in states:
            removed = certain | {edge for i,closed in enumerate(state) if closed for edge in physical[i]}
            scenario = retained_graph(graph,removed)
            outcomes[state] = {node:nx.has_path(scenario,node,5) for node in range(5)}
        by_id = {row['id']:row for row in result['settlements']}
        for node in range(5):
            row = by_id[f'point-{node}']
            values = [outcomes[state][node] for state in states]
            expected = ('baseline_disconnected' if not nx.has_path(graph,node,5)
                        else 'robust_connected' if all(values)
                        else 'robust_newly_disconnected' if not any(values)
                        else 'ambiguous')
            assert row['status'] == expected
            if row['path_witness']:
                verify_path(graph,row['path_witness'],certain|set(sum(physical[:4],[])))
            if row['unreachable_certificate']:
                verify_cut(graph,row['unreachable_certificate'],certain)
            if row['optimistic_path_witness']:
                verify_path(graph,row['optimistic_path_witness'],certain)
                verify_cut(graph,row['pessimistic_cut_certificate'],certain|set(sum(physical[:4],[])))
        for inspection in result['inspection_outcomes']:
            i = int(inspection['link_id'].split('-')[1])
            for branch,closed in [('if_open',False),('if_closed',True)]:
                resolved = []
                for node in range(5):
                    if by_id[f'point-{node}']['status'] != 'ambiguous':
                        continue
                    values = [outcomes[state][node] for state in states if state[i] == closed]
                    if all(values) or not any(values):
                        resolved.append(f'point-{node}')
                assert inspection[branch]['resolved_settlement_ids'] == resolved
                assert inspection[branch]['resolved_count'] == len(resolved)


def test_parallel_alternatives_need_two_inspections_and_oneway_is_respected():
    graph = road_graph(3)
    a = segment(graph,10,0,1)
    b = segment(graph,20,0,1)
    segment(graph,30,1,2,both=False)
    result = analyze_access_bounds(graph,[point(0)],[point(2)],uncertain_links={'a':a,'b':b},max_snap_m=1)
    assert result['settlements'][0]['status'] == 'ambiguous'
    for inspection in result['inspection_outcomes']:
        assert inspection['if_open']['resolved_count'] == 1
        assert inspection['if_closed']['resolved_count'] == 0
    reverse = analyze_access_bounds(graph,[point(2)],[point(0)],uncertain_links={'a':a,'b':b},max_snap_m=1)
    assert reverse['settlements'][0]['status'] == 'baseline_disconnected'


def test_fixed_closure_certificate_and_original_graph_not_mutated():
    graph = road_graph(3)
    a = segment(graph,10,0,1)
    segment(graph,20,1,2)
    before = deepcopy(nx.node_link_data(graph,edges='edges'))
    result = analyze_access_bounds(graph,[point(0)],[point(2)],certain_closed_edge_ids=a,max_snap_m=1)
    row = result['settlements'][0]
    assert row['status'] == 'robust_newly_disconnected'
    verify_cut(graph,row['unreachable_certificate'],set(a))
    assert nx.node_link_data(graph,edges='edges') == before


def test_unmapped_is_not_reported_as_flood_disconnection_and_empty_path_is_valid():
    graph = road_graph(2)
    segment(graph,10,0,1)
    result = analyze_access_bounds(graph,[point(50),point(1)],[point(1)],max_snap_m=1)
    assert result['settlements'][0]['status'] == 'unmapped'
    assert result['settlements'][0]['possible_newly_disconnected'] is False
    assert result['settlements'][1]['status'] == 'robust_connected'
    assert result['settlements'][1]['path_witness']['path_edge_ids'] == []
    missing = analyze_access_bounds(graph,[point(0)],[point(50)],max_snap_m=1)
    assert missing['settlements'][0]['reason'] == 'no_mapped_destinations'


def test_uncertain_groups_are_complete_disjoint_physical_segments():
    graph = road_graph(3)
    a = segment(graph,10,0,1)
    b = segment(graph,20,1,2)
    for uncertain,certain in [({'a':a[:1]},()),({'a':a,'b':a},()),({'a':a+b},()),({'a':['missing']},()),({'a':a},a),({1:a},())]:
        with pytest.raises(ValueError):
            analyze_access_bounds(graph,[],[],certain,uncertain)


def test_deterministic_output_and_independent_open_closed_rankings():
    graph = road_graph(3)
    a = segment(graph,10,0,1)
    b = segment(graph,20,1,2)
    arguments = (graph,[point(0),point(1)],[point(2)],(),{'a':a,'b':b})
    result = analyze_access_bounds(*arguments,max_snap_m=1)
    assert result == analyze_access_bounds(*arguments,max_snap_m=1)
    assert result['inspection_rankings']['if_closed'] == ['b','a']
    assert result['inspection_rankings']['if_open'] == ['b','a']


def test_invalid_edge_lengths_and_identifiers_are_rejected():
    graph = road_graph(2)
    segment(graph,10,0,1)
    graph[0][1]['10:0:f']['length_m'] = float('nan')
    with pytest.raises(ValueError):
        analyze_access_bounds(graph,[],[])
