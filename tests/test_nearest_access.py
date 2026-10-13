import copy
import networkx as nx
import pytest

from akash_flood.nearest_access import analyze_nearest_access


def point(identity, node):
    return {'id': identity, 'lon': node/1000, 'lat': 0}


def network(edges):
    graph = nx.MultiDiGraph()
    for index, (u,v,length) in enumerate(edges):
        for node in (u,v):
            graph.add_node(node,x=node/1000,y=0)
        graph.add_edge(u,v,key=str(index),edge_id=str(index),way_id=index,length_m=length)
    return graph


def test_fixed_nearest_can_be_lost_while_another_destination_remains():
    graph = network([(0,1,10),(0,2,20)])
    result = analyze_nearest_access(graph,[point('s',0)],[point('near',1),point('far',2)],certain_closed_edge_ids=['0'],max_snap_m=0)
    row = result['settlements'][0]
    assert row['nearest_baseline_destination']['destination_id'] == 'near'
    assert row['status'] == 'robust_newly_disconnected'
    assert result['any_destination_status_counts_for_comparison'] == {'robust_connected':1}
    assert row['unreachable_certificate']['mapped_destination_node_ids'] == [1]


def test_nearest_uses_directed_road_distance_not_coordinates():
    graph = network([(0,1,100),(0,2,5),(3,0,1)])
    result = analyze_nearest_access(graph,[point('s',0)],[point('close_coordinate',1),point('short_route',2),point('wrong_direction',3)],max_snap_m=0)
    selection=result['settlements'][0]['nearest_baseline_destination']
    assert selection['destination_id']=='short_route'
    assert selection['baseline_road_distance_m']==5


def test_tie_breaking_is_input_order_independent():
    graph = network([(0,1,10),(0,2,10)])
    for destinations in ([point('z',1),point('a',2)],[point('a',2),point('z',1)]):
        row=analyze_nearest_access(graph,[point('s',0)],destinations,max_snap_m=0)['settlements'][0]
        assert row['nearest_baseline_destination']['destination_id']=='a'
        assert row['path_witness']['destination_id']=='a'


def test_uncertain_link_extremes_remain_ambiguous_for_fixed_destination():
    graph=network([(0,1,10),(0,2,20)])
    row=analyze_nearest_access(graph,[point('s',0)],[point('a',1),point('b',2)],uncertain_links={'bridge':['0']},max_snap_m=0)['settlements'][0]
    assert row['status']=='ambiguous'
    assert row['optimistic_path_witness']['destination_id']=='a'
    assert row['pessimistic_cut_certificate']['mapped_destination_node_ids']==[1]


def test_unmapped_and_baseline_disconnected_are_not_new_closures():
    graph=network([(0,1,10),(2,3,5)])
    rows=analyze_nearest_access(graph,[point('unmapped',9),point('baseline',2)],[point('a',1)],max_snap_m=0)['settlements']
    assert [r['status'] for r in rows]==['unmapped','baseline_disconnected']
    assert all(r['nearest_baseline_destination'] is None for r in rows)
    assert all(r['possible_newly_disconnected'] is False for r in rows)


def test_parallel_edges_and_inputs_are_preserved():
    graph=network([(0,1,9),(0,1,2),(1,2,4)])
    points=[point('s',0)];targets=[point('a',1),point('b',2)]
    original=(copy.deepcopy(points),copy.deepcopy(targets),list(graph.edges(keys=True,data=True)))
    row=analyze_nearest_access(graph,points,targets,max_snap_m=0)['settlements'][0]
    assert row['nearest_baseline_destination']['baseline_road_distance_m']==2
    assert points==original[0] and targets==original[1] and list(graph.edges(keys=True,data=True))==original[2]


def test_invalid_weights_and_closures_cannot_hide_in_empty_inputs():
    graph=network([(0,1,1)])
    with pytest.raises(ValueError,match='unknown'):
        analyze_nearest_access(graph,[],[],certain_closed_edge_ids=['bad'])
    graph[0][1]['0']['length_m']=-1
    with pytest.raises(ValueError,match='nonnegative'):
        analyze_nearest_access(graph,[],[])
