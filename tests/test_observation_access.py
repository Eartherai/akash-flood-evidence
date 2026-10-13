from copy import deepcopy
import json
import itertools

import pytest

from akash_flood.access import _distance_m
from akash_flood.observation_access import ObservationGraph, iter_features, support_values


def feature(way, u, v, *, unknown=0, outside=0, candidate=False, reverse=True):
    coords = [[u * .001, 0], [v * .001, 0]]
    directed = [{"edge_id": f"{way}:0:f", "from_node_id": u, "to_node_id": v}]
    if reverse: directed.append({"edge_id": f"{way}:0:r", "from_node_id": v, "to_node_id": u})
    def cov(measure, multiplier=1):
        return {"measure": measure, "assessed": (100-unknown-outside)*multiplier, "unassessed":unknown*multiplier,
                "outside_aoi": outside*multiplier, "total":100*multiplier, "new_water_candidate":1 if candidate else 0}
    gid = f"way-{way}-nodes-{min(u,v)}-{max(u,v)}"
    return {"type":"Feature", "id":gid,"geometry":{"type":"LineString","coordinates":coords},
            "properties":{"physical_segment_id":gid,"source_way_id":way,"source_endpoint_node_ids":[u,v],
                "source_directed_edges":directed,"source_directed_edge_count":len(directed),
                "centerline_coverage":cov("length_m"),"coverage":cov("area_m2",20),"new_water_candidate_overlap":candidate}}


def proof(source=1, target=3, old_status="robust_connected"):
    return {"certain_closed_edge_ids": [], "uncertain_links":{},"destination_switching_after_closure":False,
            "settlements":[{"id":"village","name":"Village","snap_node_id":source,"status":old_status,
                "nearest_baseline_destination":{"destination_id":"town","destination_snap_node_id":target}}]}


def graph(*features):
    g=ObservationGraph()
    for f in features:g.add(f)
    return g


def test_unknown_on_only_route_changes_robust_to_ambiguous():
    g=graph(feature(10,1,2,unknown=20),feature(20,2,3))
    removed,reasons=g.uncertain({})
    result=g.bounds(proof(),removed)
    assert result['status_counts']=={'ambiguous':1}
    row=result['settlements'][0]
    assert row['cut_certificate']['removed_boundary_edge_ids']==['10:0:f']
    assert row['path_witness']['destination_node_id']==3
    assert reasons['way-10-nodes-1-2']==['unassessed_within_aoi']


def test_observed_alternate_keeps_robust_connectivity():
    g=graph(feature(10,1,2,unknown=20),feature(20,2,3),feature(30,1,4),feature(40,4,3))
    removed,_=g.uncertain({})
    row=g.bounds(proof(),removed)['settlements'][0]
    assert row['status']=='robust_connected'
    assert row['path_witness']['path_node_ids']==[1,4,3]
    assert row['cut_certificate'] is None


def test_fixed_nearest_target_does_not_switch_to_available_farther_target():
    g=graph(feature(10,1,2,outside=20),feature(20,2,3),feature(30,1,4))
    removed,_=g.uncertain({})
    row=g.bounds(proof(),removed)['settlements'][0]
    assert row['status']=='ambiguous'
    assert row['nearest_baseline_destination']['destination_id']=='town'
    assert row['path_witness']['destination_node_id']==3


def test_oneway_and_parallel_physical_segments_remain_distinct():
    g=graph(feature(10,1,2,unknown=20,reverse=False),feature(11,1,2,reverse=False),feature(20,2,3,reverse=False))
    removed,_=g.uncertain({})
    assert removed=={'10:0:f'}
    assert g.bounds(proof(),removed)['status_counts']=={'robust_connected':1}
    assert 1 not in g.reverse_reach(1,set()) or g.reverse_reach(1,set())[1] is None
    assert 3 not in g.reverse_reach(1,set())


def test_original_candidate_groups_keep_every_direction():
    g=graph(feature(10,1,2,candidate=True),feature(20,2,3))
    with pytest.raises(ValueError,match='incomplete'):g.uncertain({'way-10-nodes-1-2':['10:0:f']})
    with pytest.raises(ValueError,match='differs'):g.uncertain({})
    removed,reasons=g.uncertain({'way-10-nodes-1-2':['10:0:f','10:0:r']})
    assert removed=={'10:0:f','10:0:r'}
    assert reasons['way-10-nodes-1-2']==['new_water_candidate']


def test_support_length_denominator_and_repeated_traversal():
    g=graph(feature(10,1,2,unknown=10,outside=20))
    witness={'source_node_id':1,'destination_node_id':2,'path_node_ids':[1,2,1,2],
        'path_edge_ids':['10:0:f','10:0:r','10:0:f'],
        'distance_m':3*_distance_m((.001,0),(.002,0))}
    support=g.route_support(witness)
    assert support['projected_centerline_total_m']==300
    assert support['fractions_of_projected_centerline']=={'assessed':.7,'unknown_within_aoi':.1,'outside_aoi':.2}
    assert support['directed_traversal_count']==3
    assert support['unique_physical_segment_count']==1
    assert support['graph_distance_m']!=support['projected_centerline_total_m']


@pytest.mark.parametrize('edit', ['missing','direction','distance','nodes'])
def test_invalid_witness_rejected(edit):
    g=graph(feature(10,1,2)); w=g.path(1,2,g.reverse_reach(2,set()))
    if edit=='missing':w['path_edge_ids'][0]='missing'
    elif edit=='direction':w['path_edge_ids'][0]='10:0:r'
    elif edit=='distance':w['distance_m']+=1
    else:w['path_node_ids']=[1,3]
    with pytest.raises(ValueError):g.route_support(w)


@pytest.mark.parametrize('edit', ['duplicate','coordinate','measure','partition','nonfinite','overlap'])
def test_invalid_source_records_rejected(edit):
    f=feature(10,1,2); g=graph(f)
    bad=feature(20,2,3)
    if edit=='duplicate':bad=f
    elif edit=='coordinate':bad['geometry']['coordinates'][0]=[1,1]
    elif edit=='measure':bad['properties']['centerline_coverage']['measure']='area_m2'
    elif edit=='partition':bad['properties']['coverage']['assessed']=1
    elif edit=='nonfinite':bad['properties']['coverage']['total']=float('nan')
    else:bad['properties']['new_water_candidate_overlap']=True
    with pytest.raises(ValueError):g.add(bad)


def test_expanded_uncertainty_monotonicity_rejected():
    g=graph(feature(10,1,2),feature(20,2,3))
    with pytest.raises(ValueError,match='monotonicity'):g.bounds(proof(old_status='ambiguous'),set())


def test_preserves_unmapped_and_baseline_disconnected_rows():
    p=proof();p['settlements']=[]
    for i,status in enumerate(['unmapped','baseline_disconnected']):
        p['settlements'].append({'id':str(i),'name':str(i),'snap_node_id':None,'nearest_baseline_destination':None,'status':status})
    assert graph(feature(10,1,2)).bounds(p,set())['status_counts']=={'unmapped':1,'baseline_disconnected':1}


def test_bounds_agree_with_exhaustive_tiny_assignments():
    g=graph(feature(10,1,2,unknown=10),feature(20,2,3),feature(30,1,4,outside=10),feature(40,4,3))
    removed,reasons=g.uncertain({})
    outcomes=[]
    group_edges=[g.groups[gid][0] for gid in reasons]
    for states in itertools.product([False,True],repeat=len(group_edges)):
        closed={eid for state,edges in zip(states,group_edges) if state for eid in edges}
        outcomes.append(1 in g.reverse_reach(3,closed))
    assert any(outcomes) and not all(outcomes)
    assert g.bounds(proof(),removed)['status_counts']=={'ambiguous':1}


def test_streaming_collection_tiny_chunks(tmp_path):
    values=[feature(10,1,2),feature(20,2,3)]
    p=tmp_path/'roads.json';p.write_text(json.dumps({'type':'FeatureCollection','features':values},indent=2))
    assert list(iter_features(p,chunk_size=47))==values


@pytest.mark.parametrize('tail', [',]}','] garbage',']})'])
def test_streaming_rejects_corrupt_trailer(tmp_path,tail):
    p=tmp_path/'roads.json';p.write_text('{"type":"FeatureCollection","features":['+json.dumps(feature(10,1,2))+tail)
    with pytest.raises(ValueError):list(iter_features(p,chunk_size=100))


def test_streaming_rejects_duplicate_direction(tmp_path):
    f=feature(10,1,2);f['properties']['source_directed_edges'][1]=deepcopy(f['properties']['source_directed_edges'][0])
    with pytest.raises(ValueError):graph(f)
