"""Scenario access to a destination fixed by baseline road distance.

No radar candidates become closures here. The selected facility is a mapped
node association, not a verified facility entrance or operating hospital.
"""
from collections import Counter, defaultdict
import math

import networkx as nx

from .access import _snap
from .access_bounds import _edge_index, analyze_access_bounds


def analyze_nearest_access(graph, settlements, destinations, certain_closed_edge_ids=(),
                           uncertain_links=None, max_snap_m=500):
    """Fix each settlement's nearest baseline-reachable destination, then bound.

    'Nearest' means minimum directed baseline road length plus the two fixed
    nearest-node association distances. Equal distances are broken by string
    destination ID. It does not mean travel time or straight-line distance.
    The destination never switches after a closure. Original any-destination
    bounds and their frozen benchmarks are not altered by this adapter.
    """
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError('graph must be a directed multigraph')
    if not isinstance(max_snap_m, (int, float)) or not math.isfinite(max_snap_m) or max_snap_m < 0:
        raise ValueError('max_snap_m must be finite and nonnegative')
    _edge_index(graph)  # Reject missing, duplicate or invalid edge identities/lengths.
    for points in (settlements, destinations):
        if len({point['id'] for point in points}) != len(points):
            raise ValueError('Point IDs must be unique within each input collection')
    certain = tuple(certain_closed_edge_ids)
    uncertain = uncertain_links or {}
    # Also validates all supplied closure groups, including when no point maps.
    any_result = analyze_access_bounds(graph, settlements, destinations, certain, uncertain,
                                       max_snap_m, rank_inspections=False)
    destination_snaps = {point['id']: _snap(graph, point, max_snap_m) for point in destinations}
    destination_by_id = {point['id']: point for point in destinations}
    reverse = graph.reverse(copy=False)
    distances = {key: nx.single_source_dijkstra_path_length(reverse, snap['node_id'], weight='length_m')
                 for key, snap in destination_snaps.items() if snap['mapped']}
    selections, groups = {}, defaultdict(list)
    for point in settlements:
        snap = _snap(graph, point, max_snap_m)
        choices = []
        if snap['mapped']:
            for key, lengths in distances.items():
                if snap['node_id'] in lengths:
                    road = lengths[snap['node_id']]
                    score = road + snap['distance_m'] + destination_snaps[key]['distance_m']
                    choices.append((score, str(key), key, road))
        if choices:
            score, _, key, road = min(choices)
            selections[point['id']] = {'destination_id': key,
                'destination_snap_node_id': destination_snaps[key]['node_id'],
                'baseline_road_distance_m': road, 'baseline_distance_including_association_m': score,
                'destination_snap_distance_m': destination_snaps[key]['distance_m']}
            groups[key].append(point)
    rows_by_id = {row['id']: dict(row, nearest_baseline_destination=None) for row in any_result['settlements']}
    for key, points in groups.items():
        group = analyze_access_bounds(graph, points, [destination_by_id[key]], certain, uncertain,
                                      max_snap_m, rank_inspections=False)
        for row in group['settlements']:
            rows_by_id[row['id']] = dict(row, nearest_baseline_destination=selections[row['id']])
    rows = [rows_by_id[point['id']] for point in settlements]
    return {key: value for key, value in any_result.items() if key not in
            ('settlements', 'status_counts', 'inspection_outcomes', 'inspection_rankings', 'inspection_interpretation')} | {
        'settlements': rows, 'status_counts': dict(Counter(row['status'] for row in rows)),
        'destination_rule': 'Fixed nearest baseline-reachable mapped destination by directed road length plus fixed node-association distances; tie by string destination ID.',
        'destination_switching_after_closure': False,
        'any_destination_status_counts_for_comparison': any_result['status_counts'],
        'facility_scope': 'Mapped node-tagged town/hospital candidates only; no entrance, operating-status or travel-time validation.',
        'hazard_inputs_used': False,
    }
