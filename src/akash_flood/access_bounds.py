"""Exact directed access bounds for explicitly supplied closure scenarios.

This module does not infer flood, damage, or real road passability. A certain
closure means fixed closed *in the supplied scenario*. Each uncertain physical
link is either wholly open or wholly closed, including both driving directions
when present. Every combination is allowed; no probabilities are assumed.
"""

from collections import Counter, deque
from collections.abc import Iterable, Mapping
import math

import networkx as nx

from .access import _snap


def _ordered(values):
    return sorted(values, key=lambda value: (type(value).__name__, str(value)))


def _edge_index(graph):
    index = {}
    for u, v, key, data in graph.edges(keys=True, data=True):
        edge_id = data.get("edge_id")
        if not isinstance(edge_id, str) or not edge_id or edge_id in index:
            raise ValueError("Every directed edge must have a unique nonempty string edge_id")
        length = data.get("length_m")
        if not isinstance(length, (int, float)) or not math.isfinite(length) or length < 0:
            raise ValueError("Every edge must have finite nonnegative length_m")
        index[edge_id] = (u, v, key, data)
    return index


def _ids(value, label):
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an iterable of edge IDs, not a string")
    result = set(value)
    if any(not isinstance(item, str) for item in result):
        raise ValueError(f"{label} must contain string edge IDs")
    return result


def _closures(index, certain_closed_edge_ids, uncertain_links):
    certain = _ids(certain_closed_edge_ids, "certain_closed_edge_ids")
    if not certain <= index.keys():
        raise ValueError("Certain closure references unknown edge IDs")
    if not isinstance(uncertain_links, Mapping):
        raise ValueError("uncertain_links must map physical-link IDs to directed edge IDs")
    groups, assigned = {}, set()
    if any(not isinstance(link_id, str) or not link_id for link_id in uncertain_links):
        raise ValueError("Physical-link IDs must be nonempty strings")
    for link_id in sorted(uncertain_links):
        edges = _ids(uncertain_links[link_id], f"uncertain link {link_id}")
        if not edges or not edges <= index.keys():
            raise ValueError("Uncertain link must contain known, nonempty edge IDs")
        if edges & (certain | assigned):
            raise ValueError("Certain closures and uncertain physical-link groups must be disjoint")
        # A physical link is one way segment, not a whole multi-segment road.
        # One-way segments are valid singleton groups. Parallel ways are separate.
        physical = {(frozenset(index[e][:2]), index[e][3].get("way_id")) for e in edges}
        if len(physical) != 1:
            raise ValueError("An uncertain physical link must share endpoints and way_id")
        endpoints, way_id = next(iter(physical))
        if way_id is None:
            raise ValueError("Uncertain edges require way_id to identify their physical link")
        complete = {
            e for e, (u, v, _, data) in index.items()
            if frozenset((u, v)) == endpoints and data.get("way_id") == way_id
        }
        if complete != edges:
            raise ValueError("An uncertain link must include every direction of its physical segment")
        groups[link_id] = edges
        assigned |= edges
    return certain, groups, assigned


def _reverse_routes(graph, target_nodes, removed):
    """Unweighted multi-source reverse BFS; next edges witness any valid route.

    This finds shortest *hop-count* witnesses, not fastest/shortest-distance
    routes. Parallel edges and one-way directions remain distinct.
    """
    roots = {node: node for node in _ordered(target_nodes)}
    next_edges = {}
    queue = deque(roots)
    while queue:
        v = queue.popleft()
        for u in _ordered(graph.pred[v]):
            if u in roots:
                continue
            allowed = [data for data in graph.pred[v][u].values() if data["edge_id"] not in removed]
            if not allowed:
                continue
            edge = min(allowed, key=lambda data: data["edge_id"])
            roots[u] = roots[v]
            next_edges[u] = (v, edge["edge_id"])
            queue.append(u)
    return roots, next_edges


def _path_witness(source, routes, destination_by_node, index, scenario):
    roots, next_edges = routes
    node = source
    nodes, edges, distance = [node], [], 0.0
    while node in next_edges:
        node, edge_id = next_edges[node]
        nodes.append(node)
        edges.append(edge_id)
        distance += index[edge_id][3]["length_m"]
    destination, snap = destination_by_node[roots[source]]
    return {
        "scenario": scenario,
        "source_node_id": source,
        "destination_id": destination["id"],
        "destination_node_id": roots[source],
        "destination_snap_distance_m": snap["distance_m"],
        "path_node_ids": nodes,
        "path_edge_ids": edges,
        "distance_m": distance,
        "route_objective": "shortest hop count; distance is descriptive",
    }


def _unreachable_certificate(graph, source, target_nodes, removed, scenario):
    reachable = {source}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v, edges in graph.succ[u].items():
            if v not in reachable and any(data["edge_id"] not in removed for data in edges.values()):
                reachable.add(v)
                queue.append(v)
    if reachable & target_nodes:
        raise AssertionError("Cannot certify disconnection when a destination is reachable")
    boundary = []
    for u in _ordered(reachable):
        for v, edges in graph.succ[u].items():
            if v not in reachable:
                for data in edges.values():
                    edge_id = data["edge_id"]
                    if edge_id not in removed:
                        raise AssertionError("A retained edge leaves the claimed reachable cut")
                    boundary.append({"from_node_id": u, "to_node_id": v, "edge_id": edge_id})
    boundary.sort(key=lambda row: row["edge_id"])
    return {
        "scenario": scenario,
        "source_node_id": source,
        "reachable_node_ids": _ordered(reachable),
        "mapped_destination_node_ids": _ordered(target_nodes),
        "outgoing_cut_boundary": boundary,
        "removed_edge_ids": [edge["edge_id"] for edge in boundary],
        "certificate_scope": "graph-theoretic only, conditional on supplied closures and baseline OSM",
    }


def analyze_access_bounds(
    graph: nx.MultiDiGraph,
    settlements: list[dict],
    destinations: list[dict],
    certain_closed_edge_ids: Iterable[str] = (),
    uncertain_links: Mapping[str, Iterable[str]] | None = None,
    max_snap_m: float = 500,
    *,
    rank_inspections: bool = True,
) -> dict:
    """Bound access to *any* mapped destination over all uncertain-link states.

    Snaps use the existing access module's nearest baseline node association and
    never move after closures. Inputs are not mutated. ``robust_connected``
    means a route survives with every uncertain link closed. Conversely,
    ``robust_newly_disconnected`` means no route survives even with every
    uncertain link open, after baseline access has been verified. ``ambiguous``
    means both connected and disconnected assignments exist. Unmapped points
    and baseline disconnections are reported separately.

    Certificates concern the supplied graph, not observed physical passability.
    The complete uncertainty set allows every 2**k combination. If extra joint
    constraints exist, these extrema can be conservative rather than attainable.

    Inspection rankings count currently ambiguous settlements that become
    determinate in each separate one-link open/closed branch. They are exact
    branch outcomes, not expected information gain; no branch probability or
    combined scalar ranking is assumed.
    """
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("graph must be a networkx.MultiDiGraph")
    if not math.isfinite(max_snap_m) or max_snap_m < 0:
        raise ValueError("max_snap_m must be finite and nonnegative")
    for name, points in (("settlements", settlements), ("destinations", destinations)):
        if len({point["id"] for point in points}) != len(points):
            raise ValueError(f"{name} must have unique IDs")
    index = _edge_index(graph)
    certain, groups, uncertain = _closures(index, certain_closed_edge_ids, uncertain_links or {})
    destination_snaps = [(point, _snap(graph, point, max_snap_m)) for point in destinations]
    destination_by_node = {}
    for point, snap in sorted(destination_snaps, key=lambda row: str(row[0]["id"])):
        if snap["mapped"]:
            destination_by_node.setdefault(snap["node_id"], (point, snap))
    targets = set(destination_by_node)
    unmapped_destinations = [point["id"] for point, snap in destination_snaps if not snap["mapped"]]
    baseline = _reverse_routes(graph, targets, set())
    optimistic = _reverse_routes(graph, targets, certain)
    pessimistic_removed = certain | uncertain
    pessimistic = _reverse_routes(graph, targets, pessimistic_removed)
    rows = []
    for point in settlements:
        snap = _snap(graph, point, max_snap_m)
        source = snap["node_id"]
        row = {
            "id": point["id"], "name": point.get("name", str(point["id"])),
            "snap_node_id": source, "snap_distance_m": snap["distance_m"],
            "mapped_destination_count": len(destinations) - len(unmapped_destinations),
            "unmapped_destination_ids": unmapped_destinations.copy(),
            "baseline_reachable": None, "optimistic_reachable": None,
            "pessimistic_reachable": None, "possible_newly_disconnected": False,
            "path_witness": None, "unreachable_certificate": None,
            "optimistic_path_witness": None, "pessimistic_cut_certificate": None,
            "reason": None,
        }
        if not snap["mapped"]:
            row.update(status="unmapped", reason="settlement_outside_snap_limit_or_empty_graph")
        elif not targets:
            row.update(status="unmapped", reason="no_mapped_destinations")
        else:
            before = source in baseline[0]
            upper, lower = source in optimistic[0], source in pessimistic[0]
            row.update(baseline_reachable=before, optimistic_reachable=upper, pessimistic_reachable=lower)
            if not before:
                row["status"] = "baseline_disconnected"
            elif lower:
                row["status"] = "robust_connected"
                row["path_witness"] = _path_witness(source, pessimistic, destination_by_node, index, "pessimistic")
            elif not upper:
                row.update(status="robust_newly_disconnected", possible_newly_disconnected=True)
                row["unreachable_certificate"] = _unreachable_certificate(graph, source, targets, certain, "optimistic")
            else:
                row.update(status="ambiguous", possible_newly_disconnected=True)
                row["optimistic_path_witness"] = _path_witness(source, optimistic, destination_by_node, index, "optimistic")
                row["pessimistic_cut_certificate"] = _unreachable_certificate(graph, source, targets, pessimistic_removed, "pessimistic")
        rows.append(row)

    ambiguous = [row for row in rows if row["status"] == "ambiguous"]
    inspections = []
    if rank_inspections:
        for link_id, edge_ids in groups.items():
            # Open fixes one link open; closed fixes it closed. All other links
            # remain uncertain. Baseline snap associations never change.
            open_lower = _reverse_routes(graph, targets, pessimistic_removed - edge_ids)[0]
            closed_upper = _reverse_routes(graph, targets, certain | edge_ids)[0]
            resolved_open = [row["id"] for row in ambiguous if row["snap_node_id"] in open_lower]
            resolved_closed = [row["id"] for row in ambiguous if row["snap_node_id"] not in closed_upper]
            inspections.append({
                "link_id": link_id, "edge_ids": sorted(edge_ids),
                "if_open": {"resolved_count": len(resolved_open), "resolved_settlement_ids": resolved_open,
                            "resolved_status": "robust_connected", "remaining_ambiguous_count": len(ambiguous) - len(resolved_open)},
                "if_closed": {"resolved_count": len(resolved_closed), "resolved_settlement_ids": resolved_closed,
                              "resolved_status": "robust_newly_disconnected", "remaining_ambiguous_count": len(ambiguous) - len(resolved_closed)},
            })
    return {
        "scope": "explicit closure-scenario feasibility proof; no flood or physical passability inference",
        "uncertainty_model": "each disjoint physical link jointly open/closed; all combinations allowed",
        "certain_closed_edge_ids": sorted(certain),
        "uncertain_links": {key: sorted(value) for key, value in groups.items()},
        "max_snap_m": max_snap_m,
        "settlements": rows,
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "inspection_outcomes": inspections,
        "inspection_rankings": {
            branch: [item["link_id"] for item in sorted(inspections, key=lambda item: (-item[branch]["resolved_count"], item["link_id"]))]
            for branch in ("if_open", "if_closed")
        },
        "inspection_interpretation": "separate branch counts on initially ambiguous settlements; not expected information gain",
    }
