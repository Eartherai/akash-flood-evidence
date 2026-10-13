"""Auditable, CPU-only road connectivity from an Overpass JSON extract.

Distances are metres. Coordinates and edge geometries are longitude/latitude
(EPSG:4326). The default graph screens for motor-vehicle roads. Conditional
restrictions, barriers and turn restrictions still require review.
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
from shapely.geometry import LineString


_EARTH_RADIUS_M = 6_371_008.8
_FALSE_TAGS = {"", "no", "false", "0"}
_DRIVE_HIGHWAYS = {"motorway", "motorway_link", "trunk", "trunk_link", "primary",
                   "primary_link", "secondary", "secondary_link", "tertiary",
                   "tertiary_link", "unclassified", "residential", "living_street",
                   "service", "track", "road"}


def _coordinate(lon: Any, lat: Any) -> tuple[float, float]:
    lon, lat = float(lon), float(lat)
    if not (math.isfinite(lon) and math.isfinite(lat)):
        raise ValueError("Coordinates must be finite")
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError("Coordinates must be longitude/latitude in degrees")
    return lon, lat


def _distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance, with no dependency on projected CRS units."""
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def _truthy_tag(value: Any) -> bool:
    return str(value).strip().lower() not in _FALSE_TAGS


def build_road_graph(osm: dict, *, network_type="drive") -> nx.MultiDiGraph:
    """Build a directed multigraph from Overpass ``elements`` nodes and ways.

    Drive mode excludes non-motor highways and explicit no/private motor access;
    network_type="all" includes all highway ways. Shared node IDs create junctions;
    crossing lines, identical coordinates with different IDs, and layer tags do
    not create new junctions. Bridge endpoints retain their real shared nodes.
    Missing nodes break a way rather than inventing a connecting segment, and
    the missing IDs are recorded in ``graph.graph['missing_node_ids']``.

    ``oneway=yes/true/1``, ``oneway=-1/reverse`` and roundabouts are directed.
    An explicit ``oneway=no/false/0`` overrides a roundabout's implicit one-way.
    Other one-way values are recorded as unresolved and treated as bidirectional.
    Edge IDs are stable ``way_id:segment_index:f/r`` strings. Each edge retains
    ``way_id``, ``length_m``, ``geometry``, ``bridge``, ``tunnel``, ``layer``,
    ``highway``, ``oneway`` and a copy of all source ``tags``.
    """
    if network_type not in {"drive", "all"}:
        raise ValueError("network_type must be drive or all")
    graph = nx.MultiDiGraph(crs="EPSG:4326", distance_units="m", network_type=network_type)
    elements = osm.get("elements", [])
    nodes = {
        int(element["id"]): element
        for element in elements
        if element.get("type") == "node"
    }
    missing: set[int] = set()
    unresolved: set[int] = set()
    for way in elements:
        tags = dict(way.get("tags", {}))
        if way.get("type") != "way" or not tags.get("highway"):
            continue
        motor_access = tags.get("motorcar", tags.get("motor_vehicle", tags.get("vehicle", tags.get("access", "yes"))))
        if network_type == "drive" and (tags["highway"] not in _DRIVE_HIGHWAYS or motor_access in {"no", "private"}):
            continue
        way_id = int(way["id"])
        node_ids = [int(node_id) for node_id in way.get("nodes", [])]
        for node_id in node_ids:
            if node_id not in nodes:
                missing.add(node_id)
                continue
            node = nodes[node_id]
            lon, lat = _coordinate(node["lon"], node["lat"])
            graph.add_node(node_id, x=lon, y=lat, tags=dict(node.get("tags", {})))

        one_way = str(tags.get("oneway", "")).strip().lower()
        if one_way in {"yes", "true", "1"}:
            directions = ("f",)
        elif one_way in {"-1", "reverse"}:
            directions = ("r",)
        elif one_way in {"no", "false", "0"}:
            directions = ("f", "r")
        elif not one_way and tags.get("junction") == "roundabout":
            directions = ("f",)
        else:
            directions = ("f", "r")
            if one_way:
                unresolved.add(way_id)

        for index, (u, v) in enumerate(zip(node_ids, node_ids[1:])):
            if u not in nodes or v not in nodes or u == v:
                continue
            a = (graph.nodes[u]["x"], graph.nodes[u]["y"])
            b = (graph.nodes[v]["x"], graph.nodes[v]["y"])
            for direction in directions:
                start, end = (u, v) if direction == "f" else (v, u)
                coords = (a, b) if direction == "f" else (b, a)
                edge_id = f"{way_id}:{index}:{direction}"
                graph.add_edge(
                    start,
                    end,
                    key=edge_id,
                    edge_id=edge_id,
                    way_id=way_id,
                    length_m=_distance_m(a, b),
                    geometry=LineString(coords),
                    bridge=_truthy_tag(tags.get("bridge", "no")),
                    tunnel=_truthy_tag(tags.get("tunnel", "no")),
                    layer=tags.get("layer", "0"),
                    highway=tags["highway"],
                    oneway=len(directions) == 1,
                    tags=tags.copy(),
                )
    graph.graph["missing_node_ids"] = sorted(missing)
    graph.graph["unresolved_oneway_way_ids"] = sorted(unresolved)
    return graph


def _snap(graph: nx.MultiDiGraph, point: dict, max_snap_m: float) -> dict:
    coordinate = _coordinate(point["lon"], point["lat"])
    if not graph:
        return {"node_id": None, "distance_m": None, "mapped": False}
    nearest = min(
        (
            _distance_m(coordinate, (data["x"], data["y"])),
            str(node_id),
            node_id,
        )
        for node_id, data in graph.nodes(data=True)
    )
    distance_m, _, node_id = nearest
    return {
        "node_id": node_id if distance_m <= max_snap_m else None,
        "distance_m": distance_m,
        "mapped": distance_m <= max_snap_m,
    }


def _reachable(
    graph: nx.MultiDiGraph, source: Any, destinations: list[tuple[dict, dict]]
) -> list[dict]:
    distances, paths = nx.single_source_dijkstra(graph, source, weight="length_m")
    result = []
    for destination, snap in destinations:
        target = snap["node_id"]
        if not snap["mapped"] or target not in distances:
            continue
        path_nodes = paths[target]
        path_edges, way_ids = [], []
        for u, v in zip(path_nodes, path_nodes[1:]):
            # NetworkX uses the minimum-weight parallel edge for each hop.
            key, data = min(
                graph[u][v].items(),
                key=lambda item: (item[1]["length_m"], str(item[0])),
            )
            path_edges.append(data.get("edge_id", f"{u}:{v}:{key}"))
            way_ids.append(data.get("way_id"))
        result.append(
            {
                "id": destination["id"],
                "name": destination.get("name", str(destination["id"])),
                "node_id": target,
                "distance_m": distances[target],
                "destination_snap_distance_m": snap["distance_m"],
                "path_node_ids": path_nodes,
                "path_edge_ids": path_edges,
                "path_way_ids": way_ids,
            }
        )
    return result


def analyze_access(
    graph: nx.MultiDiGraph,
    settlements: list[dict],
    destinations: list[dict],
    closed_way_ids: set[int],
    max_snap_m: float = 500,
    *,
    closed_edge_ids: set[str] | None = None,
) -> list[dict]:
    """Compare each settlement's directed access before and after way closures.

    Points require ``id``, ``lon`` and ``lat``; ``name`` is optional. Each returned
    row includes all reachable ``before``/``after`` destinations and their shortest
    paths. Status is ``unmapped`` when the settlement cannot be snapped or no
    destination is mapped, ``baseline_disconnected`` when none were reachable
    before closures, ``newly_disconnected`` when all formerly reachable ones are
    lost, and ``connected`` otherwise. ``lost_destination_ids`` exposes partial
    loss even when at least one destination remains reachable.

    Snapping uses nearest *nodes*, not edges, with the same baseline node for both
    scenarios. A snap is a geometric association, not proof of an off-road route.
    Route lengths exclude point-to-node gaps, which are returned separately.
    Closures remove every directed segment of each specified way. The input graph
    is never mutated; facility operability and flood-to-road closure inference
    are the caller's responsibility.
    """
    if not math.isfinite(max_snap_m) or max_snap_m < 0:
        raise ValueError("max_snap_m must be finite and nonnegative")
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("graph must be a networkx.MultiDiGraph")
    closed_ids = {int(way_id) for way_id in closed_way_ids}
    after_graph = graph.copy()
    after_graph.remove_edges_from(
        (u, v, key)
        for u, v, key, data in graph.edges(keys=True, data=True)
        if data.get("way_id") in closed_ids or data.get("edge_id") in (closed_edge_ids or set())
    )
    destination_snaps = [(point, _snap(graph, point, max_snap_m)) for point in destinations]
    unmapped_ids = [point["id"] for point, snap in destination_snaps if not snap["mapped"]]
    mapped_count = len(destinations) - len(unmapped_ids)
    rows = []
    for settlement in settlements:
        snap = _snap(graph, settlement, max_snap_m)
        before, after = [], []
        reason = None
        if not snap["mapped"]:
            status, reason = "unmapped", "settlement_outside_snap_limit_or_empty_graph"
        elif not mapped_count:
            status, reason = "unmapped", "no_mapped_destinations"
        else:
            before = _reachable(graph, snap["node_id"], destination_snaps)
            after = _reachable(after_graph, snap["node_id"], destination_snaps)
            status = (
                "baseline_disconnected" if not before
                else "newly_disconnected" if not after
                else "connected"
            )
        after_ids = {destination["id"] for destination in after}
        lost_ids = [destination["id"] for destination in before if destination["id"] not in after_ids]
        rows.append(
            {
                "id": settlement["id"],
                "name": settlement.get("name", str(settlement["id"])),
                "status": status,
                "reason": reason,
                "snap_node_id": snap["node_id"],
                "snap_distance_m": snap["distance_m"],
                "before": before,
                "after": after,
                "before_destination_ids": [destination["id"] for destination in before],
                "after_destination_ids": [destination["id"] for destination in after],
                "lost_destination_ids": lost_ids,
                "partial_access_loss": bool(lost_ids and after),
                "unmapped_destination_ids": unmapped_ids.copy(),
                "mapped_destination_count": mapped_count,
            }
        )
    return rows
