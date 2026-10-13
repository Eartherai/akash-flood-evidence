"""Reference-free support audit and fixed-target observation scenarios.

The compact directed graph is built from hash-bound saved physical source road
records. Its source-OSM and fixed-nearest-target provenance is supplied separately.
Unknown/outside coverage is an uncertainty assumption, never observed closure.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import json
import math
import re

from .access import _distance_m

EPSILON = 1e-6


def iter_features(path, chunk_size=65536, max_record_chars=2000000):
    """Stream the bound, positive-size FeatureCollection; never load it whole."""
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as stream:
        buffer = ""
        while True:
            part = stream.read(chunk_size)
            if not part:
                raise ValueError("Missing features array")
            buffer += part
            match = re.match(r'\s*\{\s*"type"\s*:\s*"FeatureCollection"\s*,\s*"features"\s*:\s*\[', buffer)
            if match:
                buffer = buffer[match.end():]
                break
            if len(buffer) > chunk_size * 2:
                raise ValueError("Unsupported FeatureCollection header")
        first = True
        while True:
            buffer = buffer.lstrip()
            while not buffer:
                buffer = stream.read(chunk_size).lstrip()
                if not buffer:
                    raise ValueError("Truncated features array")
            if buffer.startswith("]"):
                tail = buffer[1:] + stream.read()
                if tail.strip() != "}":
                    raise ValueError("Unexpected FeatureCollection trailer")
                return
            if not first:
                if not buffer.startswith(","):
                    raise ValueError("Missing feature separator")
                buffer = buffer[1:].lstrip()
                while not buffer:
                    buffer = stream.read(chunk_size).lstrip()
                    if not buffer:
                        raise ValueError("Truncated feature")
                if buffer.startswith("]"):
                    raise ValueError("Trailing array comma")
            while True:
                try:
                    feature, used = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if len(buffer) > max_record_chars:
                        raise ValueError("Feature exceeds bounded record size")
                    part = stream.read(chunk_size)
                    if not part:
                        raise ValueError("Truncated or invalid feature")
                    buffer += part
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                raise ValueError("Expected feature object")
            yield feature
            buffer = buffer[used:]
            first = False


def support_values(record, measure):
    if record.get("measure") != measure:
        raise ValueError("Support measure differs from declared units")
    values = tuple(record.get(k) for k in ("assessed", "unassessed", "outside_aoi", "total"))
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Support values must be finite and nonnegative")
    if values[3] <= EPSILON or not math.isclose(sum(values[:3]), values[3], abs_tol=3.1e-6, rel_tol=1e-12):
        raise ValueError("Support partition does not sum to total")
    return values


class ObservationGraph:
    def __init__(self):
        self.edges = {}
        self.groups = {}
        self.nodes = {}
        self.forward = defaultdict(list)
        self.reverse = defaultdict(list)
        self.check_limit = lambda: None

    def add(self, feature):
        props = feature["properties"]
        gid = feature["id"]
        endpoints = props["source_endpoint_node_ids"]
        way = props["source_way_id"]
        expected_gid = f"way-{way}-nodes-{min(endpoints)}-{max(endpoints)}"
        if gid in self.groups or gid != props["physical_segment_id"] or gid != expected_gid or len(endpoints) != 2 or endpoints[0] == endpoints[1]:
            raise ValueError("Duplicate or inconsistent physical segment")
        coordinates = feature["geometry"].get("coordinates")
        if feature["geometry"].get("type") != "LineString" or len(coordinates) != 2:
            raise ValueError("Physical segments require exactly two endpoint coordinates")
        coords = []
        for node, coordinate in zip(endpoints, coordinates):
            if len(coordinate) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in coordinate):
                raise ValueError("Invalid node coordinates")
            point = tuple(coordinate)
            if not (-180 <= point[0] <= 180 and -90 <= point[1] <= 90):
                raise ValueError("Node outside longitude/latitude bounds")
            if node in self.nodes and self.nodes[node] != point:
                raise ValueError("Inconsistent coordinate for source node")
            self.nodes[node] = point
            coords.append(point)
        length = _distance_m(*coords)
        center = support_values(props["centerline_coverage"], "length_m")
        corridor = support_values(props["coverage"], "area_m2")
        overlap = props.get("new_water_candidate_overlap")
        if not isinstance(overlap, bool) or overlap != (props["coverage"].get("new_water_candidate", 0) > EPSILON):
            raise ValueError("Candidate overlap flag differs from measured corridor")
        directed = props["source_directed_edges"]
        if not directed or len(directed) != props["source_directed_edge_count"]:
            raise ValueError("Incomplete direction list")
        ids = []
        directions = set()
        for edge in directed:
            eid, u, v = edge["edge_id"], edge["from_node_id"], edge["to_node_id"]
            if eid in self.edges or not isinstance(eid, str) or not eid or {u, v} != set(endpoints) or (u, v) in directions:
                raise ValueError("Duplicate, inconsistent or incomplete source edge")
            directions.add((u, v))
            self.edges[eid] = (u, v, gid, length)
            self.forward[u].append(eid)
            self.reverse[v].append(eid)
            ids.append(eid)
        self.groups[gid] = (tuple(sorted(ids)), center, corridor, overlap)

    def uncertain(self, original):
        for gid, ids in original.items():
            if gid not in self.groups or len(ids) != len(set(ids)) or set(ids) != set(self.groups[gid][0]):
                raise ValueError("Original uncertainty has incomplete source directions")
        candidate = {gid for gid, data in self.groups.items() if data[3]}
        if set(original) != candidate:
            raise ValueError("Original uncertainty differs from candidate-selected roads")
        reasons = {}
        for gid, (_, _, corridor, overlap) in self.groups.items():
            tags = []
            if overlap: tags.append("new_water_candidate")
            if corridor[1] > EPSILON: tags.append("unassessed_within_aoi")
            if corridor[2] > EPSILON: tags.append("outside_aoi")
            if tags: reasons[gid] = tags
        removed = {eid for gid in reasons for eid in self.groups[gid][0]}
        return removed, reasons

    def route_support(self, witness):
        edges, nodes = witness["path_edge_ids"], witness["path_node_ids"]
        if len(nodes) != len(edges) + 1 or nodes[0] != witness["source_node_id"] or nodes[-1] != witness["destination_node_id"]:
            raise ValueError("Invalid witness node chain")
        totals = [0.0] * 4
        rows, graph_distance = [], 0.0
        for i, eid in enumerate(edges):
            if eid not in self.edges:
                raise ValueError("Witness contains missing source edge")
            u, v, gid, length = self.edges[eid]
            if (u, v) != (nodes[i], nodes[i + 1]):
                raise ValueError("Witness direction differs from source edge")
            support = self.groups[gid][1]
            totals = [a + b for a, b in zip(totals, support)]
            graph_distance += length
            rows.append({"edge_id": eid, "physical_segment_id": gid,
                         "from_node_id": u, "to_node_id": v,
                         "centerline_length_m": support[3], "assessed_length_m": support[0],
                         "unknown_within_aoi_length_m": support[1], "outside_aoi_length_m": support[2],
                         "coordinates": [self.nodes[u], self.nodes[v]]})
        if not math.isclose(graph_distance, witness["distance_m"], abs_tol=1e-6, rel_tol=1e-12):
            raise ValueError("Witness length differs from source graph")
        categories = ("assessed", "unknown_within_aoi", "outside_aoi")
        return {"graph_distance_m": graph_distance, "projected_centerline_total_m": totals[3],
                "lengths_m": dict(zip(categories, totals[:3])),
                "fractions_of_projected_centerline": {k: v / totals[3] if totals[3] else None for k, v in zip(categories, totals[:3])},
                "directed_traversal_count": len(edges), "unique_physical_segment_count": len({r["physical_segment_id"] for r in rows}),
                "segments": rows,
                "scope": "Saved source-cell observation support; assessed classification does not establish physical passability."}

    def reverse_reach(self, target, removed):
        next_edges = {target: None}
        queue = deque([target])
        while queue:
            v = queue.popleft()
            if len(next_edges) % 10000 == 0: self.check_limit()
            for eid in self.reverse.get(v, ()):
                if eid in removed: continue
                u = self.edges[eid][0]
                if u not in next_edges:
                    next_edges[u] = eid
                    queue.append(u)
        return next_edges

    def path(self, source, target, next_edges):
        nodes, edges = [source], []
        while nodes[-1] != target:
            eid = next_edges[nodes[-1]]
            edges.append(eid)
            nodes.append(self.edges[eid][1])
        return {"source_node_id": source, "destination_node_id": target, "path_node_ids": nodes,
                "path_edge_ids": edges, "distance_m": sum(self.edges[e][3] for e in edges),
                "route_objective": "shortest hop-count witness; length descriptive"}

    def cut(self, source, target, removed):
        reached = {source}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            if len(reached) % 10000 == 0: self.check_limit()
            for eid in self.forward.get(u, ()):
                v = self.edges[eid][1]
                if eid not in removed and v not in reached:
                    reached.add(v); queue.append(v)
        if target in reached:
            raise ValueError("Cannot certify a reachable target as disconnected")
        boundary = [eid for u in reached for eid in self.forward.get(u, ()) if self.edges[eid][1] not in reached]
        if any(eid not in removed for eid in boundary):
            raise ValueError("Unremoved edge crosses cut")
        return {"source_node_id": source, "destination_node_id": target, "reachable_node_ids": sorted(reached),
                "removed_boundary_edge_ids": sorted(boundary), "scope": "Conditional graph cut; no observed closure or safe-passage claim."}

    def bounds(self, old_proof, removed):
        if old_proof["certain_closed_edge_ids"] or old_proof["destination_switching_after_closure"]:
            raise ValueError("Require zero certain closures and fixed baseline targets")
        rows = []
        groups = defaultdict(list)
        for row in old_proof["settlements"]:
            selected = row["nearest_baseline_destination"]
            if selected:
                if row["snap_node_id"] not in self.nodes or selected["destination_snap_node_id"] not in self.nodes:
                    raise ValueError("Fixed node missing from source graph")
                groups[selected["destination_snap_node_id"]].append(row)
            else:
                if row["status"] not in ("unmapped", "baseline_disconnected"):
                    raise ValueError("Missing target for previously connected settlement")
                rows.append({"id": row["id"], "name": row["name"], "previous_status": row["status"], "status": row["status"],
                             "snap_node_id": row["snap_node_id"], "nearest_baseline_destination": None, "path_witness": None, "cut_certificate": None})
        for target, members in groups.items():
            self.check_limit()
            baseline = self.reverse_reach(target, set())
            worst = self.reverse_reach(target, removed)
            for row in members:
                source = row["snap_node_id"]
                if source not in baseline:
                    raise ValueError("Saved reachable fixed target no longer baseline-reachable")
                connected = source in worst
                status = "robust_connected" if connected else "ambiguous"
                if connected and row["status"] != "robust_connected":
                    raise ValueError("Expanded uncertainty violates monotonicity")
                path = self.path(source, target, worst if connected else baseline)
                path["destination_id"] = row["nearest_baseline_destination"]["destination_id"]
                path["scenario"] = "pessimistic" if connected else "optimistic"
                support = self.route_support(path)
                if connected and any(eid in removed for eid in path["path_edge_ids"]):
                    raise ValueError("Pessimistic path crosses removed edge")
                rows.append({"id": row["id"], "name": row["name"], "previous_status": row["status"], "status": status,
                             "snap_node_id": source, "nearest_baseline_destination": row["nearest_baseline_destination"],
                             "path_witness": path, "path_support": support,
                             "cut_certificate": None if connected else self.cut(source, target, removed)})
        by_id = {r["id"]: r for r in rows}
        if len(by_id) != len(old_proof["settlements"]):
            raise ValueError("Duplicate or missing settlement")
        rows = [by_id[r["id"]] for r in old_proof["settlements"]]
        return {"settlements": rows, "status_counts": dict(Counter(r["status"] for r in rows)),
                "status_transitions": dict(Counter(r["previous_status"] + "->" + r["status"] for r in rows)),
                "confirmed_closures": 0, "certain_closed_edge_ids": [], "inspection_ranking_computed": False,
                "destination_switching_after_closure": False}
