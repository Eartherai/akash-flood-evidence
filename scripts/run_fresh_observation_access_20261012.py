"""Freeze and audit observation support for one bound fresh exposure case.

This adapter reads saved source records and conditional graph evidence. It never
decodes image pixels, loads a model, trains, downloads, or reads checking labels.
Absolute SHA256 bindings avoid dependence on the two historical case folders.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import sys
import time

from akash_flood.catalog import validate_bbox
from akash_flood.fresh_case import event_boundary, reject_source_path
from akash_flood.observation_access import EPSILON, ObservationGraph, iter_features

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "akash-fresh-observation-access-protocol-v1"
RECEIPT_SCHEMA = "akash-fresh-observation-access-receipt-v1"
LIMITS = {"maximum_wall_seconds": 300, "maximum_process_rss_bytes": 2_000_000_000,
          "maximum_physical_segments": 300_000, "maximum_directed_edges": 600_000,
          "maximum_point_features": 10_000, "maximum_json_bytes": 100_000_000,
          "maximum_feature_chars": 2_000_000}
POLICY = {"positive_area_tolerance_m2": EPSILON,
          "candidate_groups": "Preserve every original new-water candidate-selected physical group.",
          "additional_uncertainty": "Every physical segment with positive-area unknown-within-AOI or outside-AOI support in its saved 20 m association corridor is jointly open/closed in all source driving directions.",
          "certain_closures": [], "all_combinations_allowed": True,
          "destination": "Preserve original point associations and each fixed nearest baseline-reachable mapped target; never switch destinations after closures.",
          "observed_nonwater_is_passable": False, "unknown_is_observed_closed": False,
          "witness_support": "Per-directed-traversal projected centreline lengths; retain assessed, unknown-within-AOI and outside-AOI denominators, distinct from great-circle graph distance.",
          "empty_path": "Zero-length graph witness fractions remain null; physical point-to-road and facility connection are unvalidated."}
CLAIMS = {"observed_closures": False, "safe_routes": False, "local_hazard_accuracy": False,
          "model_inference": False, "training": False, "network": False, "gpu": False,
          "reference_geometry_loaded": False}
SOURCE_CODE = [Path(__file__), ROOT / "src/akash_flood/observation_access.py",
               ROOT / "src/akash_flood/access.py", ROOT / "src/akash_flood/fresh_case.py",
               ROOT / "src/akash_flood/catalog.py"]
REQUIRED_OUTPUTS = {"summary.json", "roads.geojson", "settlements.geojson", "destinations.geojson",
                    "access-proof.json", "scenario-policy.json"}
STATUSES = {"robust_connected", "ambiguous", "unmapped", "baseline_disconnected"}


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_path(path):
    path = Path(path)
    reject_source_path(path)
    resolved = path.resolve()
    reject_source_path(resolved)
    for part in resolved.parts:
        if re.fullmatch(r"references?|reference[-_ ]labels?|checking[-_ ]labels?|evaluation[-_ ]only", part, re.I):
            raise ValueError("Checking/reference paths are not observation inputs")
    return resolved


def record(path):
    path = safe_path(path)
    if not path.is_file():
        raise ValueError("Expected a regular bound source file: " + path.name)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def read_json(path):
    path = safe_path(path)
    if path.stat().st_size > LIMITS["maximum_json_bytes"]:
        raise ValueError("JSON evidence exceeds the bounded metadata limit")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field: " + key)
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON constant")))
    if not isinstance(value, dict):
        raise ValueError("Evidence JSON must be an object")
    return value


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def checked(item, parent):
    if (not isinstance(item, dict) or not {"path", "bytes", "sha256"} <= item.keys()
            or not isinstance(item["path"], str)
            or isinstance(item["bytes"], bool) or not isinstance(item["bytes"], int) or item["bytes"] < 0
            or not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
        raise ValueError("Expected path, finite integer byte count and SHA256 binding")
    path = Path(item["path"])
    path = safe_path(path if path.is_absolute() else Path(parent) / path)
    actual = record(path)
    if actual["bytes"] != item["bytes"] or actual["sha256"] != item["sha256"]:
        raise ValueError("Changed source checksum or size: " + path.name)
    return path


def nested_records(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            yield value
        for child in value.values():
            yield from nested_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_records(child)


def exposure_contract(manifest_path, check_limit=lambda: None):
    """Check all exposure-bound bytes; read metadata only, never image/model content."""
    manifest_path = safe_path(manifest_path)
    binding = read_json(manifest_path)
    if (binding.get("schema") != "akash-fresh-exposure-binding-v1" or binding.get("status") != "completed"
            or binding.get("reference_labels_loaded") is not False or binding.get("model_inference_run") is not False
            or type(binding.get("network_requests")) is not int or binding["network_requests"] != 0):
        raise ValueError("Require completed reference-free fresh exposure binding")
    outputs = binding.get("outputs")
    if not isinstance(outputs, dict) or not REQUIRED_OUTPUTS <= outputs.keys():
        raise ValueError("Exposure binding is missing required saved outputs")
    if not isinstance(binding.get("inputs"), list) or not isinstance(binding.get("code"), list):
        raise ValueError("Require exposure input and code bindings")
    records = {str(manifest_path): record(manifest_path)}
    for item in nested_records(binding):
        path = checked(item, manifest_path.parent)
        records[str(path)] = record(path)
        check_limit()
    paths = {name: checked(item, manifest_path.parent) for name, item in outputs.items()}
    for name, path in paths.items():
        if path.name != name:
            raise ValueError("Named exposure output and source filename disagree")
    summary = read_json(paths["summary.json"])
    proof = read_json(paths["access-proof.json"])
    scenario = read_json(paths["scenario-policy.json"])
    case = binding.get("case")
    if (not isinstance(case, dict) or summary.get("case") != case or proof.get("case") != case
            or summary.get("schema") != "akash-fresh-exposure-v1" or summary.get("status") != "completed"
            or summary.get("reference_labels_loaded") is not False or summary.get("gpu_used") is not False):
        raise ValueError("Exposure summary/proof case or completion status differs")
    bbox, context = validate_bbox(case["bbox"]), validate_bbox(case["context_bbox"])
    if (not all(math.isfinite(v) for v in [*bbox, *context])
            or not (context[0] <= bbox[0] < bbox[2] <= context[2] and context[1] <= bbox[1] < bbox[3] <= context[3])):
        raise ValueError("Case AOI must be finite and contained in its context")
    event_boundary(case["event_date"], case["osm_snapshot"])
    if (proof.get("certain_closed_edge_ids") != [] or proof.get("destination_switching_after_closure") is not False
            or proof.get("observed_hazard_or_closures_used") is not False or proof.get("max_snap_m") != 500
            or scenario.get("scenario_is_hypothetical") is not True or scenario.get("certain_closed_edge_ids") != []
            or scenario.get("confirmed_closures") != 0 or scenario.get("positive_overlap_area_tolerance_m2") != EPSILON):
        raise ValueError("Require original hypothetical candidate-only fixed-target scenario")
    rows = proof.get("settlements")
    if not isinstance(rows, list) or len(rows) > LIMITS["maximum_point_features"]:
        raise ValueError("Settlement scope exceeds the bounded point limit")
    if (any(not isinstance(r, dict) or r.get("status") not in STATUSES for r in rows)
            or len({r["id"] for r in rows}) != len(rows)):
        raise ValueError("Invalid or duplicate original settlement statuses")
    actual = dict(Counter(r["status"] for r in rows))
    access = summary["access_scenario"]
    if (actual != proof.get("status_counts") or actual != access.get("status_counts")
            or len(rows) != access.get("n_settlements") or access.get("scenario_is_hypothetical") is not True
            or access.get("confirmed_closures") != 0 or access.get("destination_switching_after_closure") is not False):
        raise ValueError("Original summary and proof status/target scope disagree")
    water_path = checked(binding["water_source"], manifest_path.parent)
    water = read_json(water_path)
    if (water.get("schema") != "akash-fresh-water-v1" or water.get("available") is not True
            or water.get("reference_labels_loaded") is not False or water.get("case") != case
            or summary["water_source"] != binding["water_source"] or proof["water_source"] != binding["water_source"]
            or summary["temporal_counts"] != water.get("counts") or proof["temporal_counts"] != water.get("counts")
            or summary.get("source_dates") != water.get("source_dates")):
        raise ValueError("Exposure water source, dates, counts or case disagree")
    plan_path = checked(binding["plan_source"], manifest_path.parent)
    plan = read_json(plan_path)
    if plan.get("case") != case or summary["plan_source"] != binding["plan_source"] or proof["source_plan"] != binding["plan_source"]:
        raise ValueError("Exposure planning source differs from case")
    if checked(binding["summary"], manifest_path.parent) != paths["summary.json"]:
        raise ValueError("Summary binding points to a different output")
    return {"binding": binding, "manifest": record(manifest_path), "case": case,
            "paths": paths, "summary": summary, "proof": proof,
            "records": [records[k] for k in sorted(records)]}


def freeze(exposure_manifest, out):
    out = safe_path(out)
    failure = out.with_name(out.name + ".failure.json")
    if out.exists() or out.is_symlink() or failure.exists():
        raise FileExistsError("Preserve prior freeze attempts; choose a new protocol path")
    try:
        contract = exposure_contract(exposure_manifest)
        protocol = {"schema": SCHEMA, "status": "frozen", "frozen_at": now(),
                    "exposure_manifest": contract["manifest"], "case": contract["case"],
                    "records": contract["records"], "code": [record(p) for p in SOURCE_CODE],
                    "policy": POLICY, "limits": LIMITS, "claims": CLAIMS,
                    "original_status_counts": contract["proof"]["status_counts"],
                    "new_graph_outcomes_read_before_freeze": False,
                    "scope": "One saved case's conditional observation-support scenario; no new hazard inference, independent nearest-target selection or physical road validation.",
                    "failure_policy": "Preserve every failed attempt; no retries or outcome-driven policy changes."}
        write_new(out, protocol)
        return record(out)
    except BaseException as error:
        write_new(failure, {"status": "failed", "failed_at": now(), "operation": "freeze",
                            "exposure_manifest": str(exposure_manifest), "error_type": type(error).__name__,
                            "message": str(error), "no_automatic_retry": True, "claims": CLAIMS})
        raise


def rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def read_points(path, check_limit):
    points = {}
    for feature in iter_features(path, max_record_chars=LIMITS["maximum_feature_chars"]):
        identifier = feature["id"]
        geometry, props = feature["geometry"], feature["properties"]
        point = geometry.get("coordinates")
        if (identifier in points or geometry.get("type") != "Point" or not isinstance(point, list) or len(point) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point)
                or not (-180 <= point[0] <= 180 and -90 <= point[1] <= 90)):
            raise ValueError("Duplicate or invalid source point geometry")
        if props.get("id") != identifier or props.get("lon") != point[0] or props.get("lat") != point[1]:
            raise ValueError("Point geometry and source identity/coordinates disagree")
        points[identifier] = {"coordinates": point, "properties": props}
        if len(points) > LIMITS["maximum_point_features"]:
            raise ValueError("Point feature cap exceeded")
        if len(points) % 100 == 0:
            check_limit()
    return points


def forward_dfs(adjacency, source, removed, check_limit):
    """Independent forward stack traversal, separate from reverse-BFS solver."""
    reached, stack = {source}, [source]
    visits = 0
    while stack:
        u = stack.pop()
        for v, edge_id in adjacency.get(u, ()):
            if edge_id not in removed and v not in reached:
                reached.add(v)
                stack.append(v)
        visits += 1
        if visits % 1000 == 0:
            check_limit()
    return reached


def verify_extremes(adjacency, graph, original, expanded, original_removed, removed, check_limit):
    """Recompute fixed-target extreme outcomes and validate each new witness/cut."""
    old_rows = {r["id"]: r for r in original["settlements"]}
    checks, no_target = [], 0
    for row in expanded["settlements"]:
        check_limit()
        old = old_rows[row["id"]]
        selection = old["nearest_baseline_destination"]
        if selection is None:
            if row["nearest_baseline_destination"] is not None or row["status"] != old["status"]:
                raise ValueError("No-target settlement status/association changed")
            no_target += 1
            continue
        source, target = old["snap_node_id"], selection["destination_snap_node_id"]
        baseline = forward_dfs(adjacency, source, set(), check_limit)
        original_worst = forward_dfs(adjacency, source, original_removed, check_limit)
        worst = forward_dfs(adjacency, source, removed, check_limit)
        if target not in baseline:
            raise ValueError("Independent DFS contradicts saved baseline reachable fixed target")
        expected_old = "robust_connected" if target in original_worst else "ambiguous"
        expected = "robust_connected" if target in worst else "ambiguous"
        if old["status"] != expected_old or row["status"] != expected or row["nearest_baseline_destination"] != selection:
            raise ValueError("Independent forward DFS disagrees with fixed-target extreme outcome")
        witness = row["path_witness"]
        if (witness["destination_id"] != selection["destination_id"] or witness["destination_node_id"] != target
                or witness["source_node_id"] != source or graph.route_support(witness) != row["path_support"]):
            raise ValueError("New witness changed fixed target/source support")
        if expected == "robust_connected":
            if row["cut_certificate"] is not None or any(e in removed for e in witness["path_edge_ids"]):
                raise ValueError("Robust witness crosses removed edge or has a cut")
        else:
            cut = row["cut_certificate"]
            if cut["source_node_id"] != source or cut["destination_node_id"] != target or set(cut["reachable_node_ids"]) != worst:
                raise ValueError("New cut differs from independent forward reachability")
            boundary = {eid for u in worst for v, eid in adjacency.get(u, ()) if v not in worst}
            if set(cut["removed_boundary_edge_ids"]) != boundary or not boundary <= removed:
                raise ValueError("New cut boundary differs from independently checked source edges")
        checks.append({"settlement_id": row["id"], "source_node_id": source, "fixed_destination_node_id": target,
                       "all_open_reachable": True, "original_all_closed_reachable": target in original_worst,
                       "expanded_all_closed_reachable": target in worst, "status": expected})
    return {"method": "Separate forward stack DFS over independently collected source-directed edges; checks original and expanded extremes, new route support and exact cut boundaries.",
            "fixed_target_settlements_checked": len(checks), "no_target_rows_preserved": no_target,
            "passed": True, "checks": checks, "physical_passability_validated": False,
            "nearest_baseline_destination_selection_rerun": False}


def run(protocol_path, out):
    protocol_path, out = safe_path(protocol_path), safe_path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError("Preserve prior executions; use a fresh output directory")
    out.mkdir(parents=True)
    started, cpu_started = time.perf_counter(), time.process_time()
    write_new(out / "started.json", {"started_at": now(), "protocol_path": str(protocol_path), "limits": LIMITS})
    def bounded():
        if time.perf_counter() - started > LIMITS["maximum_wall_seconds"]:
            raise TimeoutError("Observation analysis 300 second ceiling exceeded")
        if rss_bytes() > LIMITS["maximum_process_rss_bytes"]:
            raise MemoryError("Observation analysis 2 GB RSS ceiling exceeded")
    try:
        protocol = read_json(protocol_path)
        if (protocol.get("schema") != SCHEMA or protocol.get("status") != "frozen"
                or protocol.get("policy") != POLICY or protocol.get("limits") != LIMITS or protocol.get("claims") != CLAIMS):
            raise ValueError("Require the unchanged declared observation protocol")
        if protocol.get("code") != [record(p) for p in SOURCE_CODE]:
            raise ValueError("Observation adapter code changed after freeze")
        for item in protocol["records"]:
            checked(item, protocol_path.parent)
            bounded()
        manifest_path = checked(protocol["exposure_manifest"], protocol_path.parent)
        contract = exposure_contract(manifest_path, bounded)
        if (contract["records"] != protocol["records"] or contract["case"] != protocol["case"]
                or contract["proof"]["status_counts"] != protocol["original_status_counts"]):
            raise ValueError("Frozen exposure scope, case or bound record set changed")
        summary, old, paths = contract["summary"], contract["proof"], contract["paths"]
        graph, adjacency = ObservationGraph(), defaultdict(list)
        graph.check_limit = bounded
        for count, feature in enumerate(iter_features(paths["roads.geojson"], max_record_chars=LIMITS["maximum_feature_chars"]), 1):
            graph.add(feature)
            for edge in feature["properties"]["source_directed_edges"]:
                adjacency[edge["from_node_id"]].append((edge["to_node_id"], edge["edge_id"]))
            if count > LIMITS["maximum_physical_segments"] or len(graph.edges) > LIMITS["maximum_directed_edges"]:
                raise ValueError("Road source feature/edge cap exceeded")
            if count % 1000 == 0:
                bounded()
        if (len(graph.groups) != summary["roads"]["mapped_feature_count"]
                or len(graph.edges) != summary["source_feature_audits"]["roads"]["source_directed_edges"]):
            raise ValueError("Source road graph counts differ from frozen exposure audit")
        points = read_points(paths["settlements.geojson"], bounded)
        destinations = read_points(paths["destinations.geojson"], bounded)
        if set(points) != {r["id"] for r in old["settlements"]} or len(destinations) != summary["access_scenario"]["n_destinations"]:
            raise ValueError("Settlement/destination source point scope differs")
        for row in old["settlements"]:
            if points[row["id"]]["properties"]["name"] != row["name"]:
                raise ValueError("Original settlement name differs from source point")
            selection = row["nearest_baseline_destination"]
            if selection is not None and selection["destination_id"] not in destinations:
                raise ValueError("Fixed target absent from source destination inventory")
        removed, reasons = graph.uncertain(old["uncertain_links"])
        original_removed = {eid for ids in old["uncertain_links"].values() for eid in ids}
        old_witnesses = []
        for row in old["settlements"]:
            selected = row["nearest_baseline_destination"]
            for key in ("path_witness", "optimistic_path_witness"):
                witness = row.get(key)
                if witness is None:
                    continue
                if (selected is None or witness["destination_id"] != selected["destination_id"]
                        or witness["destination_node_id"] != selected["destination_snap_node_id"]
                        or witness["source_node_id"] != row["snap_node_id"]):
                    raise ValueError("Original witness differs from fixed source/target")
                if witness["scenario"] == "pessimistic" and any(e in original_removed for e in witness["path_edge_ids"]):
                    raise ValueError("Original pessimistic witness crosses removed edge")
                old_witnesses.append({"id": row["id"], "name": row["name"], "status": row["status"], "witness_kind": key,
                                      "source_node_id": witness["source_node_id"], "destination_id": witness["destination_id"],
                                      "destination_node_id": witness["destination_node_id"], "support": graph.route_support(witness)})
                bounded()
        bounds = graph.bounds(old, removed)
        independent = verify_extremes(adjacency, graph, old, bounds, original_removed, removed, bounded)
        bbox, context = contract["case"]["bbox"], contract["case"]["context_bbox"]
        def inside(point, box):
            return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
        for row in bounds["settlements"]:
            point = points[row["id"]]["coordinates"]
            row.update(point_coordinates=point, inside_water_aoi=inside(point, bbox), inside_extraction_context=inside(point, context))
        scopes = {name: dict(Counter(r["status"] for r in bounds["settlements"] if predicate(r)))
                  for name, predicate in (("all_source", lambda r: True), ("inside_water_aoi", lambda r: r["inside_water_aoi"]),
                                          ("inside_extraction_context", lambda r: r["inside_extraction_context"]),
                                          ("incidental_outside_context", lambda r: not r["inside_extraction_context"]))}
        write_new(out / "original-witness-support.json", {"schema": "akash-original-witness-support-v1", "case": contract["case"],
                  "witnesses": old_witnesses, "witness_count": len(old_witnesses),
                  "scope": "Every original saved witness, not unique roads, all feasible routes or new optimal paths."})
        write_new(out / "expanded-scenario.json", {"schema": "akash-observation-access-bounds-v1", "case": contract["case"],
                  "scenario_policy": POLICY, "uncertain_physical_groups": reasons,
                  "uncertain_physical_group_count": len(reasons), "uncertain_directed_edge_count": len(removed),
                  "original_uncertain_physical_group_count": len(old["uncertain_links"]), "scope_status_counts": scopes, **bounds,
                  "scope": "Conditional all-open/all-closed graph extrema with original fixed targets; unknown coverage is not observed closure."})
        write_new(out / "independent-extremes.json", independent)
        for item in protocol["records"] + protocol["code"]:
            checked(item, protocol_path.parent)
            bounded()
        receipt = {"schema": RECEIPT_SCHEMA, "status": "completed", "completed_at": now(),
                   "protocol": record(protocol_path), "exposure_manifest": contract["manifest"], "case": contract["case"],
                   "original_status_counts": old["status_counts"], "expanded_status_counts": bounds["status_counts"],
                   "status_transitions": bounds["status_transitions"], "scope_status_counts": scopes,
                   "source_graph_nodes": len(graph.nodes), "source_directed_edges": len(graph.edges), "source_physical_groups": len(graph.groups),
                   "uncertain_physical_group_count": len(reasons), "uncertain_directed_edge_count": len(removed),
                   "original_witness_count": len(old_witnesses), "all_frozen_inputs_unchanged": True,
                   "independent_extremes_passed": True, "zero_length_graph_witness_ids": [r["id"] for r in bounds["settlements"]
                       if (r.get("path_support") or {}).get("projected_centerline_total_m") == 0],
                   "outputs": [record(out / n) for n in ("original-witness-support.json", "expanded-scenario.json", "independent-extremes.json")],
                   "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
                   "peak_process_rss_bytes": rss_bytes(), "claims": CLAIMS, "full_track_b_requirements_satisfied": False,
                   "limitations": ["Saved source hashes and graph consistency are not independent hazard or historical-provider validation.",
                                   "Original nearest target and point associations are reused; nearest-target choice is not independently rerun.",
                                   "Unknown/outside support defines hypothetical open/closed states, never observed physical damage or probability.",
                                   "Retained assessed/nonwater edges are baseline assumptions, not physical passability or facility-operability proof.",
                                   "Baseline disconnection is preserved separately from event-caused isolation; empty route fractions remain null."]}
        write_new(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        failure = {"schema": RECEIPT_SCHEMA, "status": "failed", "failed_at": now(),
                   "protocol_path": str(protocol_path), "error_type": type(error).__name__, "message": str(error),
                   "wall_seconds": time.perf_counter() - started, "peak_process_rss_bytes": rss_bytes(),
                   "no_automatic_retry": True, "claims": CLAIMS, "full_track_b_requirements_satisfied": False}
        write_new(out / "failure.json", failure)
        write_new(out / "receipt.json", failure)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--exposure-manifest", type=Path, required=True)
    freeze_parser.add_argument("--out", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--protocol", type=Path, required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = freeze(args.exposure_manifest, args.out) if args.operation == "freeze" else run(args.protocol, args.out)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
