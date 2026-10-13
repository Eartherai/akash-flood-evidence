"""Freeze and execute a reference-free, saved-road observation-support audit.

No model, image pixel, damage-reference geometry or network input is loaded.
Existing independently checked saved road and fixed-target boundaries are reused.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

from akash_flood.observation_access import EPSILON, ObservationGraph, iter_features

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "rasuwa": "outputs/fresh-one-command-replay/exposure",
    "kathmandu": "outputs/kathmandu-node-drive-access-2026-10-08",
}
SCIENTIFIC = ["outputs/sen1-cpu-v1/water-model.joblib", "outputs/sen1-cpu-v1/model-evaluation-freeze.json",
              "outputs/sen1-cpu-v1/final_test-evaluation.json", "research/sen1-cpu-training-protocol.json"]
CODE = ["scripts/audit_observation_access.py", "src/akash_flood/observation_access.py", "tests/test_observation_access.py",
        "src/akash_flood/access.py", "src/akash_flood/access_bounds.py", "src/akash_flood/nearest_access.py"]
REVIEWS = ["research/fresh-exposure-report-review.md", "research/kathmandu-node-drive-access-audit-2026-10-08.json",
           "research/kathmandu-core-access-review-2026-10-08.md"]


def now(): return datetime.now(timezone.utc).isoformat()


def record(path):
    path = Path(path).resolve()
    with path.open("rb") as stream: digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def verify(records):
    for item in records:
        actual = record(ROOT / item["path"])
        if actual != item: raise ValueError("Changed frozen input: " + item["path"])


def freeze(out):
    paths = set(SCIENTIFIC + CODE + REVIEWS)
    case_records = {}
    for case, folder in CASES.items():
        base = ROOT / folder
        binding = json.loads((base / "source-and-output-manifest.json").read_text())
        # Metadata is already public/known. The road records and new reachability
        # outcomes are not read while establishing this new analysis contract.
        for section in ("inputs", "code"):
            for item in binding[section]:
                actual = record(ROOT / item["path"])
                if actual["sha256"] != item["sha256"] or actual["bytes"] != item["bytes"]:
                    raise ValueError("Existing source boundary differs: " + item["path"])
                paths.add(str(Path(item["path"]).resolve().relative_to(ROOT)))
        for name in ("source-and-output-manifest.json", "summary.json", "access-proof.json", "scenario-policy.json", "roads.geojson", "settlements.geojson", "destinations.geojson"):
            paths.add(str((base / name).relative_to(ROOT)))
        for name in ("roads.geojson", "access-proof.json", "summary.json", "settlements.geojson", "destinations.geojson"):
            item = binding["outputs"][name]
            actual = record(ROOT / item["path"])
            if actual["sha256"] != item["sha256"] or actual["bytes"] != item["bytes"]:
                raise ValueError("Existing saved output binding differs: " + name)
        case_records[case] = {"folder": folder, "case": binding["case"],
                              "source_boundary": "Existing independently audited hash-bound saved physical road records and baseline fixed-target policy; no new independent OSM authenticity or physical-passability proof."}
    protocol = {"schema": "akash-observation-access-protocol-v1", "status": "frozen", "frozen_at": now(),
                "cases": case_records, "records": [record(ROOT / p) for p in sorted(paths)],
                "previous_outcomes_known": True, "new_scenario_outcomes_read_before_freeze": False,
                "blind_hazard_validation": False, "scope": "Reference-free observation-support audit and distinct conditional graph scenario on two already known cases.",
                "policy": {"candidate_groups": "Preserve every original new-water candidate-selected physical group.",
                           "additional_uncertainty": "Every source physical segment with >1e-6 m2 unassessed-within-AOI or outside-AOI support in the saved 20 m corridor is jointly open/closed in all its source directions.",
                           "positive_area_tolerance_m2": EPSILON, "certain_closures": [],
                           "destination": "Preserve each original fixed nearest baseline-reachable target and point associations; never switch.",
                           "observed_nonwater_is_passable": False, "unknown_is_observed_closed": False,
                           "all_combinations_allowed": True, "inspection_ranking": False,
                           "witness_support": "Sum per-directed-traversal saved projected centreline lengths; retain assessed, unassessed-within-AOI and outside-AOI denominator. Great-circle graph distance is separate."},
                "limits": {"maximum_wall_seconds_per_case": 300, "maximum_process_rss_bytes": 2000000000,
                           "max_feature_chars": 2000000, "maximum_physical_segments": 300000},
                "claims": {"observed_closures": False, "safe_routes": False, "local_hazard_accuracy": False,
                           "model_inference": False, "training": False, "network": False, "gpu": False,
                           "reference_geometry_loaded": False},
                "failure_policy": "Preserve every failed attempt and old output. Never adjust scenario or choose subsets after outcomes. Check memory/wall ceilings during bounded streaming and target reachability."}
    write_new(out, protocol)
    return record(out)


def rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def run(protocol_path, case, out):
    protocol_path, out = Path(protocol_path).resolve(), Path(out).resolve()
    if out.exists() or out.is_symlink(): raise FileExistsError("Preserve prior attempts; use a fresh output path")
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "frozen" or protocol["policy"]["positive_area_tolerance_m2"] != EPSILON:
        raise ValueError("Require the frozen declared observation scenario")
    # Bind the complete two-case contract before reading/scoring either case.
    verify(protocol["records"])
    out.mkdir(parents=True)
    started = time.perf_counter()
    cpu_started = time.process_time()
    write_new(out / "started.json", {"started_at": now(), "protocol": record(protocol_path), "case": case,
                                     "limits": protocol["limits"]})
    def bounded():
        if time.perf_counter() - started > protocol["limits"]["maximum_wall_seconds_per_case"]:
            raise TimeoutError("Frozen 300 second analysis ceiling exceeded")
        if rss_bytes() > protocol["limits"]["maximum_process_rss_bytes"]:
            raise MemoryError("Frozen 2 GB process RSS ceiling exceeded")
    try:
        base = ROOT / protocol["cases"][case]["folder"]
        summary = json.loads((base / "summary.json").read_text())
        old = json.loads((base / "access-proof.json").read_text())
        if old["case"] != protocol["cases"][case]["case"] or old["water_source"] != summary["water_source"]:
            raise ValueError("Saved case/source mismatch")
        graph = ObservationGraph()
        graph.check_limit = bounded
        for count, feature in enumerate(iter_features(base / "roads.geojson", max_record_chars=protocol["limits"]["max_feature_chars"]), 1):
            graph.add(feature)
            if count % 1000 == 0: bounded()
            if count > protocol["limits"]["maximum_physical_segments"]:
                raise ValueError("Physical segment cap exceeded")
        if len(graph.groups) != summary["roads"]["mapped_feature_count"] or len(graph.edges) != summary["source_feature_audits"]["roads"]["source_directed_edges"]:
            raise ValueError("Compact graph counts differ from frozen source audit")
        removed, reasons = graph.uncertain(old["uncertain_links"])
        original_removed = {eid for ids in old["uncertain_links"].values() for eid in ids}
        old_witnesses = []
        for row in old["settlements"]:
            selected = row["nearest_baseline_destination"]
            for key in ("path_witness", "optimistic_path_witness"):
                witness = row[key]
                if witness:
                    if selected is None or witness["destination_id"] != selected["destination_id"] or witness["destination_node_id"] != selected["destination_snap_node_id"]:
                        raise ValueError("Original witness changed fixed target")
                    if witness["scenario"] == "pessimistic" and any(e in original_removed for e in witness["path_edge_ids"]):
                        raise ValueError("Original pessimistic witness crosses original removed edge")
                    old_witnesses.append({"id": row["id"], "name": row["name"], "status": row["status"], "witness_kind": key,
                                          "source_node_id": witness["source_node_id"], "destination_id": witness["destination_id"],
                                          "destination_node_id": witness["destination_node_id"], "support": graph.route_support(witness)})
        bounded()
        # No per-link counterfactual rankings; one linear membership index and
        # reverse BFS per original fixed target. Existing target selection is frozen.
        bounds = graph.bounds(old, removed)
        bounded()
        points = {f["id"]: f["geometry"]["coordinates"] for f in iter_features(base / "settlements.geojson")}
        bbox, context = summary["case"]["bbox"], summary["case"]["context_bbox"]
        def inside(point, box): return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
        if set(points) != {r["id"] for r in old["settlements"]}: raise ValueError("Source point scope differs")
        for row in bounds["settlements"]:
            row["point_coordinates"] = points[row["id"]]
            row["inside_water_aoi"] = inside(points[row["id"]], bbox)
            row["inside_extraction_context"] = inside(points[row["id"]], context)
        scopes = {name: dict(Counter(r["status"] for r in bounds["settlements"] if predicate(r)))
                  for name, predicate in (("all_source", lambda r: True), ("inside_water_aoi", lambda r:r["inside_water_aoi"]),
                                          ("inside_extraction_context", lambda r:r["inside_extraction_context"]),
                                          ("incidental_outside_context", lambda r:not r["inside_extraction_context"]))}
        write_new(out / "original-witness-support.json", {"schema": "akash-original-witness-support-v1", "case": summary["case"],
                  "witnesses": old_witnesses, "witness_count": len(old_witnesses),
                  "all_have_some_unsupported_length": all(w["support"]["lengths_m"]["unknown_within_aoi"] + w["support"]["lengths_m"]["outside_aoi"] > EPSILON for w in old_witnesses),
                  "scope": "All original saved witnesses, not all feasible routes or a newly optimized road path."})
        write_new(out / "expanded-scenario.json", {"schema": "akash-observation-access-bounds-v1", "case": summary["case"],
                  "scenario_policy": protocol["policy"], "uncertain_physical_groups": reasons,
                  "uncertain_physical_group_count": len(reasons), "uncertain_directed_edge_count": len(removed),
                  "original_uncertain_physical_group_count": len(old["uncertain_links"]), "scope_status_counts": scopes, **bounds,
                  "scope": "Every allowed uncertain-link assignment is bounded by all-open/all-closed extrema; mathematical graph claim conditional on remaining baseline edges, original fixed targets and observed-support scenario."})
        # Rehash all frozen inputs after analysis; no input is rewritten.
        verify(protocol["records"])
        result = {"schema": "akash-observation-access-receipt-v1", "status": "completed", "finished_at": now(),
                  "protocol": record(protocol_path), "case": summary["case"], "original_status_counts": old["status_counts"],
                  "expanded_status_counts": bounds["status_counts"], "status_transitions": bounds["status_transitions"],
                  "scope_status_counts": scopes, "original_witness_count": len(old_witnesses),
                  "source_graph_nodes": len(graph.nodes), "source_directed_edges": len(graph.edges), "source_physical_groups": len(graph.groups),
                  "uncertain_physical_group_count": len(reasons), "uncertain_directed_edge_count": len(removed),
                  "new_uncertain_physical_group_count": len(reasons) - len(old["uncertain_links"]),
                  "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
                  "peak_process_rss_bytes": rss_bytes(), "all_frozen_inputs_unchanged": True,
                  "outputs": [record(out / "original-witness-support.json"), record(out / "expanded-scenario.json")],
                  "claims": protocol["claims"], "limitations": ["Source-bound graph/coverage analysis, not independent hazard validation.",
                  "Unknown/outside support is an explicit open/closed scenario, not physical observed damage or a failure probability.",
                  "Assessed nonwater does not establish physical passability; other unmapped and unspecified failures remain excluded.",
                  "Original source-to-OSM and nearest-target validation receipts are reused, not independently rerun.",
                  "Projected saved centreline support lengths differ in denominator from original great-circle graph distance."]}
        write_new(out / "receipt.json", result)
        return {k: result[k] for k in ("status", "original_status_counts", "expanded_status_counts", "status_transitions", "scope_status_counts", "uncertain_physical_group_count", "wall_seconds", "peak_process_rss_bytes")}
    except BaseException as exc:
        write_new(out / "failure.json", {"status": "failed", "recorded_at": now(), "exception": type(exc).__name__, "error": str(exc),
                  "wall_seconds": time.perf_counter() - started, "peak_process_rss_bytes": rss_bytes(), "old_artifacts_overwritten": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze"); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("run"); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--case", choices=CASES, required=True); p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.out) if args.command == "freeze" else run(args.protocol, args.case, args.out)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
