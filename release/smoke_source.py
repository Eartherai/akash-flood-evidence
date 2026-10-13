"""Run the relocated, base-dependency synthetic CLI and graph-proof smoke only."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--forbid-original-root", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    out = args.out.resolve()
    require(out.is_relative_to(ROOT), "Smoke output must stay inside the extracted source root")
    require(not out.exists(), "Preserve existing smoke; choose a new output path")
    original = args.forbid_original_root.resolve() if args.forbid_original_root else None
    require(original is None or not ROOT.is_relative_to(original), "Extraction must be outside forbidden original root")
    guards = {"network_operations_blocked": 0, "original_reads_blocked": 0,
              "subprocesses_blocked": 0, "model_imports_blocked": 0}

    def audit(event, values):
        if event in {"socket.connect", "socket.sendto", "socket.getaddrinfo", "socket.sendmsg"}:
            guards["network_operations_blocked"] += 1
            raise RuntimeError("Outbound/socket operations prohibited in synthetic smoke")
        if event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"}:
            guards["subprocesses_blocked"] += 1
            raise RuntimeError("Subprocess operations prohibited in synthetic smoke")
        if event == "open" and original is not None and isinstance(values[0], (str, bytes)):
            path = Path(values[0].decode() if isinstance(values[0], bytes) else values[0]).resolve()
            if path.is_relative_to(original):
                guards["original_reads_blocked"] += 1
                raise RuntimeError("Original workspace reads prohibited in relocated smoke")
        if event == "import" and values[0].split(".")[0] in {"sklearn", "torch", "transformers", "joblib"}:
            guards["model_imports_blocked"] += 1
            raise RuntimeError("Optional model stack prohibited in base synthetic smoke")

    sys.addaudithook(audit)
    # Deliberate local probes test guards without contacting a host or reading a file.
    probe = socket.socket()
    try:
        try:
            probe.connect(("127.0.0.1", 9))
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Network guard did not stop the deliberate probe")
    finally:
        probe.close()
    if original is not None:
        try:
            (original / "README.md").read_bytes()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Original-read guard did not stop the deliberate probe")
    probe_guards = guards.copy()

    import akash_flood
    import networkx as nx
    import numpy as np
    import rasterio
    from akash_flood.access import build_road_graph
    from akash_flood.access_bounds import analyze_access_bounds
    from akash_flood.cli import main as cli_main

    source_path = Path(akash_flood.__file__).resolve()
    require(source_path.is_relative_to(ROOT / "src"), "Installed package points outside relocated source")
    out.mkdir(parents=True)
    config_path = load_script("source_release_fixture", ROOT / "scripts/make_fixture.py").create(out / "inputs")
    old_argv = sys.argv[:]
    captured = io.StringIO()
    try:
        sys.argv = ["akash-flood", "run", str(config_path), "--out", str(out / "pipeline")]
        with redirect_stdout(captured):
            cli_main()
    finally:
        sys.argv = old_argv
    metrics = json.loads(captured.getvalue())
    require(metrics["new_water_candidate_pixels"] == 240, "Synthetic new-water count differs")
    require(abs(metrics["candidate_area_km2"] - .024) < 1e-12, "Synthetic area differs")
    require(metrics["newly_disconnected_under_scenario"] == 1, "Scenario must disconnect one fixture settlement")
    require(metrics["mapped_building_features"] == 2 and metrics["building_features_exposed"] == 1,
            "Synthetic polygon association differs")
    access = json.loads((out / "pipeline/settlement_access.json").read_text())
    require([item["status"] for item in access] == ["newly_disconnected", "connected"], "CLI access rows differ")
    manifest = json.loads((out / "pipeline/run_manifest.json").read_text())
    require(manifest["data_kind"] == "synthetic_fixture" and manifest["gpu_used"] is False,
            "Synthetic/no-GPU provenance missing")
    for name in ("report.html", "map.html"):
        require("SYNTHETIC ENGINEERING TEST" in (out / "pipeline" / name).read_text(), "Report lost synthetic label")
    with rasterio.open(out / "pipeline/hazard_candidates.tif") as dataset:
        labels = dataset.read(1)
        require(dataset.shape == (64, 64) and dataset.crs.to_epsg() == 32645 and dataset.nodata == 255,
                "Synthetic native grid differs")
        require(np.count_nonzero(labels == 1) == 240, "Exported candidate cells differ")

    config = json.loads(config_path.read_text())
    graph = build_road_graph(json.loads((out / "inputs/osm.json").read_text()))
    require(len(graph) == 4 and graph.number_of_edges() == 6, "Synthetic road graph differs")
    uncertain = {data["edge_id"] for u, v, _, data in graph.edges(keys=True, data=True)
                 if data["way_id"] == 10 and {u, v} == {2, 3}}
    require(len(uncertain) == 2, "Both physical-link driving directions must be uncertain")
    result = analyze_access_bounds(graph, config["settlements"], config["destinations"],
                                   uncertain_links={"synthetic-middle-link": uncertain}, max_snap_m=25)
    require(result["status_counts"] == {"ambiguous": 1, "robust_connected": 1}, "Bounds must show ambiguity and robust access")
    certificate_counts = load_script("source_release_certificate_checker", ROOT / "scripts/prove_access_bounds.py").verify_certificates(graph, result)
    require(certificate_counts == {"paths": 2, "cuts": 1}, "Expected two path witnesses and one cut")
    # Independently check the actual retained directed graph and exact cut set.
    retained = graph.copy()
    retained.remove_edges_from([(u, v, key) for u, v, key, data in graph.edges(keys=True, data=True)
                               if data["edge_id"] in uncertain])
    require(nx.has_path(graph, 1, 4) and not nx.has_path(retained, 1, 4) and nx.has_path(retained, 3, 4),
            "Independent directed connectivity disagrees with proof")
    require(nx.descendants(retained, 1) | {1} == {1, 2}, "Exact pessimistic cut reachable set differs")
    for item in result["settlements"]:
        witness = item["path_witness"] or item["optimistic_path_witness"]
        require(witness["destination_id"] == "Fixture Hospital" and witness["destination_node_id"] == 4,
                "Synthetic witness target differs")
    (out / "uncertainty-certificate.json").write_text(json.dumps(result, indent=2))
    require(guards == probe_guards, "Synthetic execution attempted a blocked external/model operation")
    records = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
               for path in sorted(out.rglob("*")) if path.is_file()]
    receipt = {"schema": "akash-relocated-source-synthetic-smoke-v1",
               "created_at": datetime.now(timezone.utc).isoformat(), "extracted_source_root": str(ROOT),
               "installed_package_source": str(source_path), "python_version": platform.python_version(),
               "data_kind": "synthetic_fixture", "gpu_used": False,
               "model_inference_run": False, "scientific_accuracy_established": False,
               "network_requests": 0, "network_guard_probe_blocked": probe_guards["network_operations_blocked"] == 1,
               "original_workspace_reads": 0, "original_read_guard_probe_blocked": original is not None,
               "forbidden_original_root": str(original) if original else None,
               "synthetic_metrics": metrics, "graph_nodes": len(graph), "directed_edges": graph.number_of_edges(),
               "uncertain_physical_links": 1, "uncertain_directed_edges": len(uncertain),
               "bound_status_counts": result["status_counts"], "certificate_checks": certificate_counts,
               "exact_pessimistic_cut_nodes": [1, 2], "guards": guards, "outputs": records,
               "wall_seconds": time.perf_counter() - started,
               "scope": "fresh base-dependency source/CLI relocation and synthetic path/cut behavior; no satellite/model/SNAP workflow reproduction"}
    (out / "smoke-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
