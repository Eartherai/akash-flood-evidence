"""Optional observation-aware source clauses over the unchanged two-case service.

Without --observation-audit-root, use the original service and original facts.
With it, require both completed audits and display separately declared scenarios.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import resource
import time
from http.server import ThreadingHTTPServer

from akash_flood.copilot import MODEL
from akash_flood.multi_case_copilot import default_cases
from akash_flood.observation_copilot import ObservationCaseFactStore, observation_cases

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("akash_original_multi_case_server", ROOT / "scripts/serve_multi_case_copilot.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class ObservationCaseService(_base.MultiCaseService):
    def __init__(self, endpoint="http://127.0.0.1:8782", cases=None, *, observation_audit_root=None, source_root=ROOT):
        started = time.perf_counter()
        self.observation_enabled = observation_audit_root is not None
        original_cases = cases or default_cases(source_root)
        configs = observation_cases(observation_audit_root, source_root=source_root, cases=original_cases) if self.observation_enabled else original_cases
        super().__init__(endpoint, original_cases)
        if self.observation_enabled:
            self.stores = {slug: ObservationCaseFactStore.from_verified_store(store, configs[slug]) for slug, store in self.stores.items()}
            self.html_path = ROOT / "web/observation-copilot.html"
            self.html = self.html_path.read_bytes()
            helpers = next(iter(self.stores.values()))._helpers
            self.code_bindings += [helpers.record(path) for path in (
                self.html_path, Path(__file__), ROOT / "src/akash_flood/observation_copilot.py")]
            self.activation_wall_seconds = time.perf_counter() - started

    def metadata(self):
        result = super().metadata()
        if self.observation_enabled:
            for case in result["cases"]:
                case["facts"] = [fact for fact in case["facts"] if not fact["id"].startswith("newscenario_settlement_")]
                case["observation_receipt"] = str(self.stores[case["slug"]].observation_receipt_path)
            result["observation_extension"] = {
                "status": "both completed source-bound audits active",
                "analysis": "Saved conditional graph scenarios and observation support; no new graph solve or hazard inference during questions",
                "original_scenario": "Candidate-only uncertainty; retained unknown/outside-AOI roads",
                "expanded_scenario": "Candidate-selected plus unknown-within-AOI/outside-AOI corridor support uncertain; all open/closed combinations, zero certain closures",
                "scope": "Whole source inventory and inside-water-AOI counts are distinct",
                "empty_graph_path": "Zero-length route fractions remain null; point/facility connection unvalidated",
                "validation": "Development software/model checks; no human usability or Nepali review, no operational passability validation",
            }
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8802)
    parser.add_argument("--model-endpoint", default="http://127.0.0.1:8782")
    parser.add_argument("--observation-audit-root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535: parser.error("Use an unprivileged loopback port")
    if args.out and args.out.exists(): raise FileExistsError("Preserve prior activations; choose a new output directory")
    service = ObservationCaseService(args.model_endpoint, observation_audit_root=args.observation_audit_root)
    if args.out:
        args.out.mkdir(parents=True)
        (args.out / "index.html").write_bytes(service.html)
        (args.out / "facts.json").write_text(json.dumps(service.metadata(), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        (args.out / "activation.json").write_text(json.dumps({
            "schema": "akash-observation-copilot-activation-v1", "created_at": service.started_at,
            "observation_enabled": service.observation_enabled, "code": service.code_bindings,
            "activation_wall_seconds": service.activation_wall_seconds,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "case_source_bindings": {slug: store.source_bindings for slug, store in service.stores.items()},
            "model": MODEL, "model_endpoint": args.model_endpoint,
            "cpu_only_launcher": "scripts/run_copilot_model.py --serve",
            "model_invoked_by_activation": False, "new_graph_solve_by_extension": False,
            "hazard_inference_run": False, "reference_labels_loaded": False,
            "validation": "Development checks; AI-authored Nepali, no human language or usability review",
        }, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"url": f"http://127.0.0.1:{args.port}/", "cases": list(service.stores), "observation_enabled": service.observation_enabled, "model_invoked": False}), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), _base.handler_for(service)).serve_forever()


if __name__ == "__main__": main()
