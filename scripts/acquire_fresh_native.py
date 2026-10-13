"""Acquire a selected fresh raw Sentinel-1 pair, or inspect it offline by default.

Examples:
  uv run python scripts/acquire_fresh_native.py --plan outputs/fresh-case/plan.json
  uv run python scripts/acquire_fresh_native.py --before-item before.stac.json \
    --after-item after.stac.json
  # After reviewing the offline plan and explicit byte budget:
  uv run python scripts/acquire_fresh_native.py --plan outputs/fresh-case/plan.json \
    --execute --output-dir outputs/fresh-native --cache-dir data/fresh-native-cache

A local proof replay may opt into --source-cache data/s1-snap-native. Every old
source binding is verified, and no new HTTPS download is claimed for imported
bytes. The original manifest is retained separately from the derived wrapper.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from akash_flood.fresh_native import (
    SCHEMA, DEFAULT_BUDGET, DEFAULT_MEASUREMENT_CAP, METADATA_CAP,
    acquire, digest, inspect_existing_native_cache, json_write, scenes_from_plan,
    validate_pair, validate_scene, positive_int,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, help="Saved fresh planner plan.json, exact selected-STAC checksums required")
    parser.add_argument("--before-item", type=Path)
    parser.add_argument("--after-item", type=Path)
    parser.add_argument("--before-sha256")
    parser.add_argument("--after-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--source-cache", type=Path, help="Explicit verified old native-cache import; no new remote-state claim")
    parser.add_argument("--network-budget-bytes", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--measurement-cap-bytes", type=int, default=DEFAULT_MEASUREMENT_CAP)
    parser.add_argument("--dry-run-output", type=Path, help="Optional new JSON path for offline inspection result")
    parser.add_argument("--execute", action="store_true", help="Required for acquisition/local import; default makes zero network requests")
    args = parser.parse_args(argv)
    positive_int(args.network_budget_bytes, "network budget")
    positive_int(args.measurement_cap_bytes, "measurement cap", DEFAULT_MEASUREMENT_CAP)
    if args.plan:
        if args.before_item or args.after_item or args.before_sha256 or args.after_sha256:
            parser.error("Use either a frozen planner manifest or explicit saved STAC items")
        scenes, source_plan = scenes_from_plan(args.plan)
    else:
        if not args.before_item or not args.after_item:
            parser.error("Supply --plan or both --before-item and --after-item")
        scenes = [validate_scene("before", args.before_item, args.before_sha256), validate_scene("after", args.after_item, args.after_sha256)]
        validate_pair(*scenes)
        source_plan = None
    # Explicit cache replay is checksum-verified offline even during dry-run.
    imports = inspect_existing_native_cache(scenes, args.source_cache) if args.source_cache else None
    if not args.execute:
        result = {"schema": SCHEMA, "status": "offline_inputs_checked_not_acquired", "source_plan": source_plan, "input_stac": {scene.name: scene.record() for scene in scenes}, "network_requests": 0, "network_received_bytes": 0, "network_budget_bytes": args.network_budget_bytes, "measurement_individual_cap_bytes": args.measurement_cap_bytes, "metadata_individual_cap_bytes": METADATA_CAP, "explicit_source_cache_import": {"path": str(args.source_cache.resolve()), "index_sha256": digest(args.source_cache / "derived-wrapper-index.json"), "assets_checked": len(imports)} if imports else None, "source_scope": "selected native Sentinel-1 VV/VH and original acquisition/calibration/noise metadata only; no labels/damage used", "limitations": ["No public source availability, remote size/ETag or newly downloaded bytes were checked by this dry-run.", "Cached source verification proves local SHA/MD5/size and selected-item binding, not a new remote download or current provider object identity.", "A derived SAFE wrapper is not an original ESA SAFE; source data are not calibrated or terrain-corrected here."]}
        if args.dry_run_output:
            json_write(args.dry_run_output, result)
    else:
        if not args.output_dir or not args.cache_dir:
            parser.error("--execute requires separate explicit --output-dir and --cache-dir")
        result = acquire(scenes, args.output_dir, args.cache_dir, budget_bytes=args.network_budget_bytes, measurement_cap=args.measurement_cap_bytes, source_plan=source_plan, imported_sources=imports)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
