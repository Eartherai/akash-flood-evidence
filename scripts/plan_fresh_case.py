"""Plan a new area/date from saved catalogue data, without downloads or inference."""
import argparse
import json
from pathlib import Path

from akash_flood.fresh_case import save_plan

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--event-date", required=True, help="YYYY-MM-DD; whole event day excluded from pair selection")
    parser.add_argument("--osm-snapshot", required=True, help="Pre-event date or explicit UTC timestamp")
    parser.add_argument("--catalog", type=Path, required=True, help="Saved discover JSON or STAC FeatureCollection")
    parser.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--max-days", type=int, default=24)
    parser.add_argument("--context-buffer-deg", type=float, default=.03)
    parser.add_argument("--context-bbox", type=float, nargs=4, help="Optional explicit processing context, must cover AOI")
    parser.add_argument("--s2-coverage", dest="coverage_path", type=Path,
                        help="Optional {bbox,scenes:[{item_id,scl_path,sha256}]} local SCL record")
    parser.add_argument("--osm", dest="osm_path", type=Path, help="Optional saved historic extract; requires --osm-provenance")
    parser.add_argument("--osm-provenance", type=Path)
    parser.add_argument("--requirements", dest="requirements_path", type=Path,
                        default=ROOT.parent / "research/source-extracts/Track B.txt")
    args = vars(parser.parse_args())
    catalog, out = args.pop("catalog"), args.pop("out")
    plan = save_plan(catalog, out, **args)
    selected = plan["sentinel1"]["selected_pair"]
    print(json.dumps({"case_id": plan["case"]["case_id"], "status": plan["status"], "input_plan_status": plan["input_plan_status"],
                      "blockers": plan["blockers"], "epsg": plan["projection"]["epsg"],
                      "DEM_tiles": plan["dem"]["required_tile_count"],
                      "selected_item_ids": {key: selected[key]["item_id"] for key in ("before", "after")} if selected else None,
                      "optical_status": plan["sentinel2"]["status"], "plan_path": str((Path(out) / "plan.json").resolve()),
                      "runnable_end_to_end_verified": False}, indent=2))


if __name__ == "__main__":
    main()
