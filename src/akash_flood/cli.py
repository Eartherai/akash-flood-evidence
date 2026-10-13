import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Akash Flood: inspectable CPU analysis prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("discover", help="Search real Sentinel catalogues; no imagery download")
    discover.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    discover.add_argument("--event-date", required=True)
    discover.add_argument("--out", required=True)
    osm = commands.add_parser("fetch-osm", help="Fetch historical OSM with provenance")
    osm.add_argument("--bbox", nargs=4, type=float, required=True)
    osm.add_argument("--snapshot", required=True)
    osm.add_argument("--event-date", required=True)
    osm.add_argument("--out", required=True)
    run = commands.add_parser("run", help="Analyze already calibrated and aligned dB GeoTIFFs")
    run.add_argument("config")
    run.add_argument("--out", required=True)
    prepare = commands.add_parser("prepare-s1", help="Write openEO radar job graphs; no login or execution")
    prepare.add_argument("catalog")
    prepare.add_argument("--out", required=True)
    terrain = commands.add_parser("fetch-terrain", help="Crop public Copernicus GLO-30 and derive 30 m slope on CPU")
    terrain.add_argument("--bbox", nargs=4, type=float, required=True)
    terrain.add_argument("--out", required=True)
    align = commands.add_parser("align-slope", help="Sample slope onto an existing radar grid")
    align.add_argument("slope")
    align.add_argument("reference")
    align.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "discover":
        from .catalog import discover as search
        result = search(args.event_date, args.bbox)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"S1 scenes: {len(result['sentinel1'])}; S2 scenes: {len(result['sentinel2'])}")
        print(result["selection_error"] or "Matched pre/post S1 pair saved; imagery not yet processed.")
    elif args.command == "fetch-osm":
        from .osm import fetch_historical
        data, provenance = fetch_historical(args.bbox, args.snapshot, args.event_date)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data))
        out.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
        print(f"Saved {len(data['elements'])} historical OSM elements")
    elif args.command == "run":
        from .pipeline import run
        print(json.dumps(run(args.config, args.out), indent=2))
    elif args.command == "prepare-s1":
        from .radar import prepare
        print(json.dumps(prepare(args.catalog, args.out), indent=2))
    elif args.command == "fetch-terrain":
        from .terrain import fetch_terrain
        print(json.dumps(fetch_terrain(args.bbox, args.out), indent=2))
    elif args.command == "align-slope":
        from .terrain import align_slope
        align_slope(args.slope, args.reference, args.out)
        print(f"Aligned slope saved to {args.out}")


if __name__ == "__main__":
    main()
