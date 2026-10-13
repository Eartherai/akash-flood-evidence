"""One-shot openEO job control. Browser device login; no password in code.

Run with `uv run --extra satellite python scripts/process_s1.py --help`.
No job is created by the default validate action. Start consumes CDSE resources.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import openeo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--scene", choices=["before", "after"], required=True)
    parser.add_argument("--action", choices=["validate", "start", "status", "download"], default="validate")
    args = parser.parse_args()
    request = json.loads((args.prepared/"request.json").read_text())
    graph_path = args.prepared/f"{args.scene}.process.json"
    graph_bytes = graph_path.read_bytes()
    graph = json.loads(graph_bytes)["process_graph"]
    record_path = args.prepared/f"{args.scene}.job.json"
    if args.action == "start" and record_path.exists():
        raise SystemExit("A job is already recorded. Use status or download; duplicate execution refused.")
    if args.action in {"status", "download"} and not record_path.exists():
        raise SystemExit("No recorded job. Start once before requesting status or download.")
    connection = openeo.connect(request["backend"])
    if args.action == "validate":
        validation = connection.validate_process_graph(graph)
        (args.prepared/f"{args.scene}.validation.json").write_text(json.dumps(validation, indent=2))
        print(json.dumps(validation, indent=2))
        print("Validation does not execute processing or verify available source pixels.")
        return
    # Tokens live only in this process. Repeat browser login on a later invocation.
    connection.authenticate_oidc_device(store_refresh_token=False)
    if args.action == "start":
        job = connection.create_job(graph, title=f"Akash pilot {args.scene}",
                                    description="VV linear sigma0, Copernicus DEM 30m, 10m grid. Source review pending.")
        record = {"job_id": job.job_id, "backend": request["backend"],
                  "graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
                  "created_at": datetime.now(timezone.utc).isoformat()}
        # Save identity before start: a transient start failure must not spawn duplicates.
        record_path.write_text(json.dumps(record, indent=2))
        job.start()
        print(f"Started job {job.job_id}. Run --action status later.")
        return
    record = json.loads(record_path.read_text())
    if record["graph_sha256"] != hashlib.sha256(graph_bytes).hexdigest() or record["backend"] != request["backend"]:
        raise SystemExit("Job provenance differs from prepared graph or backend")
    job = connection.job(record["job_id"])
    metadata = job.describe()
    (args.prepared/f"{args.scene}.status.json").write_text(json.dumps(metadata, indent=2))
    print("Job status:", metadata["status"])
    if args.action == "download":
        if metadata["status"] != "finished":
            raise SystemExit("Job has not finished; no outputs downloaded")
        target = args.prepared/f"{args.scene}-results"
        if target.exists() and any(target.iterdir()):
            raise SystemExit("Result directory is nonempty; refusing to overwrite evidence")
        target.mkdir(exist_ok=True)
        job.get_results().download_files(target, include_stac_metadata=True)
        print(f"Saved outputs and STAC metadata in {target}. Source identity and terrain checks remain required.")


if __name__ == "__main__":
    main()
