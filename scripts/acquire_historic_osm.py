"""Acquire one bounded historical map from an explicitly selected public server.

This is a separate attempt, never an automatic endpoint/current-map fallback.
The original fresh-case runner and its saved source records remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import requests

from akash_flood.catalog import validate_bbox, utc
from akash_flood.fresh_case import event_boundary, normalized_utc, osm_plan

ENDPOINTS = {
    "main": "https://overpass-api.de/api/interpreter",
    "private-coffee": "https://overpass.private.coffee/api/interpreter",
}
MAX_BYTES = 30_000_000
DEADLINE_SECONDS = 120


def now():
    return datetime.now(timezone.utc).isoformat()


def record(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate field in historical map JSON")
        result[key] = value
    return result


def wall_deadline_handler(signum, frame):
    raise TimeoutError("Historical acquisition wall deadline exceeded")


def validate_elements(data, cutoff):
    seen = set()
    def identity(value):
        return isinstance(value,int) and not isinstance(value,bool) and value > 0
    for element in data['elements']:
        if not isinstance(element,dict) or element.get('type') not in ('node','way','relation') or not identity(element.get('id')):
            raise ValueError('Historical elements require a supported type and positive integer ID')
        key = (element['type'],element['id'])
        if key in seen:
            raise ValueError('Historical extract contains a duplicate object identity')
        seen.add(key)
        tags = element.get('tags',{})
        if not isinstance(tags,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in tags.items()):
            raise ValueError('Historical object tags must be string pairs')
        if element['type'] == 'node':
            for name,limit in [('lon',180),('lat',90)]:
                value = element.get(name)
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or abs(value)>limit:
                    raise ValueError('Historical node needs finite in-range coordinates')
        elif element['type'] == 'way':
            nodes = element.get('nodes')
            if not isinstance(nodes,list) or len(nodes)<2 or any(not identity(n) for n in nodes):
                raise ValueError('Historical way needs valid node references')
        else:
            members = element.get('members')
            if not isinstance(members,list) or any(not isinstance(m,dict) or m.get('type') not in ('node','way','relation') or not identity(m.get('ref')) or not isinstance(m.get('role',''),str) for m in members):
                raise ValueError('Historical relation needs valid member references')
        if element.get('timestamp') and utc(element['timestamp']) > cutoff:
            raise ValueError('Historical extract contains an edit after the cutoff')


def acquire(bbox, snapshot, event_date, out, *, server="private-coffee", execute=False, session=None):
    bbox = validate_bbox(bbox)
    if (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) > 1:
        raise ValueError("Historical map extent exceeds one square degree")
    _, _, cutoff = event_boundary(event_date, snapshot)
    if cutoff.time() != datetime.min.time():
        raise ValueError("This acquisition supports a date or UTC-midnight cutoff")
    endpoint = ENDPOINTS[server]
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError("Preserve previous attempts; use a new directory")
    query = osm_plan(cutoff, bbox)["query"]
    out.mkdir(parents=True)
    protocol = {
        "schema": "akash-separate-historic-map-protocol-v1",
        "recorded_before_network_at": now(), "bbox": list(bbox),
        "snapshot": normalized_utc(cutoff), "event_date": event_date,
        "endpoint": endpoint, "query": query, "code": record(__file__),
        "maximum_requests": 1, "maximum_body_bytes": MAX_BYTES,
        "stream_budget_stop_granularity_bytes": 65_536,
        "deadline_seconds": DEADLINE_SECONDS, "execute_requested": execute,
        "tls_verified": True, "redirects_followed": False,
        "automatic_retry_or_current_map_fallback": False,
        "source_scope": "Historical server-executed body query; edit timestamps are not requested; independent server authenticity/completeness is not established",
        "endpoint_source": "https://wiki.openstreetmap.org/wiki/Overpass_API",
        "reference_labels_loaded": False, "model_inference_run": False,
    }
    write_json(out / "protocol.json", protocol)
    if not execute:
        return {"status": "protocol_only", "network_requests": 0, "protocol": record(out / "protocol.json")}
    client = session or requests.Session()
    client.trust_env = False
    if getattr(client, "auth", None) or any(
        key.casefold() in {"authorization", "proxy-authorization"}
        for key in getattr(client, "headers", {})
    ):
        raise ValueError("Historical acquisition accepts no credentials")
    receipt = {
        "schema": "akash-separate-historic-map-transport-v1",
        "status": "running", "started_at": now(), "protocol": record(out / "protocol.json"),
        "attempted_requests": 0, "received_body_bytes": 0,
        "response_status": None, "tls_verified": True, "redirects_followed": False,
        "endpoint": endpoint, "query": query,
    }
    started = time.monotonic()
    response = None
    partial = out / "received-response.partial.bin"
    previous_handler = None
    try:
        if not hasattr(signal, "setitimer") or signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise RuntimeError("Acquisition requires an available process wall timer")
        previous_handler = signal.signal(signal.SIGALRM, wall_deadline_handler)
        signal.setitimer(signal.ITIMER_REAL, DEADLINE_SECONDS)
        receipt["attempted_requests"] = 1
        response = client.get(endpoint, params={"data": query}, stream=True,
            timeout=(15, 105), allow_redirects=False, verify=True,
            headers={"User-Agent": "AkashHistoricalMap/1.0", "Accept-Encoding": "identity"})
        receipt["response_status"] = response.status_code
        if response.status_code != 200:
            raise ValueError(f"Historical server returned HTTP {response.status_code}")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise ValueError("Historical response requires identity encoding for byte accounting")
        advertised = response.headers.get("Content-Length")
        if advertised and (not advertised.isdigit() or int(advertised) > MAX_BYTES):
            raise ValueError("Historical response advertised an invalid or excessive size")
        digest = hashlib.sha256()
        with partial.open("xb") as stream:
            for chunk in response.iter_content(65_536):
                if time.monotonic()-started > DEADLINE_SECONDS:
                    raise TimeoutError("Historical response wall deadline exceeded")
                if chunk:
                    receipt["received_body_bytes"] += len(chunk)
                    if receipt["received_body_bytes"] > MAX_BYTES:
                        raise ValueError("Historical response byte cap exceeded")
                    stream.write(chunk)
                    digest.update(chunk)
        if advertised and int(advertised) != receipt["received_body_bytes"]:
            raise ValueError("Historical response is incomplete")
        if time.monotonic()-started > DEADLINE_SECONDS:
            raise TimeoutError("Historical response wall deadline exceeded")
        data = json.loads(partial.read_bytes(), object_pairs_hook=unique_object)
        if (not isinstance(data, dict) or data.get("remark")
                or not isinstance(data.get("elements"), list) or not data["elements"]):
            raise ValueError("Historical extract is empty or server reported incomplete processing")
        validate_elements(data,cutoff)
        raw_path = out / "response.json"
        partial.rename(raw_path)
        write_json(out / "osm.json", data)
        osm_record = record(out / "osm.json")
        provenance = {
            "snapshot": normalized_utc(cutoff), "event_date": event_date,
            "bbox": list(bbox), "endpoint": endpoint, "query": query,
            "retrieved_at": now(), "element_count": len(data["elements"]),
            "history_evidence": protocol["source_scope"], "sha256": osm_record["sha256"],
            "raw_response": record(raw_path), "attribution": "© OpenStreetMap contributors; ODbL",
            "acquisition_protocol": record(out / "protocol.json"),
        }
        write_json(out / "provenance.json", provenance)
        receipt.update(status="completed", osm=osm_record,
            provenance=record(out / "provenance.json"), raw_response=record(raw_path),
            raw_response_sha256=digest.hexdigest(), element_count=len(data["elements"]))
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        if partial.exists():
            receipt["retained_partial_response"] = record(partial)
        raise
    finally:
        if previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
        if response is not None:
            response.close()
        receipt.update(completed_at=now(), elapsed_seconds=time.monotonic()-started)
        write_json(out / "network-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", type=float, nargs=4, required=True)
    parser.add_argument("--event-date", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--server", choices=ENDPOINTS, default="private-coffee")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(acquire(args.bbox, args.snapshot, args.event_date, args.out,
        server=args.server, execute=args.execute), indent=2))


if __name__ == "__main__":
    main()
