"""Historical OSM extraction with query provenance, no current-data fallback."""
from datetime import date, datetime, timezone
import hashlib
import json

import requests

from .catalog import utc, validate_bbox

OVERPASS = "https://overpass-api.de/api/interpreter"


def fetch_historical(bbox, snapshot_date, event_date, *, endpoint=OVERPASS):
    validate_bbox(bbox)
    if date.fromisoformat(snapshot_date) >= date.fromisoformat(event_date):
        raise ValueError("OSM snapshot must be strictly before the event date")
    west, south, east, north = bbox
    # Historical snapshot is enforced by the query. The public endpoint rejects
    # out meta in this environment, so preserve the exact body-query provenance.
    cutoff = snapshot_date + "T00:00:00Z"
    extent = f"({south},{west},{north},{east})"
    query = (f'[out:json][timeout:90][date:"{cutoff}"];('
             f'way["highway"]{extent};nwr["building"]{extent};'
             f'nwr["place"~"^(city|town|village|hamlet)$"]{extent};'
             f'nwr["amenity"="hospital"]{extent};);(._;>;);out body;')
    response = requests.get(endpoint, params={"data": query}, timeout=105,
                            headers={"User-Agent": "AkashFloodResearchPrototype/0.1 (historical-map research)"})
    response.raise_for_status()
    data = response.json()
    if data.get("remark"):
        raise ValueError(f"Overpass did not complete cleanly: {data['remark']}")
    elements = data.get("elements", [])
    if not elements:
        raise ValueError("Historical extraction returned no elements")
    for element in elements:
        timestamp = element.get("timestamp")
        if timestamp and utc(timestamp) > utc(cutoff):
            raise ValueError("OSM element is newer than the cutoff")
    provenance = {"snapshot": cutoff, "event_date": event_date, "bbox": list(bbox),
                  "endpoint": endpoint, "query": query,
                  "retrieved_at": datetime.now(timezone.utc).isoformat(),
                  "element_count": len(elements),
                  "history_evidence": "server-executed historical query; per-element edit timestamps not requested",
                  "sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
                  "attribution": "© OpenStreetMap contributors; ODbL"}
    return data, provenance


def named_points(osm):
    """Extract node-tagged destinations/settlements. Polygon POIs need review."""
    settlements, destinations = [], []
    for element in osm.get("elements", []):
        if element.get("type") != "node":
            continue
        tags = element.get("tags", {})
        place = tags.get("place")
        if place not in {"city", "town", "village", "hamlet"} and tags.get("amenity") != "hospital":
            continue
        p = {"id": f"node/{element['id']}", "name": tags.get("name", str(element['id'])),
             "lon": element["lon"], "lat": element["lat"], "kind": place or "hospital"}
        if place in {"village", "hamlet"}:
            settlements.append(p)
        if place in {"city", "town"} or tags.get("amenity") == "hospital":
            destinations.append(p)
    return settlements, destinations
