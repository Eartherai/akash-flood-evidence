"""Public catalogue discovery; never substitutes unrelated scenes on failure."""
from datetime import date, datetime, time, timedelta, timezone
import math

import requests
from shapely.geometry import box, shape

CDSE_STAC = "https://stac.dataspace.copernicus.eu/v1"


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Acquisition timestamps must include timezone")
    return parsed.astimezone(timezone.utc)


def validate_bbox(bbox):
    if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        raise ValueError("Bounding box must contain four finite coordinates")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Use west,south,east,north; dateline-crossing boxes need splitting")
    return list(bbox)


def search(collection, bbox, start, end, *, session=None, max_pages=20):
    """Follow pagination; fail explicitly when the bounded search is incomplete."""
    validate_bbox(bbox)
    client = session or requests.Session()
    url = f"{CDSE_STAC}/search"
    params = {"collections": collection, "bbox": ",".join(map(str, bbox)),
              "datetime": f"{start}/{end}", "limit": 100}
    features, seen = [], set()
    for _ in range(max_pages):
        response = client.get(url, params=params, timeout=45)
        response.raise_for_status()
        data = response.json()
        for feature in data.get("features", []):
            if feature["id"] not in seen:
                features.append(feature)
                seen.add(feature["id"])
        next_link = next((link for link in data.get("links", []) if link.get("rel") == "next"), None)
        if not next_link:
            return features
        if next_link.get("method", "GET") != "GET":
            raise ValueError("Catalogue returned unsupported pagination method; search incomplete")
        url, params = next_link["href"], None
    raise ValueError("Catalogue page limit reached; reduce area/date window")


def select_s1_pair(features, event_date, bbox, *, max_days=24):
    """Choose nearest valid post-event scene, then nearest matching pre-scene.

    Excludes the whole event day when event time is unknown. Both footprints must
    cover the AOI; larger AOIs requiring mosaics fail instead of returning a
    partial result. Orbit, pass, mode and polarization sets must be present/match.
    """
    event = date.fromisoformat(event_date)
    event_start = datetime.combine(event, time(), tzinfo=timezone.utc)
    event_end = event_start + timedelta(days=1)
    aoi = box(*validate_bbox(bbox))
    pre, post = [], []
    for item in features:
        p = item.get("properties", {})
        keys = ("sat:relative_orbit", "sat:orbit_state", "sar:instrument_mode", "sar:polarizations")
        if any(p.get(key) is None for key in keys) or "VV" not in p["sar:polarizations"]:
            continue
        if not item.get("geometry") or not shape(item["geometry"]).covers(aoi):
            continue
        acquired = utc(p["datetime"])
        if event_start - timedelta(days=max_days) <= acquired < event_start:
            pre.append(item)
        elif event_end <= acquired <= event_end + timedelta(days=max_days):
            post.append(item)
    pairs = []
    for before in pre:
        for after in post:
            a, b = before["properties"], after["properties"]
            if any(a[k] != b[k] for k in keys[:-1]):
                continue
            if set(a[keys[-1]]) != set(b[keys[-1]]):
                continue
            separation = (utc(b["datetime"]) - utc(a["datetime"])).total_seconds() / 86400
            # Accept repeated geometry with 6/12-day cycles, allowing timing jitter.
            if min(abs(separation - cycle) for cycle in (6, 12, 18, 24)) > 0.1:
                continue
            pairs.append((utc(b["datetime"]), -utc(a["datetime"]).timestamp(), before, after))
    if not pairs:
        raise ValueError("No complete-footprint, same-orbit pre/post pair in the requested window")
    _, _, before, after = min(pairs, key=lambda value: value[:2])
    return {"before": before, "after": after,
            "selection": "earliest post-event, then latest compatible pre-event; event day excluded",
            "processing_status": "catalogued_only; calibration and terrain correction required"}


def discover(event_date, bbox, *, max_days=24):
    event = date.fromisoformat(event_date)
    start = (event - timedelta(days=max_days)).isoformat() + "T00:00:00Z"
    end = (event + timedelta(days=max_days + 1)).isoformat() + "T00:00:00Z"
    s1 = search("sentinel-1-grd", bbox, start, end)
    s2 = search("sentinel-2-l2a", bbox, start, end)
    try:
        pair, error = select_s1_pair(s1, event_date, bbox, max_days=max_days), None
    except ValueError as exc:
        pair, error = None, str(exc)
    return {"event_date": event_date, "bbox": list(bbox), "source": CDSE_STAC,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "sentinel1": s1, "sentinel2": s2, "selected_pair": pair,
            "selection_error": error, "analysis_status": "not_run"}
