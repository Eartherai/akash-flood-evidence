"""Offline case planning: catalogue constraints are not acquisition or execution.

This adapter deliberately exposes the existing zone-45N/cached-case limits. It
does not download imagery, run SNAP/models, use evaluation labels, or mutate a
current case. Optical usability is reported only from supplied, hashed SCL data.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from shapely.geometry import Polygon, box, shape

from .catalog import utc, validate_bbox


SCL_REFERENCE = "https://sentiwiki.copernicus.eu/web/s2-processing"
REJECTED_SOURCE = re.compile(r"emsr\d*|copernicus[- _]ems|mapping\.emergency\.copernicus|unosat|worldview|liss.?4|srtm|planetdem|published[- _]damage|reference[- _]map|evaluation[- _]only", re.I)
ALLOWED_INPUTS = ["Sentinel-1 raw GRD", "Sentinel-2 L2A", "Copernicus DEM", "pre-event OpenStreetMap", "listed training datasets (not consumed by this planner)"]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def reject_source_path(path):
    if REJECTED_SOURCE.search(Path(path).as_posix()):
        raise ValueError("Excluded source or evaluation-label path is not a planning input")


def event_boundary(event_date, snapshot):
    event = date.fromisoformat(event_date)
    start = datetime.combine(event, time(), tzinfo=timezone.utc)
    snapshot_time = datetime.combine(date.fromisoformat(snapshot), time(), tzinfo=timezone.utc) if len(snapshot) == 10 else utc(snapshot)
    if snapshot_time >= start:
        raise ValueError("OSM snapshot must be strictly before the start of the event date")
    return start, start + timedelta(days=1), snapshot_time


def normalized_utc(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _item_family(item):
    properties = item.get("properties", {})
    collection = str(item.get("collection", "")).casefold()
    platform = str(properties.get("platform", "")).casefold()
    product = str(properties.get("sar:product_type", "")).upper()
    if (collection == "sentinel-1-grd" and product in ("", "GRD")) or (product == "GRD" and platform.startswith("sentinel-1")):
        return "sentinel1"
    if collection in ("sentinel-2-l2a", "sentinel-2-c1-l2a") and platform.startswith("sentinel-2"):
        return "sentinel2"
    raise ValueError("Only raw Sentinel-1 GRD and Sentinel-2 L2A discovery items are accepted")


def validate_item_source(item):
    if not isinstance(item, dict) or item.get("type") != "Feature" or not isinstance(item.get("id"), str):
        raise ValueError("Discovery items must be identified STAC features")
    properties = item.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("Invalid STAC properties")
    source_fields = [item["id"], str(item.get("collection", "")), str(properties.get("source", ""))]
    for asset in item.get("assets", {}).values():
        if not isinstance(asset, dict):
            raise ValueError("Invalid STAC asset")
        if any(str(role).casefold() in ("label", "labels", "damage", "ground-truth", "reference") for role in asset.get("roles", [])):
            raise ValueError("Published damage/reference labels are excluded from planning inputs")
        source_fields.append(str(asset.get("href", "")))
    if any(REJECTED_SOURCE.search(value) for value in source_fields) or properties.get("reference_labels_loaded") is True:
        raise ValueError("Published damage maps or excluded source family cannot be a planning input")
    return _item_family(item)


def parse_discovery(document):
    if not isinstance(document, dict) or document.get("reference_labels_loaded") is True or document.get("additional_inputs"):
        raise ValueError("Discovery must contain only the declared satellite catalogue inputs")
    if REJECTED_SOURCE.search(str(document.get("source", ""))):
        raise ValueError("Excluded source family cannot provide prediction discovery inputs")
    if document.get("type") == "FeatureCollection":
        items = document.get("features", [])
    elif "sentinel1" in document:
        items = document.get("sentinel1", []) + document.get("sentinel2", [])
    else:
        raise ValueError("Supply a saved discover record or STAC FeatureCollection")
    if not isinstance(items, list) or len(items) > 10000:
        raise ValueError("Discovery must be a bounded item list")
    groups = {"sentinel1": [], "sentinel2": []}
    seen = set()
    for item in items:
        family = validate_item_source(item)
        key = (family, item["id"])
        if key in seen:
            raise ValueError("Duplicate discovery item IDs; provenance would be ambiguous")
        seen.add(key)
        groups[family].append(item)
    incomplete = bool(document.get("search_complete") is False or any(link.get("rel") == "next" for link in document.get("links", [])))
    return groups, incomplete


def _footprint(item, bbox):
    try:
        geometry = shape(item["geometry"])
        return geometry.is_valid and not geometry.is_empty and geometry.covers(box(*bbox))
    except (KeyError, TypeError, ValueError):
        return False


def select_pair(items, bbox, event_start, event_end, max_days):
    """Deterministic raw-IW dual-pol selection, with every exclusion auditable."""
    candidates, exclusions = {"before": [], "after": []}, []
    for item in items:
        properties = item["properties"]
        reason = None
        try:
            acquired = utc(properties["datetime"])
        except (KeyError, TypeError, ValueError):
            reason = "missing_or_invalid_timezone_acquisition"
        orbit = properties.get("sat:relative_orbit")
        pol = properties.get("sar:polarizations")
        if not reason and (isinstance(orbit, bool) or not isinstance(orbit, int) or not 1 <= orbit <= 175 or
                           properties.get("sat:orbit_state") not in ("ascending", "descending") or
                           properties.get("sar:instrument_mode") != "IW" or
                           not isinstance(pol, list) or any(not isinstance(value, str) for value in pol) or
                           len(set(pol)) != len(pol) or not {"VV", "VH"} <= set(pol) or
                           not str(properties.get("platform", "")).casefold().startswith("sentinel-1")):
            reason = "missing_or_unsupported_orbit_mode_platform_or_dual_pol"
        if not reason and not _footprint(item, bbox):
            reason = "footprint_does_not_cover_entire_requested_context"
        if not reason:
            if event_start - timedelta(days=max_days) <= acquired < event_start:
                candidates["before"].append(item)
            elif event_end <= acquired <= event_end + timedelta(days=max_days):
                candidates["after"].append(item)
            else:
                reason = "event_day_or_outside_requested_window"
        if reason:
            exclusions.append({"item_id": item["id"], "reason": reason})
    pairs = []
    for before in candidates["before"]:
        for after in candidates["after"]:
            left, right = before["properties"], after["properties"]
            keys = ("sat:relative_orbit", "sat:orbit_state", "sar:instrument_mode", "platform")
            if any(str(left[key]).casefold() != str(right[key]).casefold() for key in keys) or set(left["sar:polarizations"]) != set(right["sar:polarizations"]):
                continue
            days = (utc(right["datetime"]) - utc(left["datetime"])).total_seconds() / 86400
            if abs(days / 6 - round(days / 6)) * 6 > .1:
                continue
            pairs.append((utc(right["datetime"]), -utc(left["datetime"]).timestamp(), before["id"], after["id"], before, after, days))
    if not pairs:
        return None, exclusions
    _, _, _, _, before, after, days = min(pairs, key=lambda pair: pair[:4])
    properties = before["properties"]
    return {"before": before, "after": after, "separation_days": days,
            "relative_orbit": properties["sat:relative_orbit"], "orbit_state": properties["sat:orbit_state"],
            "platform": properties["platform"], "mode": properties["sar:instrument_mode"],
            "polarizations": sorted(properties["sar:polarizations"]),
            "selection": "earliest compatible post-event acquisition, latest pre-event; IDs break exact ties; whole event day excluded",
            "compatibility_scope": "same platform, relative orbit, pass, IW mode and polarization set; complete context footprints; 6-day multiples within 0.1 day",
            "geolocation_validation": "not established by catalogue pairing; inspect native SAFE metadata/GCPs and validate common grid independently"}, exclusions


def utm_zone(lon, lat):
    zone = min(60, max(1, math.floor((lon + 180) / 6) + 1))
    # Addition/division can round a nextafter() coordinate onto an exact boundary.
    # Compare the original longitude to the nominal interval to restore its side.
    if lon < -180 + 6 * (zone - 1):
        zone -= 1
    elif zone < 60 and lon >= -180 + 6 * zone:
        zone += 1
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone = 32
    if 72 <= lat < 84:
        for west, east, special in ((0, 9, 31), (9, 21, 33), (21, 33, 35), (33, 42, 37)):
            if west <= lon < east:
                zone = special
                break
    return zone


def plan_projection(bbox):
    west, south, east, north = bbox
    if south < -80 or north > 84:
        return {"status": "unsupported_polar_utm", "epsg": None, "limitations": ["Choose an appropriate polar/local projection; current native pipeline is UTM-only."]}
    lon, lat = (west + east) / 2, (south + north) / 2
    zone = utm_zone(lon, lat)
    hemisphere = "north" if lat >= 0 else "south"
    epsg = (32600 if hemisphere == "north" else 32700) + zone
    longitudes = {west, math.nextafter(east, west), lon}
    latitudes = {south, math.nextafter(north, south), lat}
    for boundary in [3, 9, 12, 21, 33, 42] + list(range(-174, 180, 6)):
        if west < boundary < east:
            longitudes.update((math.nextafter(boundary, west), boundary))
    for boundary in (0, 56, 64, 72):
        if south < boundary < north:
            latitudes.update((math.nextafter(boundary, south), boundary))
    zones = sorted({utm_zone(x, y) for x in longitudes for y in latitudes})
    crosses_equator = south < 0 < north
    status = "single_utm_zone" if zones == [zone] and not crosses_equator else "projection_review_required"
    bounds = transform_bounds(4326, epsg, *bbox, densify_pts=21)
    # Existing standard-grid centers are multiples of 10; corners are 5 mod 10.
    left, bottom = [math.floor((value - 5) / 10) * 10 + 5 for value in bounds[:2]]
    right, top = [math.ceil((value - 5) / 10) * 10 + 5 for value in bounds[2:]]
    width, height = round((right - left) / 10), round((top - bottom) / 10)
    return {"status": status, "zone": zone, "hemisphere": hemisphere, "epsg": epsg,
            "wkt1_gdal": CRS.from_epsg(epsg).to_wkt(version="WKT1_GDAL"), "zones_intersected": zones,
            "crosses_equator": crosses_equator, "derivation": "AOI center; Norway/Svalbard special zones included; crossing zones/hemispheres requires explicit review",
            "proposed_grid": {"spacing_m": 10, "corner_origin_modulo_spacing_m": 5,
                              "bounds": [left, bottom, right, top], "shape": [height, width], "estimated_pixels": width * height,
                              "status": "planning estimate only; not a processed raster"}}


def copdem_tiles(bbox, maximum=64):
    west, south, east, north = bbox
    count = (math.ceil(east) - math.floor(west)) * (math.ceil(north) - math.floor(south))
    if count > maximum:
        return {"status": "planning_tile_limit_exceeded", "required_tile_count": count, "maximum": maximum, "tiles": []}
    tiles = []
    for lat in range(math.floor(south), math.ceil(north)):
        for lon in range(math.floor(west), math.ceil(east)):
            name = f"Copernicus_DSM_COG_10_{'N' if lat >= 0 else 'S'}{abs(lat):02}_00_{'E' if lon >= 0 else 'W'}{abs(lon):03}_00_DEM"
            tiles.append({"tile_id": name, "bbox": [lon, lat, lon + 1, lat + 1],
                          "url": f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif",
                          "downloaded": False, "availability": "not probed", "sha256": None})
    return {"status": "planned_not_acquired", "required_tile_count": count, "tiles": tiles,
            "family": "Copernicus GLO-30", "source_vertical_datum": "EGM2008 orthometric height",
            "native_angular_grid": "1 arc-second nominal; not a true 10 m elevation measurement",
            "required_conversion": "Explicit EGM2008 geoid-height conversion to WGS84 ellipsoidal heights, checksum-pinned grid and numerical check; externalDEMApplyEGM=false",
            "limitations": ["Derived URLs are plans; ocean tiles may be absent and public object availability is not established.",
                            "Multi-tile mosaicking, complete processing-context coverage and nodata treatment must be implemented and checked before SNAP."]}


def safe_asset_summary(item):
    assets = {}
    for key, asset in item.get("assets", {}).items():
        url = urlsplit(asset.get("href", ""))
        if url.username or url.password:
            raise ValueError("Credential-bearing asset URLs cannot enter the plan")
        assets[key] = {"href_without_query": urlunsplit((url.scheme, url.netloc, url.path, "", "")),
                       "query_removed": bool(url.query or url.fragment), "roles": asset.get("roles", []),
                       "declared_bytes": asset.get("file:size"), "declared_checksum": asset.get("file:checksum"),
                       "downloaded": False, "content_sha256": None}
    return assets


def scene_summary(item):
    return {"item_id": item["id"], "datetime": item["properties"]["datetime"],
            "catalogue_item_canonical_sha256": canonical_sha(item), "collection": item.get("collection"),
            "assets": safe_asset_summary(item), "footprint": item["geometry"],
            "status": "catalogued_only; raw object availability and checksums not verified"}


def optical_plan(items, bbox, start, end, max_days, *, coverage_path=None):
    candidates, exclusions = [], []
    for item in items:
        try:
            acquired = utc(item["properties"]["datetime"])
        except (KeyError, TypeError, ValueError):
            exclusions.append({"item_id": item["id"], "reason": "invalid_acquisition_timestamp"}); continue
        branch = "before" if start - timedelta(days=max_days) <= acquired < start else "after" if end <= acquired <= end + timedelta(days=max_days) else None
        if not branch or not _footprint(item, bbox):
            exclusions.append({"item_id": item["id"], "reason": "event_day/outside_window_or_partial_footprint"}); continue
        candidate = scene_summary(item)
        candidate.update(branch=branch, tile_cloud_cover_percent=item["properties"].get("eo:cloud_cover"),
                         clear_coverage_status="unknown; catalogue cloud percentage is not AOI clear coverage")
        candidates.append(candidate)
    inputs = []
    if coverage_path:
        reject_source_path(coverage_path)
        coverage_path = Path(coverage_path)
        document = json.loads(coverage_path.read_text())
        if document.get("bbox") != list(bbox) or not isinstance(document.get("scenes"), list):
            raise ValueError("Supplied S2 coverage must name the exact requested bbox and a scenes list")
        inputs.append(file_record(coverage_path))
        by_id = {candidate["item_id"]: candidate for candidate in candidates}
        seen = set()
        for record in document["scenes"]:
            if not isinstance(record, dict) or set(record) != {"item_id", "scl_path", "sha256"} or record["item_id"] not in by_id or record["item_id"] in seen:
                raise ValueError("S2 coverage must identify a unique eligible catalogue scene and its hashed local SCL raster")
            seen.add(record["item_id"])
            path = Path(record["scl_path"])
            if not path.is_absolute():
                path = coverage_path.parent / path
            reject_source_path(path)
            source = file_record(path)
            if source["sha256"] != record["sha256"]:
                raise ValueError("Local S2 SCL checksum does not match the supplied record")
            inputs.append(source)
            with rasterio.open(path) as dataset:
                if dataset.count != 1 or dataset.crs is None or dataset.width * dataset.height > 4_000_000:
                    raise ValueError("Supply one georeferenced local SCL crop of at most four million pixels")
                tx = Transformer.from_crs(dataset.crs, 4326, always_xy=True)
                corners = [tx.transform(*(dataset.transform @ coordinate)) for coordinate in
                           ((0, 0), (dataset.width, 0), (dataset.width, dataset.height), (0, dataset.height))]
                if not Polygon(corners).buffer(1e-10).covers(box(*bbox)):
                    raise ValueError("SCL crop must geometrically cover the complete requested AOI")
                project = Transformer.from_crs(4326, dataset.crs, always_xy=True).transform
                west, south, east, north = bbox
                perimeter = []
                for a, b in (((west, south), (east, south)), ((east, south), (east, north)), ((east, north), (west, north)), ((west, north), (west, south))):
                    for fraction in np.linspace(0, 1, 41, endpoint=False):
                        perimeter.append(project(a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1])))
                aoi = geometry_mask([{"type": "Polygon", "coordinates": [perimeter]}], out_shape=dataset.shape, transform=dataset.transform, invert=True, all_touched=False)
                values = dataset.read(1, masked=True)
                if not aoi.any() or not np.all(np.isin(values.compressed(), np.arange(12))):
                    raise ValueError("Empty AOI sampling or unsupported SCL class encoding")
                raw = values.filled(0)
                clear = aoi & np.isin(raw, (4, 5, 6)) & ~np.ma.getmaskarray(values)
                missing = aoi & ((raw == 0) | np.ma.getmaskarray(values))
                count, clear_count = int(aoi.sum()), int(clear.sum())
                by_id[record["item_id"]].update(clear_coverage_status="all_sampled_aoi_pixels_usable" if clear_count == count else "partial_usable_coverage" if clear_count else "no_usable_optical_pixels",
                    coverage={"aoi_pixels": count, "usable_clear_pixels": clear_count, "unassessed_pixels": int(missing.sum()),
                              "excluded_class_pixels": count - clear_count - int(missing.sum()), "clear_fraction": clear_count / count,
                              "usable_classes": [4, 5, 6], "class_reference": SCL_REFERENCE,
                              "pixel_selection": "AOI pixel centers; supplied SCL grid, no resampling", "source": source,
                              "source_item_association": "declared by supplied record; provider authenticity not independently established",
                              "interpretation": "Usable vegetation/not-vegetated/water class screening only; cloud-free truth, image co-registration and debris detection are not established."})
    return {"status": "no_eligible_optical_items" if not candidates else "local_scl_coverage_inspected" if coverage_path else "catalogued_clear_coverage_unknown",
            "scenes": sorted(candidates, key=lambda scene: (scene["branch"], scene["datetime"], scene["item_id"])),
            "exclusions": exclusions, "local_inputs": inputs, "reflectance_comparison_status": "not_run",
            "limitations": ["Optical is optional when cloud screening leaves no usable pixels; missing optical coverage is explicit, never replaced with generated imagery.",
                            "A tile-wide cloud percentage cannot establish requested-AOI clear coverage."]}


def osm_plan(snapshot, bbox, *, osm_path=None, provenance_path=None):
    west, south, east, north = bbox
    cutoff = normalized_utc(snapshot)
    extent = f"({south},{west},{north},{east})"
    query = (f'[out:json][timeout:90][date:"{cutoff}"];('
             f'way["highway"]{extent};nwr["building"]{extent};nwr["place"~"^(city|town|village|hamlet)$"]{extent};'
             f'nwr["amenity"="hospital"]{extent};);(._;>;);out body;')
    result = {"snapshot": cutoff, "bbox": list(bbox), "query": query,
              "status": "query_planned_not_retrieved", "local_inputs": [], "attribution": "© OpenStreetMap contributors; ODbL",
              "limitations": ["Historical query execution, full feature completeness, facility operability and road restrictions still require validation.",
                              "Nearest baseline town/hospital endpoints and hazard-to-closure policy must be established separately."]}
    if bool(osm_path) != bool(provenance_path):
        raise ValueError("Local OSM extract requires its historical provenance file")
    if osm_path:
        reject_source_path(osm_path); reject_source_path(provenance_path)
        records = [file_record(osm_path), file_record(provenance_path)]
        source = json.loads(Path(osm_path).read_text()); provenance = json.loads(Path(provenance_path).read_text())
        if utc(provenance["snapshot"]) != snapshot or not box(*validate_bbox(provenance["bbox"])).covers(box(*bbox)):
            raise ValueError("Local OSM provenance snapshot or extraction extent does not match the requested case")
        if f'[date:"{cutoff}"]' not in provenance.get("query", ""):
            raise ValueError("Local OSM provenance must retain the exact historical query cutoff")
        canonical_legacy = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        if provenance.get("sha256") not in (records[0]["sha256"], canonical_legacy):
            raise ValueError("OSM content does not match its provenance checksum")
        if source.get("remark") or not source.get("elements"):
            raise ValueError("Local OSM extraction is incomplete or empty")
        for element in source["elements"]:
            if element.get("timestamp") and utc(element["timestamp"]) > snapshot:
                raise ValueError("OSM edit postdates the requested historical snapshot")
        result.update(status="saved_historical_extract_checked", local_inputs=records,
                      provenance_hash_basis="recorded exact-file checksum" if provenance["sha256"] == records[0]["sha256"] else "recorded legacy canonical-JSON checksum; exact file SHA independently retained",
                      history_evidence=provenance.get("history_evidence", "historical query provenance only"))
    return result


def implementation_inventory(root, projection, dem):
    """Present code is an implementation claim, never a fresh-case test result."""
    root = Path(root)
    components = {
        "raw_acquisition": ("scripts/acquire_fresh_native.py", "Raw scene acquisition and derived SAFE wrapper"),
        "dem_preparation": ("scripts/prepare_fresh_dem.py", "Eligible multi-tile CopDEM acquisition/conversion"),
        "native_processing": ("scripts/process_fresh_s1.py", "Generic projection and external-DEM processing"),
    }
    inventory = {}
    for key, (relative, purpose) in components.items():
        path = root / relative
        inventory[key] = {"path": str(path.resolve()), "purpose": purpose,
                          "status": "implementation_present; this case not executed or validated" if path.is_file() else "implementation_not_present_at_planning_time",
                          "source_sha256": sha256(path) if path.is_file() else None, "fresh_case_validated": False}
    constraints = [
        {"component": "legacy native graph/output validation", "path": "scripts/process_s1_snap.py",
         "existing_assumption": "EPSG:32645, longitude 84–90, northern hemisphere, 10 m grid"},
        {"component": "legacy CopDEM preparation", "path": "scripts/prepare_copdem_ellipsoid.py",
         "existing_assumption": "single source tile N28/E085"},
        {"component": "radar registration diagnostic", "path": "scripts/verify_radar_registration.py",
         "existing_assumption": "EPSG:32645/10 m; diagnostic is not absolute or subpixel accuracy validation"},
        {"component": "radar copilot and viewer", "path": "src/akash_flood/radar_copilot.py",
         "existing_assumption": "EPSG:32645/10 m evidence adapter; needs explicit review for a new grid"},
    ]
    for item in constraints:
        path = root / item["path"]
        item["source_sha256"] = sha256(path) if path.is_file() else None
    return {"generic_components": inventory, "legacy_case_assumptions": constraints,
            "fresh_case_execution_validated": False,
            "dem_runner_current_tile_cap": 4,
            "dem_tile_count_within_runner_cap": dem["required_tile_count"] <= 4,
            "radar_reporting_grid_adapter_review_required": projection.get("epsg") != 32645,
            "interpretation": "New generic code may remove acquisition/processing restrictions, but file presence and synthetic tests do not establish live arbitrary-area/date operation. Legacy reporting/evaluation adapter assumptions are exposed separately."}


def build_plan(document, *, bbox, event_date, osm_snapshot, max_days=24, context_buffer_deg=.03, context_bbox=None,
               coverage_path=None, osm_path=None, osm_provenance=None, project_root=None):
    bbox = validate_bbox(bbox)
    start, end, snapshot = event_boundary(event_date, osm_snapshot)
    if isinstance(max_days, bool) or not isinstance(max_days, int) or not 1 <= max_days <= 90:
        raise ValueError("Catalogue window must be an integer from 1 to 90 days")
    if not math.isfinite(context_buffer_deg) or not 0 <= context_buffer_deg <= .25:
        raise ValueError("Context buffer must be from 0 to 0.25 degrees")
    west, south, east, north = bbox
    context = validate_bbox(context_bbox) if context_bbox is not None else validate_bbox([west - context_buffer_deg, south - context_buffer_deg, east + context_buffer_deg, north + context_buffer_deg])
    if not box(*context).covers(box(*bbox)):
        raise ValueError("Explicit processing context must cover the entire requested AOI")
    groups, incomplete = parse_discovery(document)
    pair, exclusions = select_pair(groups["sentinel1"], context, start, end, max_days)
    projection, dem = plan_projection(context), copdem_tiles(context)
    optical = optical_plan(groups["sentinel2"], bbox, start, end, max_days, coverage_path=coverage_path)
    osm = osm_plan(snapshot, bbox, osm_path=osm_path, provenance_path=osm_provenance)
    arguments = {"bbox": bbox, "context_bbox": context, "event_date": event_date, "osm_snapshot": normalized_utc(snapshot),
                 "max_days": max_days, "context_buffer_deg": context_buffer_deg if context_bbox is None else None,
                 "context_source": "explicit_bbox" if context_bbox is not None else "declared_angular_buffer"}
    case_id = "fresh-" + canonical_sha(arguments)[:16]
    blockers = []
    if not pair:
        blockers.append("no_complete_context_same_orbit_dual_pol_IW_pair")
    if incomplete:
        blockers.append("saved_discovery_has_unresolved_pagination_or_explicitly_incomplete_search")
    if projection["status"] != "single_utm_zone":
        blockers.append(projection["status"])
    if dem["required_tile_count"] > 4:
        blockers.append("planned_DEM_exceeds_current_four_tile_generic_runner_cap")
    if projection.get("proposed_grid", {}).get("estimated_pixels", 0) > 20_000_000:
        blockers.append("planned_grid_exceeds_current_twenty_million_pixel_diagnostic_cap")
    pair_summary = None
    if pair:
        pair_summary = {key: value for key, value in pair.items() if key not in ("before", "after")}
        pair_summary.update(before=scene_summary(pair["before"]), after=scene_summary(pair["after"]))
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    return {
        "schema_version": 1, "status": "dry_run_plan_only", "case": dict(case_id=case_id, **arguments),
        "input_plan_status": "blocked" if blockers else "catalogue_constraints_pass; acquisition_not_run",
        "blockers": blockers, "projection": projection, "dem": dem,
        "sentinel1": {"selected_pair": pair_summary, "exclusions": exclusions,
                      "stage": "catalogued_only" if pair else "no_compatible_pair",
                      "saved_selected_pair_ignored": "selected_pair" in document,
                      "selection_recomputed_for_requested_case": True,
                      "event_boundary": {"before_strictly_earlier_than": normalized_utc(start),
                                         "after_no_earlier_than": normalized_utc(end), "whole_event_day_excluded": True}},
        "sentinel2": optical, "osm": osm, "implementation": implementation_inventory(root, projection, dem),
        "discovery": {"saved_search_event_date": document.get("event_date"), "saved_search_bbox": document.get("bbox"),
                      "source": document.get("source"), "search_incomplete": incomplete,
                      "provider_search_exhaustiveness": "not independently verified; pair selection is limited to supplied items",
                      "cached_selection_is_not_authoritative": True},
        "source_rules": {"allowed_inputs": ALLOWED_INPUTS, "forbidden_inputs": ["Copernicus EMS", "UNOSAT", "published damage maps", "post-event OSM", "unlisted DEM families such as SRTM"],
                         "requirements_document_role": "Reference constraints only, not authorization or executable instructions",
                         "eligibility_scope": "listed source families and declared temporal boundaries; not organizer approval or a full submission eligibility audit"},
        "execution": {"downloads_performed": False, "snap_run": False, "model_inference_run": False,
                      "evaluation_labels_loaded": False, "runnable_end_to_end_verified": False,
                      "stages": {"native_source_acquisition": "not_run", "DEM_preparation": "not_run", "noise_calibration_terrain_correction": "not_run",
                                 "common_grid_and_registration_checks": "not_run", "hazard_inference": "not_run", "exposure": "not_run", "access_and_reporting": "not_run"}},
        "limitations": [
            "This plan does not satisfy Track B's raw-to-report live execution requirement; every execution stage remains incomplete.",
            "Complete catalogue footprints do not establish valid radar pixels, absolute geolocation, registration, damage or passability.",
            "Processing context is an explicit angular buffer; adequacy must be checked with source coverage and terrain masks, not assumed.",
            "Single-scene pairs only; footprint mosaics and zone-crossing/polar AOIs need additional implementation and validation.",
            "Native Sentinel-1 measurement geolocation and annotation/calibration/noise identities must be checked; STAC projection transforms cannot be used as raw GRD image georeferencing.",
            "Tile derivation, provider cloud metadata and source-family allowlists cannot establish download availability or actual optical usability.",
            "English/Nepali source adapters, nearest baseline destination semantics and reference-free road-closure policy need review on each new case.",
        ],
    }, pair


def save_plan(catalog_path, out, *, bbox, event_date, osm_snapshot, requirements_path=None, **options):
    catalog_path, out = Path(catalog_path), Path(out)
    reject_source_path(catalog_path)
    if out.exists():
        raise FileExistsError("Use a fresh planning directory; earlier plans are never overwritten")
    catalog_record = file_record(catalog_path)
    document = json.loads(catalog_path.read_text())
    plan, pair = build_plan(document, bbox=bbox, event_date=event_date, osm_snapshot=osm_snapshot, **options)
    inputs = [catalog_record]
    if requirements_path:
        plan["source_rules"]["requirements_document"] = file_record(requirements_path)
        inputs.append(plan["source_rules"]["requirements_document"])
    inputs.extend(plan["sentinel2"]["local_inputs"] + plan["osm"]["local_inputs"])
    if sha256(catalog_path) != catalog_record["sha256"]:
        raise ValueError("Saved catalogue changed while planning")
    out.mkdir(parents=True, exist_ok=False)
    if pair:
        for branch in ("before", "after"):
            path = out / (branch + ".stac.json")
            path.write_text(json.dumps(pair[branch], ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            plan["sentinel1"]["selected_pair"][branch].update(item_path=str(path.resolve()), item_sha256=sha256(path))
    plan["created_at"] = normalized_utc(datetime.now(timezone.utc))
    plan["inputs"] = inputs
    plan_path = out / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    cli = Path(__file__).resolve().parents[2] / "scripts/plan_fresh_case.py"
    manifest = {"status": "complete_dry_run_plan; execution_not_run", "inputs": inputs,
                "code": {"module": file_record(Path(__file__)), "cli": file_record(cli) if cli.exists() else None},
                "outputs": {path.name: file_record(path) for path in sorted(out.iterdir()) if path.is_file()},
                "manifest_self_hash": "not included; compute externally"}
    (out / "source-and-output-manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return plan
