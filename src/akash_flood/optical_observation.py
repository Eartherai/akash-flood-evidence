"""SCL-only observation opportunity; never hazard detection or accuracy.

This is a separate, fixed temporal availability rule. It does not alter the
single-pair optical experiment, reconstruct clouded imagery, or consume bands,
reference labels, radar predictions or candidate outcomes.
"""
from __future__ import annotations

from datetime import timedelta
import hashlib
import re
from urllib.parse import urlsplit

import numpy as np
import requests
from scipy.ndimage import binary_erosion
from shapely.geometry import box, shape

from .catalog import utc, validate_bbox
from .fresh_case import parse_discovery


PUBLIC_HOST = "e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com"
POLICY = {
    "version": "scl-only-native20m-latest-pre-earliest-post-v1",
    "pre_window_days": 24,
    "post_window_days_from_event_start": 24,
    "exclude_entire_event_day": True,
    "full_aoi_footprints_required": True,
    "same_mgrs_tile_crs_and_native_grid_required": True,
    "maximum_source_items": 64,
    "land_scl_classes": [4, 5],
    "invalid_neighborhood_radius_m": 60,
    "output_resolution_m": 20,
    "date_selection": "Latest supported pre-event / earliest supported post-event per cell; acquisition time then source ID; no cloud or outcome ranking",
    "source_index_encoding": "0 unknown/outside AOI; 1-based index into the exact frozen before-then-after item list; AOI mask separates outside from unknown",
    "scope": "SCL-only upper bound on possible paired optical land observations, not reflectance validity, debris detection or accuracy",
    "reflectance_or_reference_pixels_loaded": False,
    "cloud_reconstruction": False,
    "original_single_pair_policy_changed": False,
}


def validate_public_scl_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != PUBLIC_HOST
            or parsed.query or parsed.fragment
            or not parsed.path.startswith("/sentinel-2-c1-l2a/")
            or not parsed.path.endswith("/SCL.tif")):
        raise ValueError("Require the exact unsigned public Collection-1 SCL TIFF URL")


def select_observation_sources(document, bbox, event_date):
    """Select all eligible sources by metadata, without inspecting any pixels."""
    validate_bbox(bbox)
    start = utc(event_date + "T00:00:00Z")
    end = start + timedelta(days=1)
    groups, incomplete = parse_discovery(document)
    if incomplete:
        raise ValueError("A complete bounded saved catalogue is required")
    result = {"before": [], "after": []}
    excluded = []
    for item in groups["sentinel2"]:
        when = utc(item["properties"]["datetime"])
        branch = ("before" if start - timedelta(days=24) <= when < start
                  else "after" if end <= when <= start + timedelta(days=24) else None)
        footprint = shape(item["geometry"])
        if (item.get("collection") != "sentinel-2-c1-l2a" or branch is None
                or not footprint.is_valid or not footprint.covers(box(*bbox))):
            excluded.append(item["id"])
            continue
        asset = item.get("assets", {}).get("scl", {})
        validate_public_scl_url(asset.get("href", ""))
        if (asset.get("gsd") != 20 or not isinstance(asset.get("file:size"), int)
                or isinstance(asset.get("file:size"), bool) or asset["file:size"] <= 0
                or len(asset.get("raster:bands", [])) != 1
                or asset["raster:bands"][0].get("nodata") != 0
                or asset["raster:bands"][0].get("data_type") != "uint8"):
            raise ValueError("Require an explicit single-band native20m uint8 SCL source")
        result[branch].append(item)
    if not all(result.values()):
        raise ValueError("At least one full-AOI source on each side of the event is required")
    sources = result["before"] + result["after"]
    if len(sources) > POLICY["maximum_source_items"]:
        raise ValueError("Source count exceeds the fixed observation gate cap")
    def signature(item):
        props, asset = item["properties"], item["assets"]["scl"]
        return (props["mgrs:utm_zone"], props["mgrs:latitude_band"], props["mgrs:grid_square"],
                props["proj:epsg"], tuple(asset["proj:shape"]), tuple(asset["proj:transform"]))
    if any(signature(item) != signature(sources[0]) for item in sources[1:]):
        raise ValueError("Multiple tiles or differing native grids are unsupported; no alignment/resampling")
    result["before"].sort(key=lambda item: (-utc(item["properties"]["datetime"]).timestamp(), item["id"]))
    result["after"].sort(key=lambda item: (utc(item["properties"]["datetime"]).timestamp(), item["id"]))
    return result, excluded


def eroded_land_support(scl):
    if not isinstance(scl, np.ndarray) or scl.ndim != 2 or scl.dtype != np.uint8:
        raise ValueError("Native two-dimensional uint8 SCL is required")
    if np.any(scl > 11):
        raise ValueError("SCL contains a value outside the documented0..11 class domain")
    yy, xx = np.mgrid[-3:4, -3:4]
    return binary_erosion(np.isin(scl, (4, 5)), structure=(xx * xx + yy * yy <= 9), border_value=0)


def update_date_index(index, scl, aoi, source_index):
    """Take the first supported source in the frozen branch preference order."""
    if (index.dtype != np.uint8 or aoi.dtype != bool or index.shape != aoi.shape
            or scl.shape != aoi.shape or not 1 <= source_index <= 64):
        raise ValueError("Matching native arrays and a bounded1-based frozen source index are required")
    support = eroded_land_support(scl) & aoi
    newly_available = support & (index == 0)
    index[newly_available] = source_index
    return {
        "raw_land_scl_pixels": int((np.isin(scl, (4, 5)) & aoi).sum()),
        "eroded_land_scl_pixels": int(support.sum()),
        "selected_source_pixels": int(newly_available.sum()),
    }


def summarize_availability(before_index, after_index, aoi):
    if (aoi.dtype != bool or before_index.dtype != np.uint8 or after_index.dtype != np.uint8
            or before_index.shape != aoi.shape or after_index.shape != aoi.shape
            or np.any(before_index[~aoi]) or np.any(after_index[~aoi]) or not aoi.any()):
        raise ValueError("Matching native AOI/date-index arrays with zero outside AOI are required")
    joint = aoi & (before_index != 0) & (after_index != 0)
    n = int(aoi.sum())
    counts = {
        "aoi_pixels": n,
        "pre_available_pixels": int((aoi & (before_index != 0)).sum()),
        "post_available_pixels": int((aoi & (after_index != 0)).sum()),
        "joint_available_pixels": int(joint.sum()),
        "joint_unknown_pixels": int((aoi & ~joint).sum()),
    }
    return joint, counts, counts["joint_available_pixels"] / n


class AuditedSCLRangeHTTP:
    """Exact public ranges with all attempted GETs and partial delivered bytes.

    Body counts mean bytes delivered to this client, including partial failed
    reads. They cannot count bytes buffered internally by TCP/HTTP libraries but
    never delivered to the application. No automatic HTTP retry is configured.
    """
    def __init__(self, max_bytes=20_000_000, max_attempts=256, session=None):
        if (isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1 <= max_bytes <= 20_000_000
                or isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 256):
            raise ValueError("SCL transport is capped at20MB and256 attempted GETs")
        self.max_bytes, self.max_attempts = max_bytes, max_attempts
        self.bytes_read, self.requests, self.identities = 0, [], {}
        self.client = session or requests.Session()
        self.client.trust_env = False
        if (getattr(self.client, "auth", None) or getattr(self.client, "cookies", None)
                or getattr(self.client, "params", None) or getattr(self.client, "proxies", None)
                or any(str(key).lower() in {"authorization", "proxy-authorization", "cookie", "x-api-key", "x-amz-security-token"}
                       for key in getattr(self.client, "headers", {}))):
            raise ValueError("Credential-bearing sessions are not accepted")
        if isinstance(self.client, requests.Session):
            self.client.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))

    def range(self, url, start, stop):
        validate_public_scl_url(url)
        if (isinstance(start, bool) or isinstance(stop, bool) or not isinstance(start, int)
                or not isinstance(stop, int) or start < 0 or not 0 < stop - start <= 4_000_000
                or self.bytes_read + stop - start > self.max_bytes
                or len(self.requests) >= self.max_attempts):
            raise ValueError("SCL range, delivered-body reservation or attempted GET limit exceeded")
        count = stop - start
        headers = {"Range": f"bytes={start}-{stop - 1}", "Accept-Encoding": "identity",
                   "User-Agent": "AkashSCLObservationGate/1.0"}
        if url in self.identities:
            headers["If-Match"] = self.identities[url][1]
        entry = {"attempt": len(self.requests) + 1, "method": "GET", "url": url,
                 "start": start, "requested_stop": stop, "status": None,
                 "received_body_bytes": 0, "completed": False}
        self.requests.append(entry)  # Recorded before the network call, including connection failures.
        digest, response = hashlib.sha256(), None
        try:
            response = self.client.get(url, headers=headers, stream=True, timeout=(15, 30),
                                       verify=True, allow_redirects=False)
            entry["status"] = response.status_code
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if response.status_code != 206 or not match:
                raise ValueError("Exact206 range response required; no redirect/full-scene fallback")
            first, last, total = map(int, match.groups())
            etag = response.headers.get("ETag")
            expected = last - first + 1
            entry.update({"stop": last + 1, "file_bytes": total, "etag": etag})
            if (first != start or last + 1 != min(stop, total) or expected <= 0 or not etag
                    or response.headers.get("Content-Encoding", "identity") != "identity"
                    or response.headers.get("Content-Length") != str(expected)):
                raise ValueError("Source range, ETag, encoding or declared body length differs")
            identity = (total, etag)
            if url in self.identities and self.identities[url] != identity:
                raise ValueError("Remote source identity changed")
            self.identities[url] = identity
            chunks = []
            # Content-Length is verified before any body read. No full response is read on failure.
            for chunk in response.iter_content(min(65536, expected)):
                if not chunk:
                    continue
                entry["received_body_bytes"] += len(chunk)
                self.bytes_read += len(chunk)
                digest.update(chunk)
                if entry["received_body_bytes"] > expected or self.bytes_read > self.max_bytes:
                    raise ValueError("Delivered source body exceeds its exact range or aggregate cap")
                chunks.append(chunk)
            if entry["received_body_bytes"] != expected:
                raise ValueError("Incomplete SCL source range")
            entry.update({"completed": True, "sha256": digest.hexdigest()})
            return b"".join(chunks), total
        except Exception as error:
            entry["error"] = {"type": type(error).__name__, "message": str(error)[:1000]}
            entry["partial_sha256"] = digest.hexdigest()
            raise
        finally:
            if response is not None:
                response.close()

    def audit(self):
        if self.bytes_read != sum(entry["received_body_bytes"] for entry in self.requests):
            raise RuntimeError("Transport delivered-body ledger disagrees with aggregate")
        object_coverage = {}
        for url, (length, _) in self.identities.items():
            intervals = sorted((entry["start"], entry["stop"]) for entry in self.requests
                               if entry["url"] == url and entry["completed"])
            covered_to = 0
            for start, stop in intervals:
                if start > covered_to:
                    break
                covered_to = max(covered_to, stop)
            object_coverage[url] = {"file_bytes": length,
                                   "whole_object_received_through_completed_ranges": covered_to == length}
        return {
            "schema": "akash-scl-all-attempt-transport-v1",
            "received_body_bytes": self.bytes_read,
            "attempted_get_count": len(self.requests),
            "completed_get_count": sum(entry["completed"] for entry in self.requests),
            "failed_get_count": sum(not entry["completed"] for entry in self.requests),
            "requests": self.requests, "maximum_body_bytes": self.max_bytes,
            "maximum_attempted_gets": self.max_attempts, "maximum_individual_range_bytes": 4_000_000,
            "body_accounting_scope": "All application-delivered response chunks, including partial failed reads; internal undelivered HTTP/TCP buffering is not measurable",
            "tls_verified": True, "redirects_followed": False, "automatic_http_retries": False,
            "account_used": False, "unbounded_whole_scene_fallback_used": False,
            "source_full_objects_downloaded": any(value["whole_object_received_through_completed_ranges"]
                                                    for value in object_coverage.values()),
            "source_object_range_coverage": object_coverage,
            "provider_whole_object_checksums_verified": False,
        }
