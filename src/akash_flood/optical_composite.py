"""Fixed, CPU-only SCL-selected multi-date optical surface-change support.

The composite has different acquisition dates per cell. Unknown selected pixels
are never filled by reflectance ranking or generated imagery. Surface-change
thresholds remain the original engineering hypotheses, without label tuning.
"""
from __future__ import annotations

import hashlib
import re
import time
from urllib.parse import urlsplit

import numpy as np
import requests
from scipy.ndimage import binary_erosion

from .optical_change import POLICY as ORIGINAL_SURFACE_POLICY
from .optical_observation import eroded_land_support, select_observation_sources

PUBLIC_HOST = "e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com"
BANDS = ("blue", "red", "nir", "swir16")
POLICY = {
    "version": "scl-nearest-date-native20m-composite-v1",
    "pre_window_days": 24,
    "post_window_days_from_event_start": 24,
    "exclude_entire_event_day": True,
    "source_selection": "All full-AOI same-tile sources from the saved complete catalogue; before latest first, after earliest first, datetime then item ID",
    "pixel_selection": "First 60m-eroded land SCL(4,5) source in frozen branch order; selection is fixed before any new reflectance is read",
    "reflectance_acquisition_priority": "Archived complete selected sources first; then descending count of jointly selected SCL cells, source index ties; no spectral or candidate ranking",
    "reflectance_fallback": False,
    "unacquired_or_invalid_selected_reflectance": "unknown; never choose a different date based on spectrum",
    "land_scl_classes": [4, 5],
    "invalid_neighborhood_radius_m": 60,
    "native_output_resolution_m": 20,
    "reflectance_scaling": "Each declared STAC DN*scale+offset; all four native10m samples valid for each exact2x2 average; native20m SWIR unchanged",
    "reflectance_neighborhood": "For each selected source separately, all four bands and SCL land valid within Euclidean60m; no interpolation across dates",
    "surface_change_thresholds": {key: ORIGINAL_SURFACE_POLICY[key] for key in ("pre_ndvi_min", "post_ndvi_max", "ndvi_loss_min", "post_bsi_min_exclusive", "bsi_increase_min", "reflectance_min_exclusive", "reflectance_max_inclusive", "denominator_min_exclusive")},
    "scope": "Multi-date exposed-surface/vegetation-loss candidates requiring review; no confirmed debris, landslides, damage, road closure, forecast or accuracy",
    "maximum_new_body_bytes": 200_000_000,
    "maximum_processing_acquisition_seconds": 600,
    "maximum_attempted_gets": 1024,
    "no_cloud_reconstruction": True,
    "gpu_used": False,
    "reference_labels_loaded": False,
    "original_single_scene_gate_changed": False,
}


def validate_asset_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != PUBLIC_HOST or parsed.query or parsed.fragment
            or not parsed.path.startswith("/sentinel-2-c1-l2a/")
            or parsed.path.rsplit("/", 1)[-1] not in {"SCL.tif", "B02.tif", "B04.tif", "B08.tif", "B11.tif"}):
        raise ValueError("Require an exact unsigned public frozen Collection-1 optical asset")


def select_source_index(current, scl, aoi, source_index):
    if (current.dtype != np.uint8 or aoi.dtype != bool or current.shape != aoi.shape
            or scl.shape != aoi.shape or not 1 <= source_index <= 64):
        raise ValueError("Matching AOI/source arrays and bounded source index required")
    support = eroded_land_support(scl) & aoi
    selected = support & (current == 0)
    current[selected] = source_index
    return {"raw_land_scl_pixels": int((np.isin(scl, (4, 5)) & aoi).sum()),
            "eroded_land_scl_pixels": int(support.sum()), "selected_source_pixels": int(selected.sum())}


def source_reflectance_support(scene, scl):
    if set(scene) != set(BANDS) or any(values.shape != scl.shape for values in scene.values()):
        raise ValueError("Four exact same-grid native20m reflectance bands required")
    supported = np.isin(scl, (4, 5))
    for values in scene.values():
        supported &= np.isfinite(values) & (values > 0) & (values <= 1.5)
    yy, xx = np.mgrid[-3:4, -3:4]
    return binary_erosion(supported, structure=(xx * xx + yy * yy <= 9), border_value=0)


def install_selected_reflectance(composite, available, source_index, selected_index, scene, scl, aoi):
    """Install only the previously selected date; invalidity cannot choose dates."""
    support = source_reflectance_support(scene, scl)
    chosen = aoi & (selected_index == source_index)
    valid = chosen & support
    for key in BANDS:
        composite[key][valid] = scene[key][valid]
    available[valid] = True
    return {"selected_pixels": int(chosen.sum()), "reflectance_supported_selected_pixels": int(valid.sum()),
            "selected_reflectance_invalid_pixels": int((chosen & ~support).sum())}


def screen_composite(before, after, before_available, after_available, aoi):
    if (aoi.dtype != bool or before_available.dtype != bool or after_available.dtype != bool
            or before_available.shape != aoi.shape or after_available.shape != aoi.shape):
        raise ValueError("Matching boolean availability and AOI required")
    assessed = aoi & before_available & after_available
    indices = {}
    for branch, scene in (("before", before), ("after", after)):
        if set(scene) != set(BANDS) or any(values.shape != aoi.shape for values in scene.values()):
            raise ValueError("Four same-grid composite bands required")
        denom = scene["nir"] + scene["red"]
        bsi_denom = sum(scene.values())
        finite = np.isfinite(denom) & np.isfinite(bsi_denom) & (denom > 1e-6) & (bsi_denom > 1e-6)
        for values in scene.values():
            finite &= np.isfinite(values) & (values > 0) & (values <= 1.5)
        ndvi = np.full(aoi.shape, np.nan, dtype="float32")
        bsi = ndvi.copy()
        np.divide(scene["nir"] - scene["red"], denom, out=ndvi, where=finite)
        np.divide(scene["swir16"] + scene["red"] - scene["nir"] - scene["blue"], bsi_denom, out=bsi, where=finite)
        assessed &= finite & np.isfinite(ndvi) & np.isfinite(bsi)
        indices[branch] = {"ndvi": ndvi, "bsi": bsi}
    p, q = indices["before"], indices["after"]
    rule = POLICY["surface_change_thresholds"]
    candidate = (assessed & (p["ndvi"] >= rule["pre_ndvi_min"])
                 & (q["ndvi"] <= rule["post_ndvi_max"])
                 & (p["ndvi"] - q["ndvi"] >= rule["ndvi_loss_min"])
                 & (q["bsi"] > rule["post_bsi_min_exclusive"])
                 & (q["bsi"] - p["bsi"] >= rule["bsi_increase_min"]))
    labels = np.where(assessed, candidate, 255).astype("uint8")
    for values in indices.values():
        for array in values.values():
            array[~assessed] = np.nan
    counts = {"aoi_pixels": int(aoi.sum()), "joint_assessed_pixels": int(assessed.sum()),
              "unknown_pixels": int((aoi & ~assessed).sum()), "candidate_pixels": int(candidate.sum())}
    return labels, assessed, indices, counts


class AuditedCompositeRangeHTTP:
    """Exact public ranges with all attempted GETs and partial delivered bytes.

    Body counts mean bytes delivered to this client, including partial failed
    reads. They cannot count bytes buffered internally by TCP/HTTP libraries but
    never delivered to the application. No automatic HTTP retry is configured.
    """
    def __init__(self, allowed_urls, max_bytes=200_000_000, max_attempts=1024, session=None, deadline=None):
        if (isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1 <= max_bytes <= 200_000_000
                or isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 1024):
            raise ValueError("Composite transport is capped at200MB and1024 attempted GETs")
        self.allowed_urls = frozenset(allowed_urls)
        for url in self.allowed_urls:
            validate_asset_url(url)
        self.deadline = deadline
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
        validate_asset_url(url)
        self.check_deadline()
        if url not in self.allowed_urls:
            raise ValueError("Asset URL is absent from the frozen source allowlist")
        if (isinstance(start, bool) or isinstance(stop, bool) or not isinstance(start, int)
                or not isinstance(stop, int) or start < 0 or not 0 < stop - start <= 4_000_000
                or self.bytes_read + stop - start > self.max_bytes
                or len(self.requests) >= self.max_attempts):
            raise ValueError("Composite range, delivered-body reservation or attempted GET limit exceeded")
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
            response = self.client.get(url, headers=headers, stream=True, timeout=(min(10, self.remaining_seconds()), min(15, self.remaining_seconds())),
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
                self.check_deadline()
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
            "schema": "akash-optical-composite-all-attempt-transport-v1",
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
            "gpu_used": False,
            "provider_whole_object_checksums_verified": False,
        }

    def remaining_seconds(self):
        return max(.1, self.deadline - time.perf_counter()) if self.deadline else 15

    def check_deadline(self):
        if self.deadline and time.perf_counter() >= self.deadline:
            raise TimeoutError("Frozen composite acquisition deadline reached")
