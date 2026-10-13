"""Bounded public Sentinel-1 acquisition and explicitly derived SAFE wrappers.

This is a transport/identity adapter, not a calibration or hazard algorithm.
Provider COGs do not inherit the original ESA measurement checksums. The original
manifest is retained verbatim; only the separate derived manifest gets actual
provider-byte sizes and MD5s. All source bytes also receive SHA-256 receipts.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import requests

SCHEMA = "akash-fresh-native-acquisition-v1"
BUCKET = "sentinel-s1-l1c"
HOST = f"{BUCKET}.s3.amazonaws.com"
ITEM_RE = re.compile(r"S1[ABCD]_IW_GRDH_1SDV_\d{8}T\d{6}_\d{8}T\d{6}_\d{6}_[0-9A-F]{6}\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
EXCLUDED = re.compile(r"emsr\d*|unosat|worldview|liss.?4|srtm|planetdem|published[- _]damage|reference[- _]map|evaluation[- _]only", re.I)
ALLOWED_ASSETS = {
    "safe-manifest", "vv", "vh", "thumbnail",
    "schema-product-vv", "schema-product-vh", "schema-calibration-vv",
    "schema-calibration-vh", "schema-noise-vv", "schema-noise-vh",
}
DEFAULT_BUDGET = 4_100_000_000
DEFAULT_MEASUREMENT_CAP = 2_000_000_000
METADATA_CAP = 10_000_000


def digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def json_write(path: Path, value: Any) -> None:
    """Create a new JSON artifact; existing results are never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def json_read(path: Path) -> dict:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def positive_int(value: Any, name: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds {maximum}")
    return value


def utc_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Acquisition datetime is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("Acquisition datetime must be UTC")
    return parsed


def public_url(href: str, expected_key: str) -> str:
    """Convert only the selected public bucket/key into strict HTTPS transport."""
    if not isinstance(href, str):
        raise ValueError("Asset href must be a string")
    parsed = urlsplit(href)
    if parsed.username or parsed.password or parsed.port or parsed.query or parsed.fragment:
        raise ValueError("Asset URL must not contain credentials, ports, queries or fragments")
    if "%" in parsed.path or "\\" in parsed.path:
        raise ValueError("Encoded or backslash asset paths are not accepted")
    if parsed.scheme == "s3" and parsed.netloc == BUCKET:
        key = parsed.path.removeprefix("/")
    elif parsed.scheme == "https" and parsed.netloc in {HOST, f"{BUCKET}.s3.eu-central-1.amazonaws.com"}:
        key = parsed.path.removeprefix("/")
    else:
        raise ValueError("Only the public Sentinel-1 mirror bucket is accepted")
    if key != expected_key:
        raise ValueError("Asset source identity differs from the selected product")
    return f"https://{HOST}/{key}"


@dataclass(frozen=True)
class Scene:
    name: str
    item_path: Path
    item_sha256: str
    item: dict
    item_id: str
    product_identifier: str
    prefix: str
    acquisition_time: str

    @property
    def manifest_url(self) -> str:
        return f"https://{HOST}/{self.prefix}/manifest.safe"

    def record(self) -> dict:
        return {
            "path": str(self.item_path), "sha256": self.item_sha256,
            "item_id": self.item_id, "product_identifier": self.product_identifier,
            "datetime": self.acquisition_time,
            "measurement_urls": {pol: f"https://{HOST}/{self.prefix}/measurement/iw-{pol}.tiff" for pol in ("vv", "vh")},
            "original_manifest_url": self.manifest_url,
        }


def validate_scene(name: str, item_path: Path, expected_sha256: str | None = None) -> Scene:
    if name not in {"before", "after"}:
        raise ValueError("Scene must be before or after")
    if EXCLUDED.search(item_path.as_posix()):
        raise ValueError("Published damage/reference paths are not acquisition inputs")
    item_path = item_path.resolve(strict=True)
    if not item_path.is_file() or item_path.stat().st_size > METADATA_CAP:
        raise ValueError("STAC metadata file is missing or too large")
    sha = digest(item_path)
    if expected_sha256 is not None and (not SHA_RE.fullmatch(expected_sha256) or sha != expected_sha256):
        raise ValueError("Selected STAC checksum differs from the frozen planner")
    item = json_read(item_path)
    if item.get("type") != "Feature" or item.get("collection") != "sentinel-1-grd":
        raise ValueError("Only EarthSearch Sentinel-1 GRD STAC items are accepted")
    identity = item.get("id", "")
    if not isinstance(identity, str) or not ITEM_RE.fullmatch(identity):
        raise ValueError("Unsupported Sentinel-1 IW dual-polarization product identity")
    properties = item.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("STAC properties must be an object")
    if properties.get("reference_labels_loaded") is True or EXCLUDED.search(str(properties.get("source", ""))):
        raise ValueError("Published damage/reference content is not an acquisition input")
    product = properties.get("s1:product_identifier", "")
    if not isinstance(product, str) or not re.fullmatch(re.escape(identity) + r"_[0-9A-F]{4}", product):
        raise ValueError("Product identifier does not match STAC item identity")
    if properties.get("sar:instrument_mode") != "IW" or properties.get("sar:product_type") != "GRD":
        raise ValueError("Only native IW GRD source measurements are accepted")
    if properties.get("sar:polarizations") not in (["VV", "VH"], ["VH", "VV"]):
        raise ValueError("Both VV and VH must be present without extra polarizations")
    if properties.get("sat:orbit_state") not in {"ascending", "descending"}:
        raise ValueError("Orbit direction is missing")
    positive_int(properties.get("sat:relative_orbit"), "relative orbit", 175)
    observed = utc_time(properties.get("datetime"))
    start = utc_time(properties.get("start_datetime"))
    end = utc_time(properties.get("end_datetime"))
    if not start <= observed <= end or start.date() != end.date():
        raise ValueError("STAC acquisition times are inconsistent")
    # Product identity carries the exact acquisition start/end timestamps.
    parts = identity.split("_")
    if start.strftime("%Y%m%dT%H%M%S") != parts[4] or end.strftime("%Y%m%dT%H%M%S") != parts[5]:
        raise ValueError("STAC datetime differs from product identity")
    if properties.get("platform") != "sentinel-1" + identity[2].lower():
        raise ValueError("Platform differs from product mission identity")
    if properties.get("sat:absolute_orbit", int(parts[6])) != int(parts[6]):
        raise ValueError("Absolute orbit differs from product identity")
    if str(properties.get("s1:datatake_id", int(parts[7], 16))) != str(int(parts[7], 16)):
        raise ValueError("Datatake differs from product identity")
    prefix = f"GRD/{start.year}/{start.month}/{start.day}/IW/DV/{product}"
    assets = item.get("assets")
    if not isinstance(assets, dict) or not {"vv", "vh", "safe-manifest"} <= assets.keys():
        raise ValueError("Native measurements or original manifest are missing")
    if not assets.keys() <= ALLOWED_ASSETS:
        raise ValueError("Non-source assets, including label/damage layers, are not accepted")
    paths = {"vv": "measurement/iw-vv.tiff", "vh": "measurement/iw-vh.tiff", "safe-manifest": "manifest.safe", "thumbnail": "preview/quick-look.png"}
    for pol in ("vv", "vh"):
        paths.update({f"schema-product-{pol}": f"annotation/rfi/rfi-iw-{pol}.xml", f"schema-calibration-{pol}": f"annotation/calibration/calibration-iw-{pol}.xml", f"schema-noise-{pol}": f"annotation/calibration/noise-iw-{pol}.xml"})
    for key, asset in assets.items():
        if not isinstance(asset, dict):
            raise ValueError("Asset metadata must be an object")
        public_url(asset.get("href"), f"{prefix}/{paths[key]}")
        roles = asset.get("roles", [])
        if not isinstance(roles, list) or any(not isinstance(role, str) or role.casefold() in {"labels", "label", "classification", "damage", "reference", "ground-truth"} for role in roles):
            raise ValueError("Published labels/damage are not acquisition inputs")
    self_links = [link.get("href") for link in item.get("links", []) if isinstance(link, dict) and link.get("rel") == "self"]
    expected_self = f"https://earth-search.aws.element84.com/v1/collections/sentinel-1-grd/items/{identity}"
    if self_links != [expected_self]:
        raise ValueError("STAC self identity is not the selected EarthSearch item")
    return Scene(name, item_path, sha, item, identity, product, prefix, properties["datetime"])


def validate_pair(before: Scene, after: Scene) -> None:
    if before.item_id == after.item_id or utc_time(before.acquisition_time) >= utc_time(after.acquisition_time):
        raise ValueError("Before/after must be distinct and chronologically ordered")
    for key in ("sat:relative_orbit", "sat:orbit_state", "sar:instrument_mode", "platform"):
        if before.item["properties"].get(key) != after.item["properties"].get(key):
            raise ValueError(f"Before/after pairing differs in {key}")
    if set(before.item["properties"]["sar:polarizations"]) != set(after.item["properties"]["sar:polarizations"]):
        raise ValueError("Before/after pairing differs in polarization set")


def manifest_objects(content: bytes, scene: Scene) -> tuple[ET.Element, list[dict]]:
    if len(content) > METADATA_CAP or re.search(br"<!\s*(?:DOCTYPE|ENTITY)", content, re.IGNORECASE):
        raise ValueError("Manifest XML exceeds cap or contains a document/entity declaration")
    root = ET.fromstring(content)
    if root.tag.split("}")[-1] != "XFDU":
        raise ValueError("Original product manifest must be XFDU")
    starts = [e.text for e in root.iter() if e.tag.split("}")[-1] == "startTime"]
    stops = [e.text for e in root.iter() if e.tag.split("}")[-1] == "stopTime"]
    if len(starts) != 1 or len(stops) != 1:
        raise ValueError("Manifest acquisition period is missing or ambiguous")
    def manifest_time(value):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    if any(abs((manifest_time(value) - utc_time(scene.item["properties"][key])).total_seconds()) > .001 for value, key in ((starts[0], "start_datetime"), (stops[0], "end_datetime"))):
        raise ValueError("Original manifest acquisition period differs from selected STAC")
    values = {}
    for element in root.iter():
        values.setdefault(element.tag.split("}")[-1], []).append(element.text)
    parts = scene.item_id.split("_")
    identities = {"orbitNumber": {str(int(parts[6]))}, "relativeOrbitNumber": {str(scene.item["properties"]["sat:relative_orbit"])}, "pass": {scene.item["properties"]["sat:orbit_state"].upper()}, "mode": {"IW"}, "productType": {"GRD"}, "transmitterReceiverPolarisation": {"VV", "VH"}, "missionDataTakeID": {str(int(parts[7], 16))}, "number": {scene.item_id[2]}}
    for key, expected in identities.items():
        if set(values.get(key, [])) != expected:
            raise ValueError(f"Original manifest {key} differs from selected product")
    records, seen, required = [], set(), set()
    for obj in root.iter():
        if obj.tag.split("}")[-1] != "dataObject":
            continue
        streams = [e for e in obj if e.tag.split("}")[-1] == "byteStream"]
        if len(streams) != 1:
            raise ValueError("Manifest object must contain one byteStream")
        stream = streams[0]
        locations = [e for e in stream if e.tag.split("}")[-1] == "fileLocation"]
        checksums = [e for e in stream if e.tag.split("}")[-1] == "checksum"]
        if len(locations) != 1 or len(checksums) != 1:
            raise ValueError("Manifest object location/checksum is missing or ambiguous")
        alias = locations[0].get("href", "").removeprefix("./")
        path = PurePosixPath(alias)
        if not alias or path.is_absolute() or ".." in path.parts or "\\" in alias or "%" in alias or str(path) != alias or alias in seen:
            raise ValueError("Unsafe or duplicate manifest alias path")
        seen.add(alias)
        if alias.startswith(("measurement/", "annotation/")):
            match = re.search(r"-(vv|vh)-", path.name)
            if not match:
                raise ValueError("Manifest object is not a VV/VH source component")
            pol = match.group(1)
            # Full alias identity binds mission, mode, polarization, acquisition,
            # absolute orbit and datatake, not merely the containing directory.
            parts = scene.item_id.lower().split("_")
            core = f"{parts[0]}-iw-grd-{pol}-{parts[4]}-{parts[5]}-{parts[6]}-{parts[7]}-"
            prefix = "noise-" if alias.startswith("annotation/calibration/noise-") else "calibration-" if alias.startswith("annotation/calibration/calibration-") else "rfi-" if alias.startswith("annotation/rfi/rfi-") else ""
            if not re.fullmatch(re.escape(prefix + core) + r"\d{3}\.(?:xml|tiff)", path.name):
                raise ValueError("Manifest component identity differs from selected product")
            if alias.startswith("measurement/") and len(path.parts) == 2 and path.suffix == ".tiff":
                source_relative, kind = f"measurement/iw-{pol}.tiff", "measurement"
            elif alias.startswith("annotation/calibration/noise-") and len(path.parts) == 3 and path.suffix == ".xml":
                source_relative, kind = f"annotation/calibration/noise-iw-{pol}.xml", "noise"
            elif alias.startswith("annotation/calibration/calibration-") and len(path.parts) == 3 and path.suffix == ".xml":
                source_relative, kind = f"annotation/calibration/calibration-iw-{pol}.xml", "calibration"
            elif alias.startswith("annotation/rfi/rfi-") and len(path.parts) == 3 and path.suffix == ".xml":
                source_relative, kind = f"annotation/rfi/rfi-iw-{pol}.xml", "rfi"
            elif alias.startswith("annotation/") and len(path.parts) == 2 and path.suffix == ".xml":
                source_relative, kind = f"annotation/iw-{pol}.xml", "annotation"
            else:
                raise ValueError("Unsupported native manifest component")
            if (kind, pol) in required:
                raise ValueError("Duplicate manifest component for polarization")
            required.add((kind, pol))
        elif alias in {"preview/map-overlay.kml", "preview/product-preview.html", "preview/quick-look.png"}:
            source_relative, kind = alias, "preview"
        else:
            raise ValueError("Unsupported native manifest object; labels/damage are forbidden")
        checksum = checksums[0]
        if checksum.get("checksumName") != "MD5" or not re.fullmatch(r"[0-9a-fA-F]{32}", checksum.text or ""):
            raise ValueError("Original manifest checksum is missing or unsupported")
        size = stream.get("size", "")
        if not size.isdigit() or int(size) <= 0:
            raise ValueError("Original manifest size must be positive")
        records.append({"alias": alias, "public_url": f"https://{HOST}/{scene.prefix}/{source_relative}", "source_relative": source_relative, "kind": kind, "original_manifest_size": int(size), "original_manifest_checksum": checksum.text.lower(), "original_manifest_checksum_algorithm": "MD5", "stream": stream, "checksum_element": checksum})
    needed = {(kind, pol) for kind in ("measurement", "annotation", "noise", "calibration") for pol in ("vv", "vh")}
    if not needed <= required:
        raise ValueError("Manifest lacks required VV/VH measurement, annotation, noise or calibration")
    return root, records


def validate_metadata_component(content: bytes, scene: Scene, source_relative: str) -> None:
    if not source_relative.startswith("annotation/"):
        return
    if len(content) > METADATA_CAP or re.search(br"<!\s*(?:DOCTYPE|ENTITY)", content, re.IGNORECASE):
        raise ValueError("Native metadata XML exceeds cap or contains a document/entity declaration")
    root = ET.fromstring(content)
    header = root.find("adsHeader")
    if header is None:
        raise ValueError("Native annotation/calibration/noise product identity is missing")
    values = {element.tag: element.text for element in header}
    if len(values) != len(list(header)):
        raise ValueError("Native metadata header has ambiguous identity fields")
    pol = "VV" if source_relative.endswith("-vv.xml") else "VH"
    parts = scene.item_id.split("_")
    expected = {"missionId": parts[0], "productType": "GRD", "polarisation": pol, "mode": "IW", "swath": "IW", "absoluteOrbitNumber": str(int(parts[6])), "missionDataTakeId": str(int(parts[7], 16))}
    for key, value in expected.items():
        if values.get(key) != value:
            raise ValueError(f"Native metadata {key} differs from selected source identity")
    for key, item_key in (("startTime", "start_datetime"), ("stopTime", "end_datetime")):
        value = values.get(key)
        if not value:
            raise ValueError("Native metadata acquisition time is missing")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        parsed = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
        if abs((parsed - utc_time(scene.item["properties"][item_key])).total_seconds()) > .001:
            raise ValueError("Native metadata acquisition period differs from selected source identity")


@dataclass
class NetworkBudget:
    maximum: int = DEFAULT_BUDGET
    transferred: int = 0
    requests: int = 0

    def __post_init__(self):
        positive_int(self.maximum, "network budget")

    def check(self, amount: int):
        if amount < 0 or amount > self.maximum - self.transferred:
            raise ValueError("Network byte budget would be exceeded")

    def consume(self, amount: int):
        self.check(amount)
        self.transferred += amount


class PublicClient:
    """HEAD + If-Match GET. No redirects, credentials, resumptions or TLS bypass."""
    def __init__(self, budget: NetworkBudget, session=None):
        self.budget = budget
        self.session = session if session is not None else requests.Session()
        # Disable environment/netrc auth and proxy injection for this public-only path.
        self.session.trust_env = False
        if getattr(self.session, "auth", None) or any(key.casefold() in {"authorization", "proxy-authorization"} for key in getattr(self.session, "headers", {})):
            raise ValueError("Credentials are forbidden for public Sentinel-1 acquisition")

    @staticmethod
    def validate_transport(url: str):
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != HOST or parsed.query or parsed.fragment or parsed.username or parsed.password or parsed.port:
            raise ValueError("Transport must be strict public Sentinel-1 HTTPS")
        if not parsed.path.startswith("/GRD/") or any(part in {"..", "."} for part in parsed.path.split("/")) or "%" in parsed.path or "\\" in parsed.path:
            raise ValueError("Transport path is unsafe")

    def request(self, method: str, url: str, **kwargs):
        self.validate_transport(url)
        self.budget.requests += 1
        response = self.session.request(method, url, timeout=(15, 60), allow_redirects=False, verify=True, **kwargs)
        if response.status_code != 200:
            response.close()
            raise ValueError(f"Public source returned {response.status_code}; redirects/auth fallback are disabled")
        return response

    def head(self, url: str, cap: int) -> dict:
        response = self.request("HEAD", url, headers={"Accept-Encoding": "identity"})
        try:
            length, etag = response.headers.get("Content-Length", ""), response.headers.get("ETag", "")
            if not length.isdigit() or int(length) <= 0 or int(length) > cap:
                raise ValueError("Public source advertised size is missing or exceeds individual cap")
            if not etag or etag.startswith("W/") or "\r" in etag or "\n" in etag:
                raise ValueError("Strong provider ETag is required for source identity binding")
            return {"bytes": int(length), "etag": etag, "last_modified": response.headers.get("Last-Modified"), "public_url": url}
        finally:
            response.close()

    def get(self, url: str, target: Path, expected: dict) -> dict:
        self.budget.check(expected["bytes"])
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to replace source bytes: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        response = self.request("GET", url, stream=True, headers={"Accept-Encoding": "identity", "If-Match": expected["etag"]})
        temporary = target.with_name(target.name + ".partial")
        if temporary.exists() or temporary.is_symlink():
            response.close()
            raise FileExistsError(f"Refusing stale partial file: {temporary}")
        try:
            length = response.headers.get("Content-Length", "")
            if length != str(expected["bytes"]) or response.headers.get("ETag") != expected["etag"]:
                raise ValueError("Public source changed identity between HEAD and GET")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise ValueError("Encoded transport is forbidden for byte identity")
            received = 0
            with temporary.open("xb") as stream:
                for chunk in response.iter_content(chunk_size=131072):
                    if not chunk:
                        continue
                    # Count received bodies even if a dishonest server exceeds its size.
                    self.budget.consume(len(chunk))
                    received += len(chunk)
                    if received > expected["bytes"]:
                        raise ValueError("Public source body exceeds advertised size")
                    stream.write(chunk)
            if received != expected["bytes"]:
                raise ValueError("Public source body is incomplete")
            sha = digest(temporary)
            temporary.rename(target)
            return {**expected, "sha256": sha, "path": str(target.resolve()), "transport": "anonymous HTTPS; TLS verified; redirects disabled; strong ETag + If-Match", "downloaded_at": datetime.now(timezone.utc).isoformat()}
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            response.close()


def cached_source(client: PublicClient, url: str, cache_path: Path, cap: int) -> tuple[Path, dict, Path]:
    """Rehash cached bytes and revalidate provider ETag/size before reusing them."""
    receipt_path = cache_path.with_name(cache_path.name + ".receipt.json")
    advertised = client.head(url, cap)
    if cache_path.exists() or receipt_path.exists():
        if cache_path.is_symlink() or receipt_path.is_symlink() or not cache_path.is_file() or not receipt_path.is_file():
            raise ValueError("Cache bytes and receipt must both exist as regular files")
        receipt = json_read(receipt_path)
        for field in ("bytes", "etag", "public_url"):
            if receipt.get(field) != advertised[field]:
                raise ValueError("Cached source identity differs from current public object")
        if cache_path.stat().st_size != receipt["bytes"] or digest(cache_path) != receipt.get("sha256") or receipt.get("path") != str(cache_path.resolve()):
            raise ValueError("Cached source checksum/path differs from its receipt")
        return cache_path, {**receipt, "reused_from_cache": True, "revalidated_at": datetime.now(timezone.utc).isoformat()}, receipt_path
    receipt = client.get(url, cache_path, advertised)
    json_write(receipt_path, receipt)
    return cache_path, {**receipt, "reused_from_cache": False}, receipt_path


def acquire(scenes: list[Scene], output_dir: Path, cache_dir: Path, budget_bytes: int = DEFAULT_BUDGET, measurement_cap: int = DEFAULT_MEASUREMENT_CAP, source_plan: dict | None = None, session=None, imported_sources: dict[str, dict] | None = None) -> dict:
    """Execute bounded acquisition into a new output directory."""
    if len(scenes) != 2 or [scene.name for scene in scenes] != ["before", "after"]:
        raise ValueError("Exactly before/after scenes in order are required")
    validate_pair(scenes[0], scenes[1])
    for scene in scenes:
        if digest(scene.item_path) != scene.item_sha256:
            raise ValueError("Selected STAC changed after input validation")
    if source_plan is not None and digest(Path(source_plan["path"])) != source_plan["sha256"]:
        raise ValueError("Selected planner changed after input validation")
    positive_int(measurement_cap, "measurement cap", DEFAULT_MEASUREMENT_CAP)
    output_dir = output_dir.absolute()
    cache_dir = cache_dir.absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Acquisition output directory must not exist")
    if cache_dir.is_symlink() or output_dir.is_relative_to(cache_dir) or cache_dir.is_relative_to(output_dir):
        raise ValueError("Separate, non-symlink output and cache roots are required")
    budget = NetworkBudget(budget_bytes)
    client = PublicClient(budget, session=session)
    imported_sources = imported_sources or {}
    def obtain(url, target, cap):
        if url in imported_sources:
            if imported_sources[url]["bytes"] > cap:
                raise ValueError("Imported source exceeds individual cap")
            return import_source(url, target, imported_sources[url])
        return cached_source(client, url, target, cap)
    # Probe all large objects before transferring any measurement bodies.
    raw_headers = {}
    for scene in scenes:
        for pol in ("vv", "vh"):
            url = f"https://{HOST}/{scene.prefix}/measurement/iw-{pol}.tiff"
            if url in imported_sources:
                imported = imported_sources[url]
                if imported["bytes"] > measurement_cap:
                    raise ValueError("Imported measurement exceeds individual cap")
                raw_headers[(scene.name, pol)] = {"bytes": imported["bytes"], "etag": None, "public_url": url}
            else:
                raw_headers[(scene.name, pol)] = client.head(url, measurement_cap)
    pending = sum(value["bytes"] for value in raw_headers.values())
    download_pending = sum(value["bytes"] for value in raw_headers.values() if value["public_url"] not in imported_sources)
    # Conservatively reserve all four advertised sizes, including cached assets.
    if download_pending > budget.maximum:
        raise ValueError("Four measurements exceed the network budget; no measurement bytes downloaded")
    output_dir.mkdir(parents=True)
    index = {"schema": SCHEMA, "status": "started", "created_at": datetime.now(timezone.utc).isoformat(), "source_plan": source_plan, "input_stac": {scene.name: scene.record() for scene in scenes}, "not_original_esa_safe": True, "source_scope": "public Sentinel-1 IW GRD measurements and acquisition/calibration/noise metadata only; no published hazard labels or damage input", "scenes": {}}
    json_write(output_dir / "acquisition-start.json", index)
    for scene in scenes:
        item_copy = output_dir / "metadata" / f"{scene.name}.stac.json"
        item_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scene.item_path, item_copy)
        if digest(item_copy) != scene.item_sha256:
            raise ValueError("STAC changed after acquisition inputs were validated")
        original, manifest_receipt, receipt_path = obtain(scene.manifest_url, cache_dir / scene.item_id / "manifest.safe", METADATA_CAP)
        original_copy = output_dir / "metadata" / f"{scene.name}.original-manifest.safe"
        shutil.copyfile(original, original_copy)
        root, objects = manifest_objects(original_copy.read_bytes(), scene)
        derived = output_dir / "derived-safe" / (scene.product_identifier + ".SAFE")
        derived.mkdir(parents=True)
        files = []
        for obj in objects:
            cap = measurement_cap if obj["kind"] == "measurement" else METADATA_CAP
            source, receipt, receipt_path = obtain(obj["public_url"], cache_dir / scene.item_id / obj["source_relative"], cap)
            if obj["kind"] == "measurement":
                pol = "vv" if obj["source_relative"].endswith("iw-vv.tiff") else "vh"
                if any(receipt[key] != raw_headers[(scene.name, pol)][key] for key in ("bytes", "etag", "public_url")):
                    raise ValueError("Measurement source changed after acquisition preflight")
            if obj["kind"] in {"annotation", "noise", "calibration", "rfi"}:
                validate_metadata_component(source.read_bytes(), scene, obj["source_relative"])
            target = derived / obj["alias"]
            if not target.resolve().is_relative_to(derived.resolve()):
                raise ValueError("Manifest path escapes derivative")
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
                linkage = "hardlink; exact provider bytes"
            except OSError:
                shutil.copyfile(source, target)
                linkage = "copy; exact provider bytes"
            sha = digest(target)
            if sha != receipt["sha256"]:
                raise ValueError("Derived alias bytes differ from acquired source")
            actual_md5 = digest(target, "md5")
            obj["stream"].set("size", str(receipt["bytes"]))
            obj["checksum_element"].set("checksumName", "MD5")
            obj["checksum_element"].text = actual_md5
            files.append({key: value for key, value in obj.items() if key not in {"stream", "checksum_element"}} | {"bytes": receipt["bytes"], "sha256": sha, "derived_md5": actual_md5, "provider_etag": receipt["etag"], "receipt_path": str(receipt_path.resolve()), "receipt_sha256": digest(receipt_path), "source_cache_path": str(source.resolve()), "alias_linkage": linkage, "original_manifest_checksum_verifies_provider_bytes": False, "source_identity": "provider COG/renamed asset in explicitly derived wrapper"})
        manifest = derived / "manifest.safe"
        ET.ElementTree(root).write(manifest, encoding="utf-8", xml_declaration=True)
        report = {"source_item_id": scene.item_id, "scene": scene.name, "product_identifier": scene.product_identifier, "datetime": scene.acquisition_time, "input_stac": {"path": str(item_copy.resolve()), "sha256": scene.item_sha256}, "derived_safe": str(derived.resolve()), "derived_manifest": str(manifest.resolve()), "derived_manifest_sha256": digest(manifest), "original_manifest": str(original_copy.resolve()), "original_manifest_sha256": digest(original_copy), "original_manifest_receipt_path": str((cache_dir / scene.item_id / "manifest.safe.receipt.json").resolve()), "original_manifest_receipt_sha256": digest(cache_dir / scene.item_id / "manifest.safe.receipt.json"), "not_original_esa_safe": True, "purpose": "SNAP reader wrapper around exact acquired provider bytes; original ESA source manifest/checksums retained separately and not used to assert provider-COG byte identity", "files": files}
        provenance = derived / "derived-wrapper-provenance.json"
        json_write(provenance, report)
        index["scenes"][scene.name] = report | {"provenance_path": str(provenance.resolve()), "provenance_sha256": digest(provenance)}
    index["status"] = "completed"
    index["completed_at"] = datetime.now(timezone.utc).isoformat()
    index["network"] = {"maximum_body_bytes": budget.maximum, "actual_received_body_bytes": budget.transferred, "requests": budget.requests, "measurement_individual_cap_bytes": measurement_cap, "metadata_individual_cap_bytes": METADATA_CAP, "four_measurement_source_bytes": pending, "measurement_advertised_network_bytes": download_pending, "explicit_imported_assets": len(imported_sources), "imported_asset_remote_revalidation": False if imported_sources else None, "account_required": False, "tls_verified": True, "redirects_followed": False, "etag_scope": "provider transport identity; ETag is not asserted to be a cryptographic content digest", "sha256_scope": "locally acquired bytes, preserved receipts, and exact derived aliases"}
    json_write(output_dir / "derived-wrapper-index.json", index)
    return index


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def scenes_from_plan(plan_path: Path) -> tuple[list[Scene], dict]:
    if EXCLUDED.search(plan_path.as_posix()):
        raise ValueError("Reference/damage documents are not acquisition plans")
    plan_path = plan_path.resolve(strict=True)
    plan = json_read(plan_path)
    if plan.get("reference_labels_loaded") is True or plan.get("additional_inputs"):
        raise ValueError("Reference labels/additional inputs are not acquisition-plan inputs")
    pair = plan.get("sentinel1", {}).get("selected_pair")
    if not isinstance(pair, dict):
        raise ValueError("Planner has no selected raw Sentinel-1 pair")
    scenes = []
    for name in ("before", "after"):
        record = pair.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("item_path"), str):
            raise ValueError("Planner must pin saved STAC item paths and checksums")
        path = Path(record["item_path"])
        if not path.is_absolute():
            path = plan_path.parent / path
        scene = validate_scene(name, path, record.get("item_sha256"))
        if not record.get("item_sha256") or record.get("item_id") != scene.item_id or record.get("datetime") != scene.acquisition_time or record.get("catalogue_item_canonical_sha256") != canonical_sha(scene.item):
            raise ValueError("Planner's selected item identity/checksums do not match saved STAC")
        scenes.append(scene)
    validate_pair(*scenes)
    if pair.get("relative_orbit") != scenes[0].item["properties"]["sat:relative_orbit"] or pair.get("orbit_state") != scenes[0].item["properties"]["sat:orbit_state"]:
        raise ValueError("Planner's pair geometry identity differs from selected STAC")
    return scenes, {"path": str(plan_path), "sha256": digest(plan_path), "case_id": plan.get("case", {}).get("case_id"), "bbox": plan.get("case", {}).get("bbox"), "context_bbox": plan.get("case", {}).get("context_bbox"), "scope": "selected raw Sentinel-1 pair only; other planned modalities are not acquired by this command"}


def inspect_existing_native_cache(scenes: list[Scene], source_cache: Path) -> dict[str, dict]:
    """Verify every old source binding before any same-product local import.

    This supports the earlier, case-specific index solely as an explicit cache
    import. It does not convert its records into proof of a new HTTPS download.
    No new remote state is asserted for an offline imported asset.
    """
    source_cache = source_cache.resolve(strict=True)
    index_path = source_cache / "derived-wrapper-index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("Source cache requires a regular derived-wrapper index")
    value = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Explicit old native-cache index must contain two scene records")
    by_scene = {record.get("scene"): record for record in value if isinstance(record, dict)}
    if set(by_scene) != {"before", "after"}:
        raise ValueError("Source cache scene identity is ambiguous")
    result = {}
    for scene in scenes:
        record = by_scene[scene.name]
        if record.get("source_item_id") != scene.item_id or record.get("not_original_esa_safe") is not True:
            raise ValueError("Source cache product identity differs from selected fresh item")
        old_item = validate_scene(scene.name, source_cache / "metadata" / f"{scene.name}.stac.json")
        if canonical_sha(old_item.item) != canonical_sha(scene.item):
            raise ValueError("Source cache STAC identity differs from selected fresh item")
        original = source_cache / "metadata" / f"{scene.name}.original-manifest.safe"
        if original.is_symlink() or digest(original) != record.get("original_manifest_sha256"):
            raise ValueError("Source cache original manifest checksum differs")
        _, objects = manifest_objects(original.read_bytes(), scene)
        folder = source_cache / "derived-safe" / (scene.product_identifier + ".SAFE")
        if folder.is_symlink() or digest(folder / "manifest.safe") != record.get("derived_manifest_sha256"):
            raise ValueError("Source cache derived manifest checksum differs")
        stored = record.get("files")
        if not isinstance(stored, list) or len(stored) != len(objects):
            raise ValueError("Source cache file binding list is incomplete")
        old_files = {entry.get("alias"): entry for entry in stored if isinstance(entry, dict)}
        if len(old_files) != len(objects):
            raise ValueError("Source cache contains duplicate aliases")
        for obj in objects:
            entry = old_files.get(obj["alias"], {})
            source = folder / obj["alias"]
            if source.is_symlink() or not source.resolve().is_relative_to(folder.resolve()):
                raise ValueError("Source cache alias is a symlink or escapes its wrapper")
            expected_url = public_url(entry.get("public_url"), f"{scene.prefix}/{obj['source_relative']}")
            if source.stat().st_size != entry.get("bytes") or digest(source) != entry.get("sha256") or digest(source, "md5") != entry.get("derived_md5"):
                raise ValueError("Source cache object bytes differ from its SHA/MD5/size binding")
            if obj["kind"] in {"annotation", "noise", "calibration", "rfi"}:
                validate_metadata_component(source.read_bytes(), scene, obj["source_relative"])
            if str(entry.get("original_manifest_size")) != str(obj["original_manifest_size"]) or entry.get("original_manifest_checksum") != obj["original_manifest_checksum"]:
                raise ValueError("Source cache original checksum preservation differs")
            result[expected_url] = {"path": str(source.resolve()), "bytes": entry["bytes"], "sha256": entry["sha256"], "source_cache_index": str(index_path), "source_cache_index_sha256": digest(index_path), "source_cache_original_manifest_sha256": digest(original), "source_cache_stac_sha256": old_item.item_sha256}
        result[scene.manifest_url] = {"path": str(original.resolve()), "bytes": original.stat().st_size, "sha256": digest(original), "source_cache_index": str(index_path), "source_cache_index_sha256": digest(index_path), "source_cache_original_manifest_sha256": digest(original), "source_cache_stac_sha256": old_item.item_sha256}
    return result


def import_source(url: str, target: Path, source_record: dict) -> tuple[Path, dict, Path]:
    source = Path(source_record["path"])
    if source.is_symlink() or source.stat().st_size != source_record["bytes"] or digest(source) != source_record["sha256"]:
        raise ValueError("Explicit imported source changed after cache verification")
    receipt_path = target.with_name(target.name + ".receipt.json")
    if target.exists() or receipt_path.exists() or target.is_symlink() or receipt_path.is_symlink():
        if not target.is_file() or target.is_symlink() or not receipt_path.is_file() or receipt_path.is_symlink():
            raise ValueError("Import cache bytes/receipt must both be regular files")
        receipt = json_read(receipt_path)
        if receipt.get("public_url") != url or receipt.get("sha256") != source_record["sha256"] or receipt.get("bytes") != source_record["bytes"] or digest(target) != source_record["sha256"]:
            raise ValueError("Existing imported cache identity differs")
        return target, {**receipt, "reused_from_cache": True}, receipt_path
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    if digest(target) != source_record["sha256"]:
        raise ValueError("Imported cache bytes differ from verified old source")
    receipt = {"public_url": url, "path": str(target.resolve()), "bytes": source_record["bytes"], "sha256": source_record["sha256"], "etag": None, "transport": "explicit offline cache import; no new remote identity/download claim", "imported_at": datetime.now(timezone.utc).isoformat(), "import_origin": source_record, "import_verification": "original and derived manifests, selected STAC, every alias SHA-256/MD5/size and original-checksum preservation verified", "remote_revalidated": False}
    json_write(receipt_path, receipt)
    return target, {**receipt, "reused_from_cache": False}, receipt_path
