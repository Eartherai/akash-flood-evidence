"""Bounded Sentinel-1 GRD access and native-coordinate sigma0 calibration.

This module does not terrain correct, co-register, denoise, segment floods, or
infer road passability. Source ground control points locate an approximate native
window; a STAC bounding-box affine is never treated as measurement geolocation.
"""

from __future__ import annotations

import io
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from rasterio.control import GroundControlPoint
from rasterio.transform import GCPTransformer
from rasterio.windows import Window


class DownloadBudgetExceeded(RuntimeError):
    pass


class BoundedHTTP:
    """Unsigned public reads with an enforced cumulative response-byte budget."""

    def __init__(self, max_bytes=50_000_000, timeout=45):
        self.max_bytes = int(max_bytes)
        self.timeout = timeout
        self.bytes_read = 0
        self.requests = []

    def _available(self, count):
        if count < 0 or self.bytes_read + count > self.max_bytes:
            raise DownloadBudgetExceeded("Public data response would exceed the byte budget")

    def get(self, url, max_response_bytes=2_000_000):
        self._available(max_response_bytes)
        request = urllib.request.Request(url, headers={"User-Agent": "Akash-ReadOnlyCropProof/1.0"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > max_response_bytes:
                raise DownloadBudgetExceeded("Metadata response exceeds its individual cap")
            data = response.read(max_response_bytes + 1)
            self.bytes_read += len(data)
            if len(data) > max_response_bytes or self.bytes_read > self.max_bytes:
                raise DownloadBudgetExceeded("Metadata response exceeds the byte budget")
            self.requests.append({"url": url, "status": response.status, "bytes": len(data), "kind": "metadata"})
            return data

    def range(self, url, start, stop):
        """Read [start, stop); require a byte-range response, never full-scene fallback."""
        count = stop - start
        if start < 0 or count <= 0 or count > 4_000_000:
            raise ValueError("Invalid or excessively large range request")
        self._available(count)
        request = urllib.request.Request(url, headers={
            "Range": f"bytes={start}-{stop - 1}", "User-Agent": "Akash-ReadOnlyCropProof/1.0",
        })
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if response.status != 206 or not match:
                raise RuntimeError("Server did not honor the requested range; full-scene fallback refused")
            first, last, total = map(int, match.groups())
            if first != start or last + 1 != min(stop, total):
                raise RuntimeError("Content-Range differs from requested source bytes")
            data = response.read(count + 1)
            self.bytes_read += len(data)
            if len(data) != last - first + 1 or self.bytes_read > self.max_bytes:
                raise RuntimeError("Incomplete, oversized, or over-budget range response")
            self.requests.append({"url": url, "status": 206, "start": first, "stop": last + 1,
                                  "bytes": len(data), "file_bytes": total, "kind": "raster_range"})
            return data, total


class HTTPRangeReader(io.RawIOBase):
    """Seekable bounded public range reader for Rasterio's custom opener."""

    def __init__(self, url, client):
        super().__init__()
        self.name = url
        self.client = client
        self.position = 0
        first, self.length = client.range(url, 0, 65536)
        self.cache = [(0, len(first), first)]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset if whence == io.SEEK_SET else self.position + offset if whence == io.SEEK_CUR else self.length + offset
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END) or position < 0:
            raise ValueError("Invalid seek")
        self.position = int(position)
        return self.position

    def read(self, size=-1):
        if size is None or size < 0:
            raise RuntimeError("Unbounded raster reads are refused")
        if size == 0 or self.position >= self.length:
            return b""
        start, stop = self.position, min(self.position + size, self.length)
        for first, last, data in self.cache:
            if first <= start and stop <= last:
                answer = data[start - first:stop - first]
                self.position = stop
                return answer
        answer, total = self.client.range(self.name, start, stop)
        if total != self.length:
            raise RuntimeError("Remote source length changed during range reads")
        self.cache.append((start, stop, answer))
        self.position = stop
        return answer

    def readinto(self, destination):
        data = self.read(len(destination))
        destination[:len(data)] = data
        return len(data)


class SingleRasterOpener:
    """Allow one canonical raster, refusing discovery/download of sidecar files."""

    def __init__(self, url, client):
        self.url = url
        self.client = client

    def __call__(self, path, mode="rb"):
        if path != self.url or mode not in ("r", "rb"):
            raise FileNotFoundError(path)
        return HTTPRangeReader(self.url, self.client)


def unsigned_s3_https(href):
    parsed = urllib.parse.urlsplit(href)
    if parsed.scheme != "s3" or parsed.netloc != "sentinel-s1-l1c" or parsed.query or parsed.fragment:
        raise ValueError("Expected a canonical unsigned Sentinel-1 public S3 asset")
    return "https://sentinel-s1-l1c.s3.eu-central-1.amazonaws.com" + parsed.path


def native_window_for_bbox(gcps, bbox, width, height, halo=8, samples=33):
    west, south, east, north = map(float, bbox)
    if not west < east or not south < north or halo < 0 or samples < 2:
        raise ValueError("Invalid AOI or boundary sampling settings")
    longitudes = np.linspace(west, east, samples)
    latitudes = np.linspace(south, north, samples)
    xs = np.concatenate([longitudes, longitudes, np.full(samples, west), np.full(samples, east)])
    ys = np.concatenate([np.full(samples, south), np.full(samples, north), latitudes, latitudes])
    with GCPTransformer(gcps) as transformer:
        rows, columns = transformer.rowcol(xs, ys, op=lambda values: values)
    rows, columns = np.asarray(rows), np.asarray(columns)
    if not np.isfinite(rows).all() or not np.isfinite(columns).all():
        raise ValueError("AOI could not be located using source ground control points")
    row0 = max(0, int(np.floor(rows.min())) - halo)
    col0 = max(0, int(np.floor(columns.min())) - halo)
    row1 = min(height, int(np.ceil(rows.max())) + halo + 1)
    col1 = min(width, int(np.ceil(columns.max())) + halo + 1)
    if row1 <= row0 or col1 <= col0 or (row1 - row0) > 2048 or (col1 - col0) > 2048:
        raise ValueError("AOI does not form a nonempty bounded native window")
    return Window(col0, row0, col1 - col0, row1 - row0)


def shifted_gcps(gcps, window):
    return [GroundControlPoint(row=g.row - window.row_off, col=g.col - window.col_off,
                               x=g.x, y=g.y, z=g.z, id=g.id, info=g.info) for g in gcps]


def gcp_records(gcps):
    return [{"row": float(g.row), "col": float(g.col), "x": float(g.x), "y": float(g.y),
             "z": float(g.z), "id": g.id, "info": g.info} for g in gcps]


@dataclass
class CalibrationLUT:
    lines: np.ndarray
    pixels: list[np.ndarray]
    sigma: list[np.ndarray]
    header: dict


def parse_calibration_xml(data):
    root = ET.fromstring(data)
    if root.tag != "calibration":
        raise ValueError("Expected calibration XML, not product or RFI annotation")
    vectors = root.findall("./calibrationVectorList/calibrationVector")
    declared = root.find("./calibrationVectorList")
    if declared is None or len(vectors) < 2 or int(declared.attrib.get("count", len(vectors))) != len(vectors):
        raise ValueError("Calibration vector list is incomplete")
    lines, pixels, sigma = [], [], []
    for vector in vectors:
        lines.append(float(vector.findtext("line")))
        p = np.fromstring(vector.findtext("pixel", ""), sep=" ", dtype=np.float64)
        s = np.fromstring(vector.findtext("sigmaNought", ""), sep=" ", dtype=np.float64)
        if len(p) < 2 or len(p) != len(s) or not np.all(np.diff(p) > 0) or not np.all(np.isfinite(s) & (s > 0)):
            raise ValueError("Invalid calibration pixel coordinates or sigma-nought LUT")
        for tag, values in [("pixel", p), ("sigmaNought", s)]:
            if int(vector.find(tag).attrib.get("count", len(values))) != len(values):
                raise ValueError("Calibration LUT count mismatch")
        pixels.append(p)
        sigma.append(s)
    lines = np.array(lines, dtype=np.float64)
    if not np.all(np.isfinite(lines)) or not np.all(np.diff(lines) > 0):
        raise ValueError("Calibration lines are not strictly increasing")
    header = {child.tag: child.text for child in root.findall("./adsHeader/*")}
    return CalibrationLUT(lines, pixels, sigma, header)


def interpolate_sigma_lut(lut, rows, columns):
    """Bilinear LUT interpolation at original zero-based measurement coordinates."""
    rows, columns = np.asarray(rows, dtype=np.float64), np.asarray(columns, dtype=np.float64)
    if rows.ndim != 1 or columns.ndim != 1 or not len(rows) or not len(columns):
        raise ValueError("Expected nonempty original row and column axes")
    if rows.min() < lut.lines[0] or rows.max() > lut.lines[-1]:
        raise ValueError("Native crop falls outside calibration line coverage")
    low = np.clip(np.searchsorted(lut.lines, rows, side="right") - 1, 0, len(lut.lines) - 2)
    needed = sorted(set(low.tolist()) | set((low + 1).tolist()))
    by_line = {}
    for i in needed:
        if columns.min() < lut.pixels[i][0] or columns.max() > lut.pixels[i][-1]:
            raise ValueError("Native crop falls outside calibration range coverage")
        by_line[i] = np.interp(columns, lut.pixels[i], lut.sigma[i])
    result = np.empty((len(rows), len(columns)), dtype=np.float64)
    for j, i in enumerate(low):
        fraction = (rows[j] - lut.lines[i]) / (lut.lines[i + 1] - lut.lines[i])
        result[j] = (1 - fraction) * by_line[i] + fraction * by_line[i + 1]
    return result


def calibrate_sigma0(raw_amplitude, sigma_lut, source_mask):
    raw = np.asarray(raw_amplitude)
    sigma_lut = np.asarray(sigma_lut, dtype=np.float64)
    if raw.shape != sigma_lut.shape or raw.shape != np.asarray(source_mask).shape:
        raise ValueError("Raw amplitude, mask and calibration grid shapes differ")
    valid = (np.asarray(source_mask) != 0) & (raw > 0) & np.isfinite(sigma_lut) & (sigma_lut > 0)
    power = np.full(raw.shape, np.nan, dtype=np.float64)
    power[valid] = np.square(raw[valid].astype(np.float64)) / np.square(sigma_lut[valid])
    db = np.full(raw.shape, np.nan, dtype=np.float64)
    db[valid] = 10 * np.log10(power[valid])
    return power, db, valid
