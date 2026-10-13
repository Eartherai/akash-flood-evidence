"""Composite dates are chosen by SCL and cannot change after spectral outcomes."""
import time

import numpy as np
import pytest

from akash_flood.optical_composite import (POLICY, PUBLIC_HOST, AuditedCompositeRangeHTTP,
    select_source_index, install_selected_reflectance, screen_composite, validate_asset_url)
from akash_flood.optical_change import POLICY as SINGLE_POLICY


def scene(shape, nir=.6, red=.1, swir16=.1, blue=.1):
    return {key: np.full(shape, value, dtype="float32")
            for key, value in (("nir", nir), ("red", red), ("swir16", swir16), ("blue", blue))}


def test_nearest_scl_source_wins_and_spectrum_cannot_reselect_date():
    dims = (25, 25)
    aoi = np.ones(dims, bool)
    scl = np.full(dims, 4, dtype="uint8")
    selected = np.zeros(dims, dtype="uint8")
    select_source_index(selected, scl, aoi, 1)
    select_source_index(selected, scl, aoi, 2)
    assert selected[12, 12] == 1
    composed = {key: np.full(dims, np.nan, dtype="float32") for key in ("blue", "red", "nir", "swir16")}
    available = np.zeros(dims, bool)
    invalid = scene(dims)
    invalid["blue"][12, 12] = np.nan
    install_selected_reflectance(composed, available, 1, selected, invalid, scl, aoi)
    install_selected_reflectance(composed, available, 2, selected, scene(dims), scl, aoi)
    assert not available[12, 12] and np.isnan(composed["nir"][12, 12])
    assert selected[12, 12] == 1
    assert not available[12, 15] and available[12, 16]


def test_clouded_nearest_source_allows_older_scl_source_only():
    dims = (25, 25)
    aoi = np.ones(dims, bool)
    selected = np.zeros(dims, dtype="uint8")
    clouded = np.full(dims, 4, dtype="uint8")
    clouded[:, :12] = 9
    select_source_index(selected, clouded, aoi, 1)
    select_source_index(selected, np.full(dims, 4, dtype="uint8"), aoi, 2)
    assert selected[12, 6] == 2 and selected[12, 18] == 1 and selected[0, 0] == 0


def test_both_dates_need_valid_acquired_reflectance_and_thresholds_are_unchanged():
    for key, value in POLICY["surface_change_thresholds"].items():
        assert value == SINGLE_POLICY[key]
    dims = (25, 25)
    aoi = np.ones(dims, bool)
    before = scene(dims)
    after = scene(dims, nir=.1, red=.2, swir16=.6, blue=.1)
    available = np.ones(dims, bool)
    after_available = available.copy()
    after_available[12, 12] = False
    labels, assessed, _, counts = screen_composite(before, after, available, after_available, aoi)
    assert labels[12, 12] == 255 and not assessed[12, 12]
    assert labels[12, 13] == 1
    assert counts["unknown_pixels"] == 1 and counts["candidate_pixels"] == 624


URL = "https://" + PUBLIC_HOST + "/sentinel-2-c1-l2a/45/R/UM/2026/8/a/B02.tif"


class Response:
    status_code = 206
    headers = {"Content-Range": "bytes 0-3/100", "ETag": "fixed", "Content-Length": "4"}

    def __init__(self, fail=False, delay=False):
        self.fail, self.delay = fail, delay

    def iter_content(self, size):
        if self.delay:
            time.sleep(.02)
        yield b"ab" if self.fail else b"abcd"
        if self.fail:
            raise OSError("interrupted body")

    def close(self):
        pass


class Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_transport_exact_allowlist_body_budget_and_failed_partial_accounting():
    session = Session(Response(fail=True))
    client = AuditedCompositeRangeHTTP([URL], max_bytes=4, session=session)
    with pytest.raises(OSError):
        client.range(URL, 0, 4)
    audit = client.audit()
    assert audit["received_body_bytes"] == 2 and audit["failed_get_count"] == 1
    assert session.calls[0]["verify"] is True and session.calls[0]["allow_redirects"] is False
    with pytest.raises(ValueError, match="reservation"):
        client.range(URL, 0, 4)
    assert len(session.calls) == 1
    with pytest.raises(ValueError, match="allowlist"):
        client.range(URL.replace("B02", "B04"), 0, 1)


def test_deadline_includes_delivered_chunk_in_transport_ledger():
    client = AuditedCompositeRangeHTTP([URL], session=Session(Response(delay=True)), deadline=time.perf_counter() + .01)
    with pytest.raises(TimeoutError):
        client.range(URL, 0, 4)
    assert client.audit()["received_body_bytes"] == 4
    assert client.audit()["failed_get_count"] == 1


@pytest.mark.parametrize("url", [URL + "?token=secret", URL.replace("https:", "http:"), URL.replace("B02", "labels"), URL.replace(PUBLIC_HOST, "example.com")])
def test_unfrozen_or_nonpublic_assets_are_refused(url):
    with pytest.raises(ValueError):
        validate_asset_url(url)
