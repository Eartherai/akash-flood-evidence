import copy
import pytest
from shapely.geometry import box, mapping
from akash_flood.catalog import select_s1_pair, validate_bbox


def scene(name, day, orbit=85):
    return {"id": name, "geometry": mapping(box(85, 28, 86, 29)),
            "properties": {"datetime": f"2026-08-{day:02d}T12:00:00Z", "sat:relative_orbit": orbit,
                           "sat:orbit_state": "ascending", "sar:instrument_mode": "IW",
                           "sar:polarizations": ["VV", "VH"]}}


def test_matching_orbit_and_event_boundary():
    items = [scene("pre", 16), scene("post", 28), scene("wrong_orbit", 27, 19), scene("same_day", 26)]
    pair = select_s1_pair(items, "2026-08-26", [85.3, 28.1, 85.4, 28.3])
    assert pair["before"]["id"] == "pre"
    assert pair["after"]["id"] == "post"


def test_no_fallback_to_partial_footprint_or_unknown_orbit():
    before, after = scene("pre", 16), scene("post", 28)
    with pytest.raises(ValueError):
        select_s1_pair([before, after], "2026-08-26", [84, 28.1, 85.4, 28.3])
    after = copy.deepcopy(after)
    after["properties"].pop("sat:relative_orbit")
    with pytest.raises(ValueError):
        select_s1_pair([before, after], "2026-08-26", [85.3, 28.1, 85.4, 28.3])


def test_invalid_bbox():
    with pytest.raises(ValueError):
        validate_bbox([86, 28, 85, 29])
