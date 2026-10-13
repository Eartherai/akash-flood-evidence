import pytest
from shapely.geometry import box, mapping, shape

from akash_flood.buildings import building_footprints, intersect_buildings


def fixture():
    coords = [(0, 0), (.004, 0), (.004, .004), (0, .004),
              (.001, .001), (.003, .001), (.003, .003), (.001, .003)]
    nodes = [dict(type="node", id=i, lon=x, lat=y) for i, (x, y) in enumerate(coords, 1)]
    return nodes + [
        dict(type="way", id=10, nodes=[1, 2, 3], tags={"building": "yes"}),
        dict(type="way", id=11, nodes=[1, 4, 3]),  # reverse member order
        dict(type="way", id=12, nodes=[5, 6, 7, 8, 5]),
        dict(type="relation", id=20, tags={"type": "multipolygon", "building": "yes"}, members=[
            dict(type="way", ref=10, role="outer"), dict(type="way", ref=11, role="outer"),
            dict(type="way", ref=12, role="inner")])]


def hazard(geometry):
    return dict(type="FeatureCollection", features=[dict(type="Feature", geometry=mapping(geometry), properties={})])


def test_relation_joins_reverse_members_preserves_hole_and_deduplicates():
    footprints, audit = building_footprints({"elements": fixture()})
    assert len(footprints["features"]) == 1
    geometry = shape(footprints["features"][0]["geometry"])
    assert len(geometry.interiors) == 1
    assert audit["suppressed_relation_members"] == ["way/10"]
    assert not audit["excluded"]
    assert not intersect_buildings(footprints, hazard(box(.0012, .0012, .0028, .0028)), 32631)["features"]
    exposed = intersect_buildings(footprints, hazard(box(-.001, -.001, .005, .005)), 32631)
    assert exposed["features"][0]["properties"]["overlap_fraction"] == pytest.approx(1)


def test_missing_relation_member_rejects_partial_geometry():
    elements = [e for e in fixture() if (e["type"], e["id"]) != ("way", 11)]
    footprints, audit = building_footprints({"elements": elements})
    assert not footprints["features"]
    assert audit["excluded"][0]["osm_id"] == "relation/20"


def test_open_way_and_point_buildings_are_audited():
    elements = fixture()[:4] + [dict(type="way", id=30, nodes=[1, 2, 3], tags={"building": "yes"}),
                               dict(type="node", id=40, lon=1, lat=1, tags={"building": "yes"})]
    footprints, audit = building_footprints({"elements": elements})
    assert not footprints["features"] and len(audit["excluded"]) == 2


def test_boundary_contact_is_not_exposure_and_degrees_are_rejected():
    elements = fixture()[:4] + [dict(type="way", id=30, nodes=[1, 2, 3, 4, 1], tags={"building": "yes"})]
    footprints, _ = building_footprints({"elements": elements})
    assert not intersect_buildings(footprints, hazard(box(.004, 0, .008, .004)), 32631)["features"]
    with pytest.raises(ValueError, match="metre"):
        intersect_buildings(footprints, hazard(box(0, 0, 1, 1)), 4326)
