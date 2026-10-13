import networkx as nx
import pytest

from akash_flood.access import analyze_access, build_road_graph


def node(node_id, lon, lat=0):
    return {"type": "node", "id": node_id, "lon": lon, "lat": lat}


def way(way_id, node_ids, **tags):
    return {"type": "way", "id": way_id, "nodes": node_ids, "tags": {"highway": "residential", **tags}}


def point(point_id, lon, lat=0):
    return {"id": point_id, "name": point_id, "lon": lon, "lat": lat}


def graph(*elements):
    return build_road_graph({"elements": list(elements)})


def test_alternate_route_remains_and_reports_changed_path_without_mutating_graph():
    roads = graph(
        node(1, 0), node(2, 0.001), node(3, 0.002), node(4, 0.001, 0.002),
        way(10, [1, 2]), way(11, [2, 3]), way(12, [1, 4, 3]),
    )
    result = analyze_access(roads, [point("village", 0)], [point("hospital", 0.002)], {11})[0]
    assert result["status"] == "connected"
    assert result["before"][0]["path_way_ids"] == [10, 11]
    assert result["after"][0]["path_way_ids"] == [12, 12]
    assert result["after"][0]["distance_m"] > result["before"][0]["distance_m"]
    assert result["before"][0]["path_edge_ids"] == ["10:0:f", "11:0:f"]
    assert roads.number_of_edges() == 8
    result = analyze_access(roads, [point("village", 0)], [point("hospital", 0.002)], {11, 12})[0]
    assert result["status"] == "newly_disconnected"
    assert result["lost_destination_ids"] == ["hospital"]


@pytest.mark.parametrize(
    "tags,forward,reverse",
    [
        ({"oneway": "yes"}, True, False),
        ({"oneway": "-1"}, False, True),
        ({"junction": "roundabout"}, True, False),
        ({"junction": "roundabout", "oneway": "no"}, True, True),
    ],
)
def test_oneway_and_roundabout_direction(tags, forward, reverse):
    roads = graph(node(1, 0), node(2, 0.001), way(10, [1, 2], **tags))
    assert nx.has_path(roads, 1, 2) is forward
    assert nx.has_path(roads, 2, 1) is reverse
    if tags.get("oneway") == "-1":
        assert list(roads[2][1]["10:0:r"]["geometry"].coords) == [(0.001, 0), (0, 0)]
        result = analyze_access(roads, [point("s", 0)], [point("d", 0.001)], set())[0]
        assert result["status"] == "baseline_disconnected"


def test_parallel_ways_are_independent_and_report_actual_path_edges():
    roads = graph(node(1, 0), node(2, 0.001), way(10, [1, 2]), way(20, [1, 2]))
    assert roads.number_of_edges(1, 2) == 2
    result = analyze_access(roads, [point("s", 0)], [point("d", 0.001)], {10})[0]
    assert result["status"] == "connected"
    assert result["after"][0]["path_edge_ids"] == ["20:0:f"]
    assert result["after"][0]["path_way_ids"] == [20]


def test_baseline_disconnection_is_distinct_from_closure_effects():
    roads = graph(
        node(1, 0), node(2, 0.001), node(3, 0.02), node(4, 0.021),
        way(10, [1, 2]), way(20, [3, 4]),
    )
    result = analyze_access(roads, [point("s", 0)], [point("d", 0.021)], {10})[0]
    assert result["status"] == "baseline_disconnected"
    assert result["before"] == result["after"] == []
    assert result["lost_destination_ids"] == []


def test_geometric_crossings_do_not_connect_bridge_tunnel_or_other_layers():
    roads = graph(
        node(1, -0.001), node(2, 0), node(3, 0.001),
        node(4, 0, -0.001), node(5, 0), node(6, 0, 0.001),
        way(10, [1, 2, 3], bridge="yes", layer="1"),
        way(20, [4, 5, 6], tunnel="yes", layer="-1"),
    )
    assert not nx.has_path(roads, 1, 4)
    assert nx.has_path(roads, 1, 3)
    assert roads[1][2]["10:0:f"]["bridge"] is True
    assert roads[4][5]["20:0:f"]["tunnel"] is True
    assert roads[4][5]["20:0:f"]["layer"] == "-1"
    # A real shared endpoint still connects the bridge to its approach.
    connected = graph(node(1, 0), node(2, 0.001), node(3, 0.002),
                      way(10, [1, 2], bridge="yes", layer="1"), way(20, [2, 3]))
    assert nx.has_path(connected, 1, 3)


def test_snap_limit_uses_meters_for_settlements_and_destinations():
    roads = graph(node(1, 0), node(2, 0.001), way(10, [1, 2]))
    near, far = analyze_access(
        roads, [point("near", 0, 0.0009), point("far", 0, 0.01)],
        [point("d", 0.001)], set(), max_snap_m=150,
    )
    assert near["status"] == "connected"
    assert near["snap_distance_m"] == pytest.approx(100.0756, rel=1e-4)
    assert near["before"][0]["distance_m"] == pytest.approx(111.195, rel=1e-4)
    assert far["status"] == "unmapped"
    assert far["snap_node_id"] is None
    result = analyze_access(roads, [point("s", 0)], [point("far", 10)], set())[0]
    assert result["status"] == "unmapped"
    assert result["reason"] == "no_mapped_destinations"
    assert result["unmapped_destination_ids"] == ["far"]


def test_all_destinations_and_partial_loss_are_reported():
    roads = graph(node(1, 0), node(2, 0.001), node(3, 0.002),
                  way(10, [1, 2]), way(20, [2, 3]))
    result = analyze_access(
        roads, [point("s", 0)], [point("clinic", 0.001), point("hospital", 0.002)], {20}
    )[0]
    assert result["before_destination_ids"] == ["clinic", "hospital"]
    assert result["after_destination_ids"] == ["clinic"]
    assert result["status"] == "connected"
    assert result["partial_access_loss"] is True
    assert result["lost_destination_ids"] == ["hospital"]


def test_missing_nodes_break_way_and_empty_graph_is_unmapped():
    roads = graph(node(1, 0), node(3, 0.002), way(10, [1, 2, 3]))
    assert roads.number_of_edges() == 0
    assert roads.graph["missing_node_ids"] == [2]
    result = analyze_access(graph(), [point("s", 0)], [point("d", 0.001)], set())[0]
    assert result["status"] == "unmapped"
    assert result["snap_distance_m"] is None


def test_invalid_snap_limit_is_rejected():
    with pytest.raises(ValueError):
        analyze_access(graph(), [], [], set(), max_snap_m=-1)
