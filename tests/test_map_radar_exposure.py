"""Synthetic spatial associations, coverage gaps and provenance; no event QA."""
import importlib.util
import json
from pathlib import Path

from affine import Affine
import numpy as np
from pyproj import CRS, Transformer
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, shape


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/map_radar_exposure.py"
SPEC = importlib.util.spec_from_file_location("map_radar_exposure", SCRIPT)
mapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mapper)
GRID = (CRS.from_epsg(32645), from_origin(500000, 3100020, 10, 10), 2, 2)


def arrays():
    return {"aoi": np.ones((2, 2), dtype="uint8"),
            "assessed": np.array([[1, 0], [1, 0]], dtype="uint8"),
            "change": np.array([[1, 255], [0, 255]], dtype="uint8"),
            "water": np.array([[0, 255], [0, 255]], dtype="uint8")}


def osm_fixture():
    inverse = Transformer.from_crs(GRID[0], 4326, always_xy=True)
    elements = []

    def node(identity, x, y, **extra):
        lon, lat = inverse.transform(x, y)
        elements.append(dict(type="node", id=identity, lon=lon, lat=lat, **extra))

    node(1, 499995, 3100015)
    node(2, 500025, 3100015)
    node(3, 500100, 3100015)
    node(4, 500130, 3100015)
    elements.extend([
        dict(type="way", id=10, nodes=[1, 2], tags={"highway": "secondary", "bridge": "yes"}),
        dict(type="way", id=20, nodes=[1, 2], tags={"highway": "secondary", "oneway": "yes"}),
        dict(type="way", id=30, nodes=[3, 4], tags={"highway": "service", "oneway": "-1"}),
    ])
    for number, polygon in enumerate((box(500005, 3100005, 500015, 3100015),
                                     box(500002, 3100002, 500004, 3100004),
                                     box(500015, 3100002, 500018, 3100005),
                                     box(500100, 3100002, 500104, 3100006))):
        coordinates = list(polygon.exterior.coords)[:-1]
        ids = [100 + number * 10 + index for index in range(len(coordinates))]
        for identity, coordinate in zip(ids, coordinates):
            node(identity, *coordinate)
        elements.append(dict(type="way", id=1000 + number, nodes=ids + [ids[0]], tags={"building": "yes"}))
    # Incomplete and point-only source buildings must stay in the exclusion audit.
    elements.append(dict(type="way", id=2000, nodes=[100, 101, 102], tags={"building": "yes"}))
    node(2999, 500010, 3100010, tags={"building": "yes"})
    return {"elements": elements}


def collections(buffer=10):
    masks, polygons, geometries = mapper.mask_geometries(arrays(), GRID)
    return mapper.overlap_features(osm_fixture(), GRID, geometries, road_buffer_m=buffer)


def test_crossing_centerline_and_explicit_metric_corridor_preserve_unknowns():
    result, audit = collections()
    roads = {feature["id"]: feature for feature in result["roads"]["features"]}
    crossing = roads["way-10-nodes-1-2"]["properties"]
    centerline = crossing["centerline_coverage"]
    assert centerline["total"] == pytest.approx(30, abs=1e-5)
    assert centerline["change_candidate"] == pytest.approx(10, abs=1e-5)
    assert centerline["unassessed"] == pytest.approx(10, abs=1e-5)
    assert centerline["outside_aoi"] == pytest.approx(10, abs=1e-5)
    corridor = crossing["coverage"]
    assert corridor["measure"] == "area_m2"
    assert corridor["total"] == pytest.approx(600, abs=1e-5)
    assert corridor["assessed"] == pytest.approx(150, abs=1e-5)
    assert corridor["unassessed"] == pytest.approx(150, abs=1e-5)
    assert corridor["outside_aoi"] == pytest.approx(300, abs=1e-5)
    assert corridor["change_candidate"] == pytest.approx(100, abs=1e-5)
    assert corridor["water_candidate"] == 0
    assert not corridor["fully_assessed_no_water_candidate"]
    assert crossing["road_buffer_half_width_m"] == 10
    outside = roads["way-30-nodes-3-4"]["properties"]["coverage"]
    assert outside["status"] == "outside_aoi"
    assert outside["assessed"] == outside["change_candidate"] == outside["water_candidate"] == 0
    assert not outside["fully_assessed_no_change_candidate"]


def test_directed_dedup_keeps_parallel_ways_and_each_original_direction():
    result, audit = collections(0)
    roads = {feature["id"]: feature["properties"] for feature in result["roads"]["features"]}
    assert len(roads) == 3
    assert roads["way-10-nodes-1-2"]["source_directed_edges"] == [
        {"edge_id": "10:0:f", "from_node_id": 1, "to_node_id": 2},
        {"edge_id": "10:0:r", "from_node_id": 2, "to_node_id": 1}]
    assert roads["way-20-nodes-1-2"]["source_directed_edge_count"] == 1
    assert roads["way-30-nodes-3-4"]["source_directed_edges"] == [
        {"edge_id": "30:0:r", "from_node_id": 4, "to_node_id": 3}]
    assert audit["roads"]["source_directed_edges"] == 4
    assert roads["way-10-nodes-1-2"]["coverage"]["measure"] == "length_m"
    assert not result["road-corridors"]["features"]
    assert len(result["bridges"]["features"]) == 1
    bridges = mapper.summarize_features(result["bridges"])
    assert bridges["features_with_change_candidate_overlap"] == 1
    assert bridges["features_with_water_candidate_overlap"] == 0
    assert not any(feature["properties"]["road_closure_assigned"] for feature in result["roads"]["features"])


def test_partial_building_unknown_is_not_fully_assessed_negative():
    result, audits = collections()
    buildings = {feature["id"]: feature["properties"] for feature in result["buildings"]["features"]}
    partial = buildings["way/1000"]["coverage"]
    assert partial["total"] == pytest.approx(100, abs=1e-5)
    assert partial["assessed"] == pytest.approx(50, abs=1e-5)
    assert partial["unassessed"] == pytest.approx(50, abs=1e-5)
    assert partial["change_candidate"] == pytest.approx(25, abs=1e-5)
    assert partial["assessed_no_water_candidate"] == pytest.approx(50, abs=1e-5)
    assert not partial["fully_assessed_no_water_candidate"]
    assert buildings["way/1001"]["coverage"]["fully_assessed_no_change_candidate"]
    assert buildings["way/1002"]["coverage"]["status"] == "within_aoi_fully_unassessed"
    assert buildings["way/1003"]["coverage"]["status"] == "outside_aoi"
    summary = mapper.summarize_features(result["buildings"])
    assert summary["mapped_feature_count"] == 4
    assert summary["features_with_unassessed_radar"] == 2
    assert summary["features_outside_aoi"] == 1
    assert summary["features_fully_assessed_no_change_candidate"] == 1
    assert summary["features_with_change_candidate_overlap"] == 1
    assert summary["features_with_water_candidate_overlap"] == 0
    assert len(audits["buildings"]["excluded"]) == 2
    assert all("new-water" not in properties["candidate_overlap_basis"]["change"] for properties in buildings.values())


def test_polygonization_preserves_holes_affine_cell_area_and_no_diagonal_merge():
    affine = Affine(10, 2, 500000, 1, -10, 3100020)
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    polygons = mapper.pixel_polygons(mask, affine)
    assert len(polygons) == 1 and len(polygons[0].interiors) == 1
    assert polygons[0].area == pytest.approx(8 * 102)
    diagonal = mapper.pixel_polygons(np.eye(2, dtype=bool), affine)
    assert len(diagonal) == 2
    assert sum(polygon.area for polygon in diagonal) == pytest.approx(2 * 102)
    data = {"aoi": mask.astype("uint8"), "assessed": mask.astype("uint8"),
            "change": mask.astype("uint8"), "water": np.zeros_like(mask, dtype="uint8")}
    grid = (GRID[0], affine, 3, 3)
    masks, parts, unions = mapper.mask_geometries(data, grid)
    collections, summary = mapper.candidate_features(parts, masks, grid)
    assert summary["change"]["candidate_area_m2"] == pytest.approx(816)
    assert len(shape(collections["change-candidates"]["features"][0]["geometry"]).interiors) == 1
    assert collections["water-candidates"]["features"] == []


def write_inputs(folder, data=None, *, shifted=None, crs=GRID[0]):
    folder.mkdir()
    for key, filename in mapper.RASTER_FILES.items():
        affine = GRID[1] if shifted != key else GRID[1] @ Affine.translation(.5, 0)
        with rasterio.open(folder / filename, "w", driver="GTiff", width=2, height=2,
                           count=1, dtype="uint8", crs=crs, transform=affine,
                           nodata=0 if key in ("aoi", "assessed") else 255) as dataset:
            dataset.write((data or arrays())[key], 1)


@pytest.mark.parametrize("mutation,match", [
    (lambda data: data["aoi"].__setitem__((0, 0), 0), "within the AOI"),
    (lambda data: data["assessed"].__setitem__((0, 1), 1), "candidate nodata"),
    (lambda data: data["change"].__setitem__((0, 1), 1), "assessed radar support"),
    (lambda data: data["water"].__setitem__((0, 0), 255), "candidate nodata"),
    (lambda data: data["aoi"].__setitem__((1, 1), 2), "encoding"),
])
def test_inconsistent_support_and_encoding_rejected(tmp_path, mutation, match):
    data = arrays()
    mutation(data)
    folder = tmp_path / "synthetic-inputs"
    write_inputs(folder, data)
    with pytest.raises(ValueError, match=match):
        mapper.read_evidence(folder)


def test_grid_shift_and_nonmetric_crs_are_rejected(tmp_path):
    shifted = tmp_path / "shifted"
    write_inputs(shifted, shifted="water")
    with pytest.raises(ValueError, match="same grid"):
        mapper.read_evidence(shifted)
    degrees = tmp_path / "degrees"
    write_inputs(degrees, crs=4326)
    with pytest.raises(ValueError, match="metre"):
        mapper.read_evidence(degrees)


def test_fresh_synthetic_run_hashes_every_source_and_saved_output(tmp_path):
    folder, osm_path, output = tmp_path / "synthetic-inputs", tmp_path / "osm.json", tmp_path / "fresh-output"
    write_inputs(folder)
    osm_path.write_text(json.dumps(osm_fixture()))
    summary = mapper.run(folder, osm_path, output)
    assert summary["coverage_pixels"] == {"aoi": 4, "assessed": 2, "unassessed": 2}
    assert summary["road_buffer"]["half_width_m"] == 10
    assert summary["candidate_polygons"]["change"]["candidate_area_m2"] == 100
    assert summary["reference_labels_loaded"] is False
    assert summary["road_closures_assigned"] is False
    manifest = json.loads((output / "source-and-output-manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["inputs"]) == 5
    for path, record in manifest["inputs"].items():
        assert mapper.sha256(path) == record["sha256"]
    for name, record in manifest["outputs"].items():
        assert mapper.sha256(output / name) == record["sha256"]
    original = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError, match="fresh"):
        mapper.run(folder, osm_path, output)
    assert (output / "summary.json").read_bytes() == original


def test_official_evaluation_path_is_rejected_without_reading(tmp_path):
    with pytest.raises(ValueError, match="evaluation"):
        mapper.read_evidence(tmp_path / "evaluation-only" / "EMSR927")
