"""Synthetic provenance, unknown coverage and conditional graph proofs only."""
import copy
import json
from pathlib import Path
import sys

from affine import Affine
import numpy as np
from pyproj import CRS, Transformer
import pytest
import rasterio
from shapely.geometry import box

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import map_fresh_exposure as mapper

GRID = (CRS.from_epsg(32645), Affine(10, 0, 500005, 0, -10, 3100025), 3, 3)


def arrays():
    before = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 255]], dtype="uint8")
    after = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 255]], dtype="uint8")
    valid = before != 255
    return {"aoi": np.ones((3, 3), dtype="uint8"), "assessed": valid.astype("uint8"),
            "before": before, "after": after,
            "new": np.where(valid, (before == 0) & (after == 1), 255).astype("uint8"),
            "lost": np.where(valid, (before == 1) & (after == 0), 255).astype("uint8")}


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def fixture(tmp_path):
    inverse = Transformer.from_crs(GRID[0], 4326, always_xy=True)
    west, south = inverse.transform(500000, 3099990)
    east, north = inverse.transform(500040, 3100030)
    bbox = [west, south, east, north]
    case = {"case_id": "test-case", "bbox": bbox,
            "context_bbox": [west-.002, south-.002, east+.002, north+.002],
            "event_date": "2026-08-20", "osm_snapshot": "2026-07-01T00:00:00Z"}
    elements = []

    def node(identity, x, y, tags=None):
        lon, lat = inverse.transform(x, y)
        item = {"type": "node", "id": identity, "lon": lon, "lat": lat}
        if tags:
            item["tags"] = tags
        elements.append(item)

    node(1, 499975, 3100020, {"place": "village", "name": "Village"})
    node(2, 500060, 3100020, {"place": "town", "name": "Near town"})
    node(3, 499975, 3100220, {"amenity": "hospital", "name": "Far hospital"})
    elements.extend([{"type": "way", "id": 10, "nodes": [1, 2], "tags": {"highway": "secondary", "bridge": "yes"}},
                     {"type": "way", "id": 20, "nodes": [1, 2], "tags": {"highway": "secondary", "oneway": "yes"}},
                     {"type": "way", "id": 30, "nodes": [1, 3], "tags": {"highway": "secondary"}}])
    for index, polygon in enumerate((box(500007, 3100017, 500013, 3100023),
                                     box(500026, 3099997, 500031, 3100002),
                                     box(500100, 3100005, 500105, 3100010))):
        identities = []
        for coordinate_index, coordinate in enumerate(list(polygon.exterior.coords)[:-1]):
            identity = 100 + index * 10 + coordinate_index
            node(identity, *coordinate)
            identities.append(identity)
        elements.append({"type": "way", "id": 1000+index, "nodes": identities+[identities[0]], "tags": {"building": "yes"}})
    osm = {"elements": elements}
    osm_path = write_json(tmp_path / "osm.json", osm)
    provenance = {"snapshot": case["osm_snapshot"], "event_date": case["event_date"],
                  "bbox": [west-.01, south-.01, east+.01, north+.01],
                  "query": f'[out:json][date:"{case["osm_snapshot"]}"];out body;',
                  "element_count": len(elements), "sha256": mapper.sha256(osm_path),
                  "history_evidence": "synthetic fixture, not a historical source validation"}
    provenance_path = write_json(tmp_path / "osm.provenance.json", provenance)
    plan = {"case": case, "projection": {"epsg": 32645}, "osm": {
        "snapshot": case["osm_snapshot"], "status": "saved_historical_extract_checked",
        "local_inputs": [mapper.record(osm_path), mapper.record(provenance_path)]}}
    plan_path = write_json(tmp_path / "plan.json", plan)
    processing = {"schema": "akash-fresh-s1-processing-v1", "status": "completed", "completed_at": "2026-10-07T00:00:00Z",
                  "case": case, "source_plan": mapper.record(plan_path),
                  "scenes": {"before": {"status": "completed", "source_datetime": "2026-08-10T12:00:00Z"},
                             "after": {"status": "completed", "source_datetime": "2026-08-22T12:00:00Z"}},
                  "common_grid": {"crs": GRID[0].to_string(), "transform": list(GRID[1]), "shape": [3, 3], "resampling_used": False}}
    processing_path = write_json(tmp_path / "processing.json", processing)
    for name in ("policy.json", "freeze.json"):
        write_json(tmp_path / name, {"scope": "test fixture only; no actual model or inference"})
    model = tmp_path / "own-test-model.joblib"
    model.write_bytes(b"test fixture; never loaded or unpickled")
    folder = tmp_path / "water"
    folder.mkdir()
    data = arrays()
    for key, name in mapper.MASK_FILES.items():
        with rasterio.open(folder/name, "w", driver="GTiff", width=3, height=3, count=1, dtype="uint8",
                           transform=GRID[1], crs=GRID[0], nodata=0 if key in ("aoi", "assessed") else 255) as dataset:
            dataset.write(data[key], 1)
    for name in ("before-water-score.tif", "after-water-score.tif"):
        with rasterio.open(folder/name, "w", driver="GTiff", width=3, height=3, count=1, dtype="float32",
                           transform=GRID[1], crs=GRID[0], nodata=-9999) as dataset:
            dataset.write(np.full((3, 3), .5, dtype="float32"), 1)
    water = {"schema": "akash-fresh-water-v1", "available": True, "reference_labels_loaded": False,
             "case": case, "outputs": {path.name: mapper.record(path) for path in folder.iterdir()},
             "counts": mapper.validate_temporal_arrays(data), "code": [],
             "grid": {"crs": GRID[0].to_string(), "transform": list(GRID[1]), "shape": [3, 3],
                      "resolution_m": 10, "resampling_used_for_date_alignment": False},
             "model": {**mapper.record(model), "freeze": mapper.record(tmp_path/"freeze.json")},
             "policy": mapper.record(tmp_path/"policy.json"), "processing_record": mapper.record(processing_path),
             "source_dates": {scene: processing["scenes"][scene]["source_datetime"] for scene in ("before", "after")},
             "label_encoding": {"0": "predicted nonwater", "1": "predicted water", "255": "unknown"}}
    manifest_path = write_json(folder/"manifest.json", water)
    return manifest_path, plan_path, water, plan, osm


def test_all_eight_temporal_counts_recounted_and_unknown_not_nonwater():
    assert mapper.validate_temporal_arrays(arrays()) == {"aoi_pixels": 9, "assessed_pixels": 8, "unknown_pixels": 1,
        "before_water_pixels": 3, "after_water_pixels": 4, "new_water_candidate_pixels": 2,
        "lost_water_candidate_pixels": 1, "unchanged_water_pixels": 2}


@pytest.mark.parametrize("mutate,match", [
    (lambda value: value["after"].__setitem__((2, 2), 0), "joint assessment"),
    (lambda value: value["new"].__setitem__((0, 0), 0), "disagree"),
    (lambda value: value["lost"].__setitem__((0, 0), 1), "disagree"),
    (lambda value: value["before"].__setitem__((0, 0), 2), "encoding"),
    (lambda value: value["aoi"].__setitem__((0, 0), 0), "within a nonempty AOI"),
])
def test_invalid_candidate_encoding_and_temporal_domains_are_rejected(mutate, match):
    data = arrays()
    mutate(data)
    with pytest.raises(ValueError, match=match):
        mapper.validate_temporal_arrays(data)


def test_every_output_hash_is_checked_before_raster_decoding(tmp_path, monkeypatch):
    manifest, plan_path, water, plan, _ = fixture(tmp_path)
    score = Path(water["outputs"]["after-water-score.tif"]["path"])
    with score.open("ab") as stream:
        stream.write(b"changed score bytes")
    monkeypatch.setattr(mapper.rasterio, "open", lambda *args, **kwargs: pytest.fail("No raster decoding before all output bindings pass"))
    with pytest.raises(ValueError, match="checksum-bound water output"):
        mapper.read_water(manifest, plan_path, plan)


def test_exact_grid_and_all_eight_counts_are_checked(tmp_path):
    manifest, plan_path, water, plan, _ = fixture(tmp_path)
    assert mapper.read_water(manifest, plan_path, plan)[2] == water["counts"]
    water["counts"]["unchanged_water_pixels"] += 1
    write_json(manifest, water)
    with pytest.raises(ValueError, match="eight temporal counts"):
        mapper.read_water(manifest, plan_path, plan)
    water["counts"]["unchanged_water_pixels"] -= 1
    shifted = Path(water["outputs"]["after-water.tif"]["path"])
    with rasterio.open(shifted, "r+") as dataset:
        dataset.transform = GRID[1] @ Affine.translation(1, 0)
    water["outputs"]["after-water.tif"] = mapper.record(shifted)
    write_json(manifest, water)
    with pytest.raises(ValueError, match="exactly one native metric grid"):
        mapper.read_water(manifest, plan_path, plan)


def test_historical_source_cutoff_and_node_edits_checked(tmp_path):
    _, plan_path, _, plan, osm = fixture(tmp_path)
    assert mapper.read_historical_osm(plan_path, plan)[1]["snapshot"] == "2026-07-01T00:00:00Z"
    osm["elements"][0]["timestamp"] = "2026-07-02T00:00:00Z"
    osm_path = Path(plan["osm"]["local_inputs"][0]["path"])
    provenance_path = Path(plan["osm"]["local_inputs"][1]["path"])
    write_json(osm_path, osm)
    provenance = json.loads(provenance_path.read_text())
    provenance["sha256"] = mapper.sha256(osm_path)
    write_json(provenance_path, provenance)
    plan["osm"]["local_inputs"] = [mapper.record(osm_path), mapper.record(provenance_path)]
    with pytest.raises(ValueError, match="source edit newer"):
        mapper.read_historical_osm(plan_path, plan)
    plan["case"]["osm_snapshot"] = "2026-08-21T00:00:00Z"
    with pytest.raises(ValueError, match="strictly before"):
        mapper.read_historical_osm(plan_path, plan)


def test_new_lost_geometry_preserves_unknown_and_bidirectional_grouping(tmp_path):
    _, _, _, _, osm = fixture(tmp_path)
    masks, polygons, regions = mapper.mask_regions(arrays(), GRID)
    collections, audits, graph = mapper.overlap_features(osm, GRID, regions)
    roads = {feature["id"]: feature["properties"] for feature in collections["roads"]["features"]}
    assert len(roads) == 3
    assert roads["way-10-nodes-1-2"]["source_directed_edge_count"] == 2
    assert roads["way-20-nodes-1-2"]["source_directed_edge_count"] == 1
    assert roads["way-10-nodes-1-2"]["new_water_candidate_overlap"]
    assert roads["way-10-nodes-1-2"]["lost_water_candidate_overlap"]
    assert not roads["way-30-nodes-1-3"]["new_water_candidate_overlap"]
    assert not any(properties["road_closure_assigned"] for properties in roads.values())
    buildings = {feature["id"]: feature["properties"]["coverage"] for feature in collections["buildings"]["features"]}
    assert buildings["way/1001"]["status"] == "within_aoi_fully_unassessed"
    assert buildings["way/1001"]["new_water_candidate"] == buildings["way/1001"]["lost_water_candidate"] == 0
    assert not buildings["way/1001"]["fully_assessed_no_new_water_candidate"]
    assert buildings["way/1002"]["status"] == "outside_aoi"
    assert "change_candidate" not in roads["way-10-nodes-1-2"]["coverage"]
    candidates, summary = mapper.candidate_features(polygons, masks, GRID)
    assert summary["new"]["candidate_area_m2"] == 200
    assert summary["lost"]["candidate_area_m2"] == 100


def test_hypothetical_access_keeps_fixed_nearest_target_and_all_group_directions(tmp_path):
    _, _, _, _, osm = fixture(tmp_path)
    _, _, regions = mapper.mask_regions(arrays(), GRID)
    collections, _, graph = mapper.overlap_features(osm, GRID, regions)
    original_edges = list(graph.edges(keys=True, data=True))
    proof, _, _ = mapper.access_scenario(graph, osm, collections["roads"])
    assert proof["uncertain_links"] == {"way-10-nodes-1-2": ["10:0:f", "10:0:r"], "way-20-nodes-1-2": ["20:0:f"]}
    assert proof["certain_closed_edge_ids"] == []
    row = proof["settlements"][0]
    assert row["status"] == "ambiguous"
    assert row["nearest_baseline_destination"]["destination_id"] == "node/2"
    assert row["optimistic_path_witness"]["destination_id"] == "node/2"
    assert row["pessimistic_cut_certificate"]["mapped_destination_node_ids"] == [2]
    assert proof["any_destination_status_counts_for_comparison"] == {"robust_connected": 1}
    assert proof["certificate_checks"] == {"paths": 1, "cuts": 1}
    assert proof["fixed_destination_certificate_checks"] == {"paths": 1, "cuts": 1}
    assert list(graph.edges(keys=True, data=True)) == original_edges


def test_saved_binding_covers_all_sources_outputs_and_no_rerun_overwrite(tmp_path):
    manifest, plan_path, water, _, _ = fixture(tmp_path)
    out = tmp_path / "new-output"
    summary = mapper.run(manifest, plan_path, out)
    assert summary["roads"]["mapped_feature_count"] == 3
    assert summary["bridges"]["mapped_feature_count"] == 1
    assert summary["buildings"]["features_with_unknown_water_state"] == 1
    assert summary["access_scenario"]["n_uncertain_physical_segments"] == 2
    assert summary["access_scenario"]["n_uncertain_directed_edges"] == 3
    assert summary["access_scenario"]["confirmed_closures"] == 0
    binding = json.loads((out/"source-and-output-manifest.json").read_text())
    assert binding["schema"] == "akash-fresh-exposure-binding-v1" and binding["status"] == "completed"
    assert binding["summary"] == binding["outputs"]["summary.json"]
    for item in binding["inputs"] + binding["code"] + list(binding["outputs"].values()):
        assert mapper.record(Path(item["path"])) == item
    frozen = (out/"summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        mapper.run(manifest, plan_path, out)
    assert (out/"summary.json").read_bytes() == frozen
