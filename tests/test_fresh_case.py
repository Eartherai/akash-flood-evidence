"""Meaningful dry-run boundaries; no network, native processing or model calls."""
from copy import deepcopy
from datetime import datetime, timezone
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from akash_flood.fresh_case import build_plan, copdem_tiles, event_boundary, plan_projection, save_plan, sha256


BBOX = [85.3, 28.12, 85.43, 28.32]


def scene(identity, day, *, satellite=1, orbit=85, footprint=None, cloud=0):
    properties = {"datetime": f"2026-08-{day:02}T12:00:00Z", "platform": "sentinel-1d" if satellite == 1 else "sentinel-2a",
                  "sat:relative_orbit": orbit, "sat:orbit_state": "ascending"}
    if satellite == 1:
        properties.update({"sar:instrument_mode": "IW", "sar:polarizations": ["VV", "VH"]})
    else:
        properties["eo:cloud_cover"] = cloud
    return {"type": "Feature", "id": identity, "collection": "sentinel-1-grd" if satellite == 1 else "sentinel-2-l2a",
            "properties": properties, "geometry": mapping(footprint or box(85, 28, 86, 29)),
            "assets": {"vv" if satellite == 1 else "scl": {"href": "s3://public-sentinel/source/" + identity,
                      "roles": ["data"], "file:size": 100, "file:checksum": "provider-declared-not-content-verified"}}}


def catalog():
    return {"source": "saved test discovery", "event_date": "2026-08-26", "bbox": BBOX,
            "sentinel1": [scene("before", 16), scene("after", 28)], "sentinel2": []}


def plan(document=None, **kwargs):
    return build_plan(document or catalog(), bbox=kwargs.pop("bbox", BBOX), event_date="2026-08-26",
                      osm_snapshot="2026-07-27", **kwargs)[0]


def test_selection_recomputed_full_context_orbit_event_day_and_source_identity():
    data = catalog()
    data["selected_pair"] = {"before": scene("cached-other-identity", 16), "after": scene("cached-other-after", 28)}
    data["sentinel1"] += [scene("event-day", 26), scene("wrong-orbit", 27, orbit=121),
                         scene("later-post", 29), scene("partial-footprint", 27, footprint=box(*BBOX))]
    result = plan(data)
    pair = result["sentinel1"]["selected_pair"]
    assert pair["before"]["item_id"] == "before"
    assert pair["after"]["item_id"] == "after"
    assert pair["relative_orbit"] == 85 and pair["separation_days"] == 12
    assert result["sentinel1"]["saved_selected_pair_ignored"]
    assert result["case"]["context_bbox"] == pytest.approx([85.27, 28.09, 85.46, 28.35])
    exclusions = {row["item_id"]: row["reason"] for row in result["sentinel1"]["exclusions"]}
    assert exclusions["event-day"] == "event_day_or_outside_requested_window"
    assert exclusions["partial-footprint"] == "footprint_does_not_cover_entire_requested_context"
    assert result["execution"]["runnable_end_to_end_verified"] is False
    assert all(value == "not_run" for value in result["execution"]["stages"].values())
    assert not any(result["execution"][key] for key in ("downloads_performed", "snap_run", "model_inference_run", "evaluation_labels_loaded"))


@pytest.mark.parametrize("mutation", [
    lambda item: item["properties"].update({"sat:orbit_state": "descending"}),
    lambda item: item["properties"].update({"platform": "sentinel-1c"}),
    lambda item: item["properties"].update({"sar:polarizations": ["VV"]}),
    lambda item: item["properties"].update({"sar:instrument_mode": "EW"}),
    lambda item: item["properties"].update({"sat:relative_orbit": None}),
    lambda item: item["properties"].update({"datetime": "2026-08-28T12:00:00"}),
    lambda item: item["properties"].update({"datetime": "2026-08-29T12:00:00Z"}),
])
def test_incompatible_pair_is_blocked_without_substituting_or_faking_completion(mutation):
    data = catalog()
    mutation(data["sentinel1"][1])
    result = plan(data)
    assert result["sentinel1"]["selected_pair"] is None
    assert result["input_plan_status"] == "blocked"
    assert "no_complete_context_same_orbit_dual_pol_IW_pair" in result["blockers"]


@pytest.mark.parametrize("snapshot", ["2026-08-26", "2026-08-26T00:00:00Z", "2026-08-27", "2026-08-25T20:00:00-05:00"])
def test_osm_snapshot_cannot_reach_event_day_even_with_timezone_offset(snapshot):
    with pytest.raises(ValueError, match="strictly before"):
        event_boundary("2026-08-26", snapshot)


def test_timezone_normalization_and_historical_query_have_exact_pre_event_cutoff():
    start, end, snapshot = event_boundary("2026-08-26", "2026-08-25T23:00:00+05:30")
    assert snapshot == datetime(2026, 8, 25, 17, 30, tzinfo=timezone.utc)
    result = plan()
    assert '[date:"2026-07-27T00:00:00Z"]' in result["osm"]["query"]
    assert result["osm"]["status"] == "query_planned_not_retrieved"


def test_explicit_processing_context_is_bound_to_case_identity_and_pair_coverage():
    context = [85.27, 28.08, 85.46, 28.36]
    explicit = plan(context_bbox=context)
    assert explicit["case"]["context_bbox"] == context
    assert explicit["case"]["context_buffer_deg"] is None
    assert explicit["case"]["context_source"] == "explicit_bbox"
    assert explicit["case"]["case_id"] != plan()["case"]["case_id"]
    with pytest.raises(ValueError, match="entire requested AOI"):
        plan(context_bbox=[85.31, 28.13, 85.42, 28.31])


@pytest.mark.parametrize("mutation", [
    lambda data: data["sentinel1"][0].update(id="EMSR927-reference"),
    lambda data: data["sentinel1"][0].update(collection="sentinel-1-rtc"),
    lambda data: data["sentinel1"][0]["properties"].update({"sar:product_type": "RTC"}),
    lambda data: data["sentinel1"][0]["assets"]["vv"].update(roles=["labels"]),
    lambda data: data["sentinel1"][0]["assets"]["vv"].update(href="https://unosat.org/damage.tif"),
    lambda data: data.update(additional_inputs=[{"source": "SRTM"}]),
    lambda data: data.update(reference_labels_loaded=True),
    lambda data: data.update(source="https://mapping.emergency.copernicus.eu/"),
])
def test_excluded_sources_and_published_labels_are_rejected_before_selection(mutation):
    data = catalog()
    mutation(data)
    with pytest.raises(ValueError):
        plan(data)


def test_saved_search_pagination_cannot_be_called_complete():
    data = catalog()
    data["links"] = [{"rel": "next", "href": "https://example.org/next"}]
    result = plan(data)
    assert result["sentinel1"]["selected_pair"] is not None
    assert result["input_plan_status"] == "blocked"
    assert result["discovery"]["search_incomplete"]


def test_utm_changes_with_new_location_hemisphere_and_exposes_zone_boundary():
    assert plan_projection(BBOX)["epsg"] == 32645
    south = plan_projection([18.1, -34.1, 18.2, -34])
    assert south["epsg"] == 32734 and south["hemisphere"] == "south"
    assert south["status"] == "single_utm_zone"
    assert plan_projection([85, 28, 90, 29])["status"] == "single_utm_zone"  # exact east edge does not include zone 46
    assert plan_projection([89.9, 28, 90.1, 29])["status"] == "projection_review_required"
    assert plan_projection([3.1, 60, 3.2, 60.1])["zone"] == 32  # Norway
    assert plan_projection([9.1, 75, 9.2, 75.1])["zone"] == 33  # Svalbard
    assert plan_projection([85, -.1, 85.1, .1])["crosses_equator"]
    assert plan_projection([85, 84.1, 85.1, 84.2])["status"] == "unsupported_polar_utm"


def test_copdem_tile_origins_cover_integer_boundaries_and_south_west_locations():
    result = copdem_tiles([84.9, 27.9, 86, 29])
    assert result["required_tile_count"] == 4
    assert {tile["tile_id"] for tile in result["tiles"]} == {
        "Copernicus_DSM_COG_10_N27_00_E084_00_DEM", "Copernicus_DSM_COG_10_N27_00_E085_00_DEM",
        "Copernicus_DSM_COG_10_N28_00_E084_00_DEM", "Copernicus_DSM_COG_10_N28_00_E085_00_DEM"}
    southwest = copdem_tiles([-3.9, -2.9, -3.1, -2.1])["tiles"][0]
    assert "S03_00_W004_00" in southwest["tile_id"]
    assert southwest["bbox"] == [-4, -3, -3, -2]
    assert southwest["downloaded"] is False and southwest["sha256"] is None
    assert copdem_tiles([0, 0, 20, 20])["status"] == "planning_tile_limit_exceeded"


def test_optical_tile_cloud_percentage_never_becomes_clear_aoi_measurement():
    data = catalog()
    data["sentinel2"] = [scene("optical-pre", 16, satellite=2, cloud=0), scene("optical-after", 28, satellite=2, cloud=0)]
    result = plan(data)
    assert result["sentinel2"]["status"] == "catalogued_clear_coverage_unknown"
    assert all(scene["clear_coverage_status"].startswith("unknown") for scene in result["sentinel2"]["scenes"])
    assert all("coverage" not in scene for scene in result["sentinel2"]["scenes"])


def test_s2_current_collection1_l2a_is_eligible_but_l1c_is_not():
    data = catalog()
    item = scene("current-optical-pre", 16, satellite=2)
    item["collection"] = "sentinel-2-c1-l2a"
    data["sentinel2"] = [item]
    assert plan(data)["sentinel2"]["scenes"][0]["item_id"] == "current-optical-pre"
    item["collection"] = "sentinel-2-l1c"
    with pytest.raises(ValueError, match="L2A"):
        plan(data)


def test_supplied_hashed_scl_reports_clear_unknown_and_excluded_classes_locally(tmp_path):
    data = catalog()
    data["sentinel2"] = [scene("optical-pre", 16, satellite=2)]
    raster = tmp_path / "local-scl.tif"
    with rasterio.open(raster, "w", driver="GTiff", count=1, width=2, height=2, dtype="uint8", crs=4326,
                       transform=from_origin(85.3, 28.32, .065, .1), nodata=0) as dataset:
        dataset.write(np.array([[4, 6], [9, 0]], dtype="uint8"), 1)
    record = tmp_path / "optical-coverage.json"
    record.write_text(json.dumps({"bbox": BBOX, "scenes": [{"item_id": "optical-pre", "scl_path": raster.name, "sha256": sha256(raster)}]}))
    result = plan(data, coverage_path=record)
    optical = result["sentinel2"]["scenes"][0]
    assert optical["clear_coverage_status"] == "partial_usable_coverage"
    assert optical["coverage"]["aoi_pixels"] == 4
    assert optical["coverage"]["usable_clear_pixels"] == 2
    assert optical["coverage"]["unassessed_pixels"] == 1
    assert optical["coverage"]["excluded_class_pixels"] == 1
    assert optical["coverage"]["clear_fraction"] == .5
    assert result["sentinel2"]["reflectance_comparison_status"] == "not_run"
    bad = json.loads(record.read_text()); bad["scenes"][0]["sha256"] = "0" * 64
    record.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="checksum"):
        plan(data, coverage_path=record)


def test_source_identity_hashes_fresh_writes_and_query_redaction(tmp_path):
    data = catalog()
    data["sentinel1"][0]["assets"]["vv"]["href"] = "https://example.org/public/source?temporary_signature=abc"
    source, out = tmp_path / "catalog.json", tmp_path / "fresh-plan"
    source.write_text(json.dumps(data))
    result = save_plan(source, out, bbox=BBOX, event_date="2026-08-26", osm_snapshot="2026-07-27", project_root=tmp_path)
    pair = result["sentinel1"]["selected_pair"]
    assert pair["before"]["assets"]["vv"]["href_without_query"] == "https://example.org/public/source"
    assert pair["before"]["assets"]["vv"]["query_removed"]
    assert pair["before"]["item_sha256"] == sha256(out / "before.stac.json")
    assert json.loads((out / "before.stac.json").read_text())["id"] == "before"
    manifest = json.loads((out / "source-and-output-manifest.json").read_text())
    assert manifest["inputs"][0]["sha256"] == sha256(source)
    for name, record in manifest["outputs"].items():
        assert sha256(out / name) == record["sha256"]
    original = (out / "plan.json").read_bytes()
    with pytest.raises(FileExistsError, match="fresh"):
        save_plan(source, out, bbox=BBOX, event_date="2026-08-26", osm_snapshot="2026-07-27")
    assert (out / "plan.json").read_bytes() == original


def historical_inputs(tmp_path):
    source = {"elements": [{"type": "node", "id": 1, "lon": 85.35, "lat": 28.2,
                            "timestamp": "2026-07-26T00:00:00Z"}]}
    extract, metadata = tmp_path / "osm.json", tmp_path / "osm.provenance.json"
    extract.write_text(json.dumps(source))
    provenance = {"snapshot": "2026-07-27T00:00:00Z", "bbox": [85, 28, 86, 29],
                  "query": '[out:json][date:"2026-07-27T00:00:00Z"];nwr;out body;',
                  "sha256": sha256(extract)}
    metadata.write_text(json.dumps(provenance))
    return extract, metadata, source, provenance


def test_saved_historical_osm_checks_cutoff_extent_and_exact_content_hash(tmp_path):
    extract, metadata, source, provenance = historical_inputs(tmp_path)
    result = plan(osm_path=extract, osm_provenance=metadata)
    assert result["osm"]["status"] == "saved_historical_extract_checked"
    assert result["osm"]["local_inputs"][0]["sha256"] == sha256(extract)
    source["elements"][0]["timestamp"] = "2026-07-28T00:00:00Z"
    extract.write_text(json.dumps(source))
    provenance["sha256"] = sha256(extract)
    metadata.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="postdates"):
        plan(osm_path=extract, osm_provenance=metadata)


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p.update(snapshot="2026-08-27T00:00:00Z"), "snapshot"),
    (lambda p: p.update(bbox=[85.31, 28.13, 85.42, 28.31]), "extent"),
    (lambda p: p.update(query="current OSM without historical query cutoff"), "cutoff"),
    (lambda p: p.update(sha256="0" * 64), "checksum"),
])
def test_local_osm_without_matching_pre_event_provenance_is_rejected(tmp_path, mutation, match):
    extract, metadata, source, provenance = historical_inputs(tmp_path)
    mutation(provenance)
    metadata.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match=match):
        plan(osm_path=extract, osm_provenance=metadata)


def test_generic_code_presence_is_not_case_validation_or_fixed_projection_blocker(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/process_fresh_s1.py").write_text("# synthetic implementation inventory fixture\n")
    data = catalog()
    for item in data["sentinel1"]:
        item["geometry"] = mapping(box(18, -35, 19, -33))
    result = plan(data, bbox=[18.1, -34.1, 18.2, -34], project_root=tmp_path)
    assert result["projection"]["epsg"] == 32734
    component = result["implementation"]["generic_components"]["native_processing"]
    assert component["status"].startswith("implementation_present")
    assert component["source_sha256"] and component["fresh_case_validated"] is False
    assert result["implementation"]["radar_reporting_grid_adapter_review_required"]
    assert not any("32645" in blocker for blocker in result["blockers"])
    assert result["execution"]["runnable_end_to_end_verified"] is False
