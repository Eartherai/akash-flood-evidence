import json
from pathlib import Path
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import process_fresh_s1 as pipeline


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def make_plan(tmp_path):
    bbox = [-69.55, -18.35, -69.50, -18.30]
    context = [-69.56, -18.36, -69.49, -18.29]
    pair = {"relative_orbit": 88, "orbit_state": "descending", "platform": "sentinel-1a",
            "mode": "IW", "polarizations": ["VV", "VH"], "separation_days": 12}
    for scene, timestamp in (("before", "2026-08-16T10:00:00Z"), ("after", "2026-08-28T10:00:00Z")):
        item = {"type": "Feature", "id": "native-" + scene, "collection": "sentinel-1-grd",
                "geometry": mapping(box(-69.7, -18.5, -69.3, -18.1)), "assets": {},
                "properties": {"datetime": timestamp, "sat:relative_orbit": 88, "sat:orbit_state": "descending",
                               "platform": "sentinel-1a", "sar:instrument_mode": "IW",
                               "sar:product_type": "GRD", "sar:polarizations": ["VV", "VH"]}}
        path = write_json(tmp_path / (scene + ".stac.json"), item)
        pair[scene] = {"item_path": str(path), "item_sha256": pipeline.sha256(path), "item_id": item["id"],
                       "datetime": timestamp, "catalogue_item_canonical_sha256": pipeline.canonical_sha(item)}
    plan = {"case": {"case_id": "fresh-south-test", "bbox": bbox, "context_bbox": context,
                     "event_date": "2026-08-26", "osm_snapshot": "2026-08-25"},
            "projection": pipeline.plan_projection(bbox), "sentinel1": {"selected_pair": pair}}
    path = write_json(tmp_path / "plan.json", plan)
    return path, plan


def make_acquisition(tmp_path, plan_path, plan):
    index = {"schema": "akash-fresh-native-acquisition-v1", "status": "completed",
             "source_plan": pipeline.record(plan_path), "input_stac": {}, "scenes": {}}
    for scene in ("before", "after"):
        selected = plan["sentinel1"]["selected_pair"][scene]
        safe = tmp_path / (scene + ".SAFE")
        safe.mkdir()
        manifest = safe / "manifest.safe"
        manifest.write_text("derived test wrapper " + scene)
        original = tmp_path / (scene + "-original.safe")
        original.write_text("original test manifest " + scene)
        alias = safe / "source.xml"
        alias.write_text("test source object")
        index["input_stac"][scene] = pipeline.record(Path(selected["item_path"]))
        index["scenes"][scene] = {"source_item_id": selected["item_id"], "datetime": selected["datetime"],
                                  "derived_safe": str(safe), "derived_manifest": str(manifest),
                                  "derived_manifest_sha256": pipeline.sha256(manifest), "original_manifest": str(original),
                                  "original_manifest_sha256": pipeline.sha256(original), "not_original_esa_safe": True,
                                  "files": [{"alias": "source.xml", "bytes": alias.stat().st_size, "sha256": pipeline.sha256(alias)}]}
    return write_json(tmp_path / "derived-wrapper-index.json", index), index


def make_dem(tmp_path, context, monkeypatch):
    dem = tmp_path / "ellipsoid.tif"
    with rasterio.open(dem, "w", driver="GTiff", width=40, height=40, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(-69.7, -18.15, .01, .01), nodata=-32767) as ds:
        ds.write(np.full((40, 40), -5, dtype="float32"), 1)
    source = tmp_path / "source.tif"
    source.write_bytes(b"Bound test source; not a production tile")
    geoid = tmp_path / "test-geoid.tif"
    geoid.write_bytes(b"Bound test geoid; production constants are never changed")
    monkeypatch.setattr(pipeline, "GRID_SHA256", pipeline.sha256(geoid))
    monkeypatch.setattr(pipeline, "GRID_BYTES", geoid.stat().st_size)
    proof = {"formula": "h_WGS84 = H_EGM2008 + N_EGM2008", "mandatory_local_grid": True,
             "fallback_allowed": False, "network_enabled_for_proj": False, "independent_sample_count": 289,
             "max_proj_vs_independent_error_m": 1e-10}
    diagnostics = write_json(tmp_path / "conversion-diagnostics.json", proof)
    manifest = {"schema": "akash-fresh-dem-v1", "completed_at": "2026-10-07T00:00:00Z", "context_bbox": list(context),
                "snap_settings": {"externalDEMApplyEGM": False},
                "source_tiles": [{**pipeline.record(source), "url": "https://copernicus-dem-30m.s3.amazonaws.com/test/test.tif"}],
                "geoid": pipeline.record(geoid), "diagnostics": pipeline.record(diagnostics),
                "output": {**pipeline.record(dem), "vertical_datum": "WGS84 ellipsoidal height", "horizontal_crs": "EPSG:4326",
                           "nodata": -32767, "formula": proof["formula"]}}
    return write_json(tmp_path / "dem-manifest.json", manifest), manifest


def test_derives_southern_projection_and_refuses_changed_stac_or_plan(tmp_path):
    path, plan = make_plan(tmp_path)
    _, _, _, projection, _ = pipeline.load_plan(path)
    assert projection["epsg"] == 32719
    plan["projection"]["epsg"] = 32645
    write_json(path, plan)
    with pytest.raises(ValueError, match="Planned projection differs"):
        pipeline.load_plan(path)
    plan["projection"]["epsg"] = 32719
    write_json(path, plan)
    Path(plan["sentinel1"]["selected_pair"]["before"]["item_path"]).write_text("changed")
    with pytest.raises(ValueError, match="Checksum"):
        pipeline.load_plan(path)


def test_rejects_unreviewed_cross_zone_context_and_missing_pair(tmp_path):
    path, plan = make_plan(tmp_path)
    plan["case"]["context_bbox"] = [-72.01, -18.36, -69.49, -18.29]
    write_json(path, plan)
    with pytest.raises(ValueError, match="single UTM"):
        pipeline.load_plan(path)
    plan["case"]["context_bbox"] = [-69.56, -18.36, -69.49, -18.29]
    plan["sentinel1"]["selected_pair"] = None
    write_json(path, plan)
    with pytest.raises(ValueError, match="no selected native"):
        pipeline.load_plan(path)


def test_acquisition_requires_completion_matching_plan_and_safe_aliases(tmp_path):
    plan_path, plan = make_plan(tmp_path)
    _, _, _, _, items = pipeline.load_plan(plan_path)
    index_path, index = make_acquisition(tmp_path, plan_path, plan)
    assert set(pipeline.acquired_sources(index_path, plan_path, plan, items)[0]) == {"before", "after"}
    index["status"] = "running"
    write_json(index_path, index)
    with pytest.raises(ValueError, match="not complete"):
        pipeline.acquired_sources(index_path, plan_path, plan, items)
    index["status"] = "completed"
    index["scenes"]["before"]["files"][0]["alias"] = "../escape.xml"
    write_json(index_path, index)
    with pytest.raises(ValueError, match="Unsafe"):
        pipeline.acquired_sources(index_path, plan_path, plan, items)


def test_prepared_dem_requires_eligible_datum_proof_and_interpolation_coverage(tmp_path, monkeypatch):
    context = (-69.56, -18.36, -69.49, -18.29)
    path, manifest = make_dem(tmp_path, context, monkeypatch)
    dem, nodata, _ = pipeline.prepared_dem(path, context)
    assert dem == Path(manifest["output"]["path"]) and nodata == -32767
    manifest["source_tiles"][0]["url"] = "https://example.com/srtm.tif"
    write_json(path, manifest)
    with pytest.raises(ValueError, match="unsupported elevation"):
        pipeline.prepared_dem(path, context)
    manifest["source_tiles"][0]["url"] = "https://copernicus-dem-30m.s3.amazonaws.com/test/test.tif"
    outside = (-69.70, -18.36, -69.49, -18.29)
    manifest["context_bbox"] = list(outside)
    write_json(path, manifest)
    with pytest.raises(ValueError, match="interpolation margin"):
        pipeline.prepared_dem(path, outside)


def test_graph_preserves_linear_negative_power_and_explicit_external_dem(tmp_path):
    doc = pipeline.graph("before", tmp_path / "manifest.safe", tmp_path / "out.dim",
                         [-69.56, -18.36, -69.49, -18.29], 32719, tmp_path / "ellipsoid.tif", -32767).getroot()
    assert doc.findtext("./node[@id='noise']/parameters/clipNegativeValues") == "false"
    assert doc.findtext("./node[@id='calibrate']/parameters/outputImageScaleInDb") == "false"
    assert doc.findtext("./node[@id='terrain']/parameters/externalDEMApplyEGM") == "false"
    assert doc.findtext("./node[@id='terrain']/parameters/demName") == "External DEM"
    assert doc.findtext("./node[@id='terrain']/parameters/applyRadiometricNormalization") == "false"
    assert "19S" in doc.findtext("./node[@id='terrain']/parameters/mapProjection")
    assert all("SRTM" not in str(element.text) for element in doc.iter())


def test_preflight_never_runs_snap_or_marks_raster_processing_complete(tmp_path, monkeypatch):
    plan_path, plan = make_plan(tmp_path)
    index_path, _ = make_acquisition(tmp_path, plan_path, plan)
    dem_manifest, _ = make_dem(tmp_path, plan["case"]["context_bbox"], monkeypatch)
    patch, proof = tmp_path / "overlay.json", tmp_path / "proof.json"
    monkeypatch.setattr(pipeline, "verified_reader_patch", lambda *args: (tmp_path / "overlay.jar", {"scope": "test fixture only"}))
    monkeypatch.setattr(pipeline, "command", lambda args: ["java", "-cp", "fixture", *args])
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: pytest.fail("SNAP must not run during preflight"))
    output = tmp_path / "preflight"
    report = pipeline.run(plan_path, index_path, dem_manifest, patch, proof, output)
    assert report["status"] == "preflight_only" and "completed_at" not in report
    assert report["scenes"]["before"]["status"] == "not_executed"
    assert (output / "after-terrain.xml").exists()
    assert not (output / "after-terrain.dim").exists()
    with pytest.raises(FileExistsError):
        pipeline.run(None, None, None, None, None, output)
