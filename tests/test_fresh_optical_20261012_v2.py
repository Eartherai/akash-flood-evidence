"""Synthetic adapter contracts; these tests are not satellite accuracy evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from pyproj import Transformer
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("fresh_optical_adapter_v2", ROOT / "scripts/run_fresh_optical_20261012_v2.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

BBOX = [85.30, 28.12, 85.31, 28.13]
EVENT = "2024-09-28"


def item(identity, when, cloud=0, tile="UM", epsg=32645):
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(BBOX[0], BBOX[3])
    left, top = int(x // 20) * 20 - 1000, int(y // 20) * 20 + 1000
    assets = {}
    for key, filename in adapter.BAND_FILENAMES.items():
        spacing = 20 if key in ("scl", "swir16") else 10
        band = {"nodata": 0, "data_type": "uint8" if key == "scl" else "uint16"}
        if key != "scl":
            band.update(scale=.0001, offset=-.1)
        assets[key] = {"href": "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/sentinel-2-c1-l2a/45/R/UM/2024/9/" + identity + "/" + filename,
                       "gsd": spacing, "file:size": 1000000,
                       "proj:transform": [spacing, 0, left, 0, -spacing, top],
                       "proj:shape": [4000 if spacing == 10 else 2000] * 2, "raster:bands": [band]}
    return {"type": "Feature", "id": identity, "collection": "sentinel-2-c1-l2a", "assets": assets,
            "geometry": {"type": "Polygon", "coordinates": [[[85, 28], [86, 28], [86, 29], [85, 29], [85, 28]]]},
            "properties": {"datetime": when, "platform": "sentinel-2a", "eo:cloud_cover": cloud,
                           "proj:epsg": epsg, "mgrs:utm_zone": 45, "mgrs:latitude_band": "R", "mgrs:grid_square": tile}}


def document():
    return {"type": "FeatureCollection", "bbox": BBOX, "event_date": EVENT, "search_complete": True,
            "features": [item("old-clear", "2024-09-25T10:00:00Z"),
                         item("latest-cloudy", "2024-09-27T10:00:00Z", cloud=95),
                         item("event-day", "2024-09-28T20:00:00Z"),
                         item("first-cloudy", "2024-09-29T10:00:00Z", cloud=95),
                         item("later-clear", "2024-09-30T10:00:00Z")]}


def prepare_args(tmp_path, doc=None, plan=None):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(doc or document()))
    argv = ["--prepare-only", "--catalog", str(catalog), "--bbox", *map(str, BBOX),
            "--event-date", EVENT, "--out", str(tmp_path / "optical")]
    if plan is not None:
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(plan))
        argv += ["--plan", str(path)]
    return adapter.parser().parse_args(argv)


def test_prepare_freezes_nearest_compatible_pair_without_network_or_pixel_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "read_band", lambda *a: pytest.fail("Preparation cannot read pixels"))
    args = prepare_args(tmp_path)
    receipt = adapter.prepare(args)
    assert receipt["selected"] == {"before": "latest-cloudy", "after": "first-cloudy"}
    assert receipt["network_requests"] == 0 and receipt["pixels_read"] is False
    protocol, _, _, _, _ = adapter.read_frozen(args.out)
    assert protocol["policy"] == adapter.POLICY
    assert protocol["crop_reuse"] is None and protocol["reference_labels_loaded"] is False


@pytest.mark.parametrize("change", ["incomplete", "next", "bbox", "event", "reference", "duplicate"])
def test_bad_discovery_is_rejected_before_freeze(tmp_path, change):
    doc = document()
    if change == "incomplete":
        doc["search_complete"] = False
    elif change == "next":
        doc["links"] = [{"rel": "next", "href": "https://example.com/next"}]
    elif change == "bbox":
        doc["bbox"] = [85.305, 28.12, 85.32, 28.13]
    elif change == "event":
        doc["event_date"] = "2024-09-29"
    elif change == "reference":
        doc["features"][0]["assets"]["blue"]["roles"] = ["reference"]
    else:
        doc["features"].append(copy.deepcopy(doc["features"][0]))
    args = prepare_args(tmp_path, doc)
    with pytest.raises(ValueError):
        adapter.prepare(args)
    assert not args.out.exists()


def test_case_plan_date_bbox_and_native_projection_must_match(tmp_path):
    plan = {"schema": "akash-fresh-case-runner-v1", "case": {"bbox": BBOX, "event_date": "2024-09-27"}}
    args = prepare_args(tmp_path, plan=plan)
    with pytest.raises(ValueError, match="plan bbox/event"):
        adapter.prepare(args)
    plan["case"]["event_date"] = EVENT
    plan["projection"] = {"status": "single_utm_zone", "epsg": 32644}
    args.plan.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="projection"):
        adapter.prepare(args)
    assert not args.out.exists()


@pytest.mark.parametrize("kind", ["filename", "signed", "grid", "scale", "datatype", "nodata"])
def test_selected_assets_need_exact_band_identity_geometry_and_radiometry(tmp_path, kind):
    doc = document()
    asset = doc["features"][1]["assets"]["blue"]
    if kind == "filename":
        asset["href"] = asset["href"].replace("B02.tif", "B04.tif")
    elif kind == "signed":
        asset["href"] += "?token=secret"
    elif kind == "grid":
        asset["proj:transform"][2] += 10
    elif kind == "scale":
        del asset["raster:bands"][0]["scale"]
    elif kind == "datatype":
        asset["raster:bands"][0]["data_type"] = "float32"
    else:
        asset["raster:bands"][0]["nodata"] = None
    args = prepare_args(tmp_path, doc)
    with pytest.raises(ValueError):
        adapter.prepare(args)
    assert not args.out.exists()


def test_native_cell_budget_and_strict_date_are_enforced(tmp_path):
    args = prepare_args(tmp_path)
    args.max_cells = 1
    with pytest.raises(ValueError, match="cell count"):
        adapter.prepare(args)
    args.max_cells = 1000000
    args.event_date = "20240928"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        adapter.prepare(args)


@pytest.mark.parametrize("which", ["catalog", "protocol", "source", "plan"])
def test_frozen_inputs_cannot_change_between_prepare_and_execute(tmp_path, which):
    plan = {"schema": "akash-fresh-case-runner-v1", "case": {"bbox": BBOX, "event_date": EVENT}}
    args = prepare_args(tmp_path, plan=plan)
    adapter.prepare(args)
    path = {"catalog": args.catalog, "protocol": args.out / "protocol.json",
            "source": args.out / "before.stac.json", "plan": args.plan}[which]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="identity changed"):
        adapter.read_frozen(args.out)


class FakeClient:
    """Zero-network synthetic execution fixture, not a transport verification."""
    def __init__(self, *args, **kwargs):
        self.bytes_read = 0
        self.requests = []

    def audit(self):
        return {"received_body_bytes": 0, "requests": [], "synthetic_fixture_only": True}


def fake_reader(item, key, window, grid, client, out):
    dims = (int(grid[3]), int(grid[2]))
    if key == "scl":
        values = np.full(dims, 4, dtype="uint8")
        if item["id"].startswith("first"):
            values[12, 12] = 9
    else:
        original = {"blue": .1, "red": .1, "nir": .4, "swir16": .1}
        changed = {"blue": .08, "red": .2, "nir": .22, "swir16": .4}
        values = np.full(dims, (changed if item["id"].startswith("first") else original)[key], dtype="float32")
    path = out / (key + "-dn.tif")
    adapter.write_layer(path, values, grid, 0)
    return values, {"crop": adapter.file_record(path), "asset": item["assets"][key], "synthetic_fixture_only": True}


def test_execution_unknown_masks_are_not_negative_damage_and_manifest_attaches_to_report(tmp_path):
    args = prepare_args(tmp_path)
    adapter.prepare(args)
    execute_args = adapter.parser().parse_args(["--execute", "--out", str(args.out)])
    result = adapter.execute(execute_args, client_factory=FakeClient, band_reader=fake_reader)
    manifest = adapter.load(args.out / "manifest.json")
    assert result["status"] == "completed"
    assert manifest["counts"]["unknown_pixels"] > 0
    assert manifest["counts"]["candidate_pixels"] > 0
    assert manifest["source_item_ids"] == {"before": "latest-cloudy", "after": "first-cloudy"}
    assert manifest["partial_exploration_is_original_gate_pass"] is False
    from build_fresh_report import read_optical
    water = {"case": {"bbox": BBOX, "event_date": EVENT}, "grid": {"epsg": 32645}}
    facts, _ = read_optical(args.out / "manifest.json", water)
    assert facts["counts"] == manifest["counts"]
    with pytest.raises(FileExistsError, match="already attempted"):
        adapter.execute(execute_args, client_factory=FakeClient, band_reader=fake_reader)


def test_failed_raw_read_preserves_failure_and_prohibits_automatic_retry(tmp_path):
    args = prepare_args(tmp_path)
    adapter.prepare(args)
    execute_args = adapter.parser().parse_args(["--execute", "--out", str(args.out)])
    def fail(*args):
        raise OSError("synthetic interrupted source")
    with pytest.raises(OSError, match="interrupted"):
        adapter.execute(execute_args, client_factory=FakeClient, band_reader=fail)
    failure = adapter.load(args.out / "failed.json")
    assert failure["no_automatic_retry"] is True and failure["completed_optical_map"] is False
    assert failure["missing_observations_are_safe_or_negative"] is False
    assert (args.out / "transport.json").exists() and not (args.out / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        adapter.execute(execute_args, client_factory=FakeClient, band_reader=fail)


def test_alias_budget_and_timeout_contract_is_parsed_and_bounded(tmp_path):
    args = adapter.parser().parse_args(["--prepare-only", "--out", str(tmp_path / "new"), "--budget-bytes", "60000000", "--timeout-seconds", "600"])
    assert args.max_body_bytes == 60000000 and args.max_seconds == 600
    with pytest.raises(ValueError):
        adapter.bounded_int(60000001, 60000000, "budget")
    with pytest.raises(ValueError):
        adapter.bounded_int(601, 600, "timeout")


def queried_document(context):
    doc = document()
    doc.update(bbox=context, queried_context_bbox=context, max_days=24,
               search_records=[{"collection": "sentinel-2-c1-l2a", "search_complete": True,
                                "query": {"bbox": context, "datetime": "2024-09-04T00:00:00Z/2024-10-23T00:00:00Z"}}])
    return doc


def test_complete_containing_plan_context_catalogue_prepares_exact_aoi_without_relabelling(tmp_path):
    context = [85.29, 28.11, 85.32, 28.14]
    plan = {"schema": "akash-fresh-case-runner-v1", "case": {"bbox": BBOX, "context_bbox": context, "event_date": EVENT}}
    args = prepare_args(tmp_path, queried_document(context), plan)
    adapter.prepare(args)
    protocol, _, _, _, _ = adapter.read_frozen(args.out)
    assert protocol["bbox"] == BBOX and protocol["saved_query_bbox"] == context
    assert protocol["provider_query_context_bbox"] == context
    assert protocol["adapter_schema"].endswith("v2")


def test_documented_exact_aoi_query_is_allowed_with_larger_case_plan_context(tmp_path):
    plan = {"schema": "akash-fresh-case-runner-v1", "case": {"bbox": BBOX, "context_bbox": [85.29, 28.11, 85.32, 28.14], "event_date": EVENT}}
    args = prepare_args(tmp_path, queried_document(BBOX), plan)
    assert adapter.prepare(args)["network_requests"] == 0


@pytest.mark.parametrize("kind", ["contradictory_plan", "narrow_saved_days", "late_window_start", "early_window_end", "provider_bbox", "missing_optical_record", "incomplete_provider_query"])
def test_noncovering_or_contradictory_query_metadata_is_refused(tmp_path, kind):
    context = [85.29, 28.11, 85.32, 28.14]
    doc = queried_document(context)
    plan = {"schema": "akash-fresh-case-runner-v1", "case": {"bbox": BBOX, "context_bbox": context, "event_date": EVENT}}
    if kind == "contradictory_plan":
        doc["bbox"] = [85.28, 28.10, 85.33, 28.15]
    elif kind == "narrow_saved_days":
        doc["max_days"] = 12
    elif kind == "late_window_start":
        doc["search_records"][0]["query"]["datetime"] = "2024-09-05T00:00:00Z/2024-10-23T00:00:00Z"
    elif kind == "early_window_end":
        doc["search_records"][0]["query"]["datetime"] = "2024-09-04T00:00:00Z/2024-10-22T00:00:00Z"
    elif kind == "provider_bbox":
        doc["search_records"][0]["query"]["bbox"] = [85.305, 28.12, 85.32, 28.13]
    elif kind == "missing_optical_record":
        doc["search_records"][0]["collection"] = "sentinel-1-grd"
    else:
        doc["search_records"][0]["search_complete"] = False
    args = prepare_args(tmp_path, doc, plan)
    with pytest.raises(ValueError):
        adapter.prepare(args)
    assert not args.out.exists()
