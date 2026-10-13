"""Completion transport-plan tests; no remote assets or checking references."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import rasterio
from affine import Affine
from pyproj import CRS

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("optical_completion", SCRIPTS / "complete_optical_inputs_20261011.py")
completion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(completion)


def test_missing_plan_preserves_original_order_and_bands():
    bands = completion.BANDS
    entries = {"8": {k: {} for k in bands}, "11": {"scl": {}, "blue": {}}, "10": {"scl": {}}}
    assert completion.acquisition_plan([8, 11, 10], entries) == [
        {"source_index": 11, "missing_bands": ["red", "nir", "swir16"]},
        {"source_index": 10, "missing_bands": list(bands)}]


def test_no_unselected_or_alternate_source_added():
    assert completion.acquisition_plan([2], {"1": {}, "2": {"scl": {}}, "3": {}}) == [
        {"source_index": 2, "missing_bands": list(completion.BANDS)}]


def test_cached_entry_rejects_changed_crop_and_asset(tmp_path):
    path = tmp_path / "scl-dn.tif"
    affine = Affine(20, 0, 0, 0, -20, 100)
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1,
                       dtype="uint8", transform=affine, crs="EPSG:32645", nodata=0) as ds:
        ds.write(np.full((2, 2), 4, dtype="uint8"), 1)
    asset = {"proj:transform": list(affine), "proj:shape": [2, 2], "gsd": 20,
             "raster:bands": [{"nodata": 0}]}
    item = {"properties": {"proj:epsg": 32645}, "assets": {"scl": asset}}
    from rasterio.windows import Window
    grid = (CRS.from_epsg(32645), affine, 2, 2)
    completion.dump(tmp_path / "scl-crop-receipt.json", {"crop": completion.record(path), "asset": asset})
    assert completion.cached_entry(tmp_path, item, "scl", Window(0, 0, 2, 2), grid)
    changed = {**item, "assets": {"scl": {**asset, "gsd": 10}}}
    with pytest.raises(ValueError, match="asset"):
        completion.cached_entry(tmp_path, changed, "scl", Window(0, 0, 2, 2), grid)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Frozen file changed"):
        completion.cached_entry(tmp_path, item, "scl", Window(0, 0, 2, 2), grid)
