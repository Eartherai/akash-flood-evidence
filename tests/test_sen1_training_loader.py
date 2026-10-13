"""Metadata handling tests use synthetic arrays, never holdout labels."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

pytest.importorskip("sklearn")
_path = Path(__file__).resolve().parents[1] / "scripts/train_sen1_cpu.py"
_spec = importlib.util.spec_from_file_location("sen1_training_loader", _path)
loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loader)


def write_chip(cache, *, name="Ghana_24858", label_nodata=0, shift_pixels=0):
    (cache / "s1").mkdir(); (cache / "label").mkdir()
    source_transform = Affine(.00009, 0., 20., 0., -.00009, 10.)
    row = {"s1": name + "_S1Hand.tif", "label": name + "_LabelHand.tif", "collection": "Ghana"}
    with rasterio.open(cache / "s1" / row["s1"], "w", driver="GTiff", width=512, height=512,
                       count=2, dtype="float32", crs="EPSG:4326", transform=source_transform, nodata=np.nan) as ds:
        ds.write(np.zeros((2, 512, 512), dtype="float32"))
    labels = np.zeros((512, 512), dtype="int16"); labels[0, 0] = -1; labels[0, 1] = 1
    with rasterio.open(cache / "label" / row["label"], "w", driver="GTiff", width=512, height=512,
                       count=1, dtype="int16", crs="EPSG:4326",
                       transform=source_transform * Affine.translation(shift_pixels, 0), nodata=label_nodata) as ds:
        ds.write(labels, 1)
    return row, labels


def test_documented_label_zero_retained_despite_one_source_metadata_exception(tmp_path):
    row, expected = write_chip(tmp_path)
    db, observation_mask, labels, metadata = loader.read_chip(tmp_path, row)
    assert np.array_equal(labels, expected)
    assert observation_mask.all() and (db == 0).all()
    assert (labels == 0).sum() == 512 * 512 - 2
    assert metadata["label_nodata_metadata"] == 0
    assert metadata["label_validity"].startswith("raw class -1 ignored")


def test_unknown_label_nodata_zero_exception_rejected(tmp_path):
    row, _ = write_chip(tmp_path, name="Ghana_other")
    with pytest.raises(ValueError, match="nodata metadata"):
        loader.read_chip(tmp_path, row)


def test_loader_rejects_real_half_pixel_shift_without_resampling(tmp_path):
    row, _ = write_chip(tmp_path, label_nodata=None, shift_pixels=.5)
    with pytest.raises(ValueError, match="alignment"):
        loader.read_chip(tmp_path, row)
