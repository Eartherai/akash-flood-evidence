import io
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from rasterio.control import GroundControlPoint
from rasterio.transform import GCPTransformer
from rasterio.windows import Window

from akash_flood.s1_crop import (BoundedHTTP, CalibrationLUT, DownloadBudgetExceeded,
                               HTTPRangeReader, calibrate_sigma0, interpolate_sigma_lut,
                               native_window_for_bbox, parse_calibration_xml, shifted_gcps)


def test_original_pixel_coordinates_determine_calibration_without_uint16_overflow():
    # A(x,y)=2+0.1x+0.2y is an analytic bilinear LUT; cropped origin is not (0,0).
    lut = CalibrationLUT(np.array([0., 100.]), [np.array([0., 200.])] * 2,
                         [np.array([2., 22.]), np.array([22., 42.])], {})
    rows, columns = np.array([45., 46.]), np.array([70., 71., 72.])
    expected = 2 + .2 * rows[:, None] + .1 * columns[None, :]
    actual = interpolate_sigma_lut(lut, rows, columns)
    np.testing.assert_allclose(actual, expected)
    raw = np.full(actual.shape, 65535, dtype=np.uint16)
    power, db, valid = calibrate_sigma0(raw, actual, np.full(raw.shape, 255, dtype=np.uint8))
    np.testing.assert_allclose(power, 65535.0 ** 2 / expected ** 2)
    np.testing.assert_allclose(db, 10 * np.log10(power))
    assert valid.all()


def test_calibration_refuses_extrapolation_and_keeps_zero_or_masked_pixels_unknown():
    lut = CalibrationLUT(np.array([0., 10.]), [np.array([0., 20.])] * 2,
                         [np.array([2., 2.])] * 2, {})
    with pytest.raises(ValueError, match="line coverage"):
        interpolate_sigma_lut(lut, [11], [1])
    with pytest.raises(ValueError, match="range coverage"):
        interpolate_sigma_lut(lut, [1], [21])
    power, db, valid = calibrate_sigma0(np.array([[0, 20, 20]], dtype=np.uint16),
                                      np.full((1, 3), 2.), np.array([[255, 0, 255]]))
    assert valid.tolist() == [[False, False, True]]
    assert np.isnan(power[0, :2]).all() and np.isnan(db[0, :2]).all()
    assert power[0, 2] == 100. and db[0, 2] == 20.


def test_source_gcps_shift_exactly_with_native_crop_window():
    gcps = [GroundControlPoint(row=r, col=c, x=85 + c * .001, y=29 - r * .001, z=0)
            for r in [0, 100, 200] for c in [0, 100, 200]]
    window = native_window_for_bbox(gcps, [85.05, 28.88, 85.08, 28.9], 201, 201, halo=8)
    assert window.row_off <= 100 and window.col_off <= 50
    shifted = shifted_gcps(gcps, window)
    with GCPTransformer(gcps) as original, GCPTransformer(shifted) as crop:
        expected = original.xy([window.row_off + 3], [window.col_off + 4])
        actual = crop.xy([3], [4])
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_rfi_xml_is_not_accepted_as_calibration():
    with pytest.raises(ValueError, match="Expected calibration XML"):
        parse_calibration_xml(b"<rfi><line>0</line></rfi>")


def test_written_native_tiff_retains_source_gcp_projection_without_invented_affine(tmp_path):
    import rasterio

    entry = Path(__file__).resolve().parents[1] / "scripts/fetch_s1_crop.py"
    spec = importlib.util.spec_from_file_location("fetch_s1_crop_test", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    points = [GroundControlPoint(row=r, col=c, x=85 + c * .001, y=29 - r * .001)
              for r in [0, 100] for c in [0, 100]]
    path = tmp_path / "native.tif"
    module.write_native(path, np.ones((10, 10), dtype="uint16"), points,
                        rasterio.crs.CRS.from_epsg(4326), 0, "raw amplitude")
    with rasterio.open(path) as source:
        assert source.crs is None
        assert source.gcps[1].to_epsg() == 4326
        assert len(source.gcps[0]) == 4
        assert source.read(1).sum() == 100


def test_range_reader_seeks_without_whole_file_download():
    data = bytes(range(256)) * 600

    class Client:
        def __init__(self):
            self.calls = []

        def range(self, url, start, stop):
            self.calls.append((start, stop))
            return data[start:stop], len(data)

    client = Client()
    reader = HTTPRangeReader("https://public.example/data.tiff", client)
    reader.seek(100)
    assert reader.read(10) == data[100:110]
    reader.seek(-10, io.SEEK_END)
    assert reader.read(20) == data[-10:]
    assert client.calls == [(0, 65536), (len(data) - 10, len(data))]
    with pytest.raises(RuntimeError, match="Unbounded"):
        reader.read()
    with pytest.raises(DownloadBudgetExceeded):
        BoundedHTTP(1024)._available(1025)
