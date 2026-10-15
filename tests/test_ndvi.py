"""Offline smoke tests for the NDVI extra; no CDSE credentials are required."""

import geopandas as gpd
import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from vito_agri_tutorials.openeo.extract_data import (
    get_dekads_between_dates,
    get_ndvi_1km_point,
)
from vito_agri_tutorials.utils.anomalies import compute_anomalies
from vito_agri_tutorials.utils.map import ui_map


def test_point_widget_and_geopackage(tmp_path, monkeypatch):
    def no_download(*args, **kwargs):
        raise AssertionError("The NDVI point widget must not download resources")

    monkeypatch.setattr("vito_agri_tutorials.resources.download_file", no_download)
    widget = ui_map(geometry_type="point")
    widget.draw_control.last_draw = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [5.1, 51.2]},
        "properties": {},
    }
    widget.inputfield.value = "test-point"
    widget.handle_submit(None)

    points = widget.get_objects()
    outfile = tmp_path / "points.gpkg"
    points.to_file(outfile, driver="GPKG")
    restored = gpd.read_file(outfile)

    assert restored["id"].tolist() == ["test-point"]
    assert restored.crs.to_epsg() == 4326
    assert restored.geometry.x.tolist() == [5.1]
    assert restored.geometry.y.tolist() == [51.2]


def test_ndvi_helpers_import_and_expected_dates():
    assert callable(get_ndvi_1km_point)
    dates = get_dekads_between_dates("2018-01-01", "2024-12-31")
    assert len(dates) == 7 * 36
    assert dates[0] == "20180101"
    assert dates[-1] == "20241221"


def test_anomaly_output_shapes_and_values(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    series = np.tile(np.arange(36), 7) + np.repeat(np.arange(7), 36)

    diff, relative, standardized, vci = compute_anomalies(
        series, nyears=7, ref_period=5, label_years=["2023", "2024"]
    )

    for result in (diff, relative, standardized, vci):
        assert np.asarray(result).shape == (2, 36)
        assert np.isfinite(result).all()
    np.testing.assert_allclose(diff[0], 3)
    np.testing.assert_allclose(diff[1], 4)
    np.testing.assert_allclose(standardized[0], 3 / np.sqrt(2))
    np.testing.assert_allclose(vci[0], 1.25)
    np.testing.assert_allclose(vci[1], 1.5)
