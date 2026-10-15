"""Offline import checks for the additional drought tutorial extras."""

import importlib
from importlib import metadata

import pytest
from packaging.requirements import Requirement


@pytest.mark.parametrize(
    "extra,modules",
    [
        ("asis", ("vito_agri_tutorials.fao.asis", "vito_agri_tutorials.fao.wpop")),
        ("wapor", ("vito_agri_tutorials.fao.wapor",)),
        ("vici", ("vito_agri_tutorials.vici", "netCDF4")),
    ],
)
def test_tutorial_imports(extra, modules):
    for module in (*modules, "vito_agri_tutorials.utils.map"):
        importlib.import_module(module)
    requirements = [
        Requirement(value) for value in metadata.requires("vito-agri-tutorials") or []
    ]
    required = {
        item.name.lower()
        for item in requirements
        if item.marker and item.marker.evaluate({"extra": extra})
    }
    assert "rasterio" in required
    assert "ipyleaflet" in required
    if extra == "vici":
        assert {"scikit-learn", "netcdf4", "tqdm", "openeo"} <= required
    else:
        assert "scikit-learn" not in required
