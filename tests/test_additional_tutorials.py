"""Offline dependency and data checks for the parcel, openEO and timeseries tutorials."""

import ast
import json
from importlib import metadata

import matplotlib
import pandas as pd
import pytest
from packaging.requirements import Requirement

from vito_agri_tutorials.tutorials import copy_tutorial


matplotlib.use("Agg")

REQUIRED_PACKAGES = {
    "parcel-preprocessing": {
        "ipython", "numpy", "pandas", "matplotlib", "xarray", "netcdf4",
        "dask", "numba", "python-dateutil", "loguru", "requests",
    },
    "parcel-phenology": {
        "ipython", "numpy", "pandas", "matplotlib", "xarray", "netcdf4",
        "dask", "numba", "python-dateutil", "scipy", "scikit-image",
        "loguru", "requests",
    },
    "parcel-croptype": {
        "ipython", "numpy", "pandas", "matplotlib", "xarray", "netcdf4",
        "dask", "numba", "python-dateutil", "geopandas", "shapely", "pyproj",
        "rasterio", "scipy", "scikit-learn", "imbalanced-learn", "catboost",
        "seaborn", "loguru", "requests",
    },
    "sentinel2-retrieval": {
        "ipython", "openeo", "ipyleaflet", "ipywidgets", "geopandas", "shapely",
        "pyproj", "matplotlib", "geojson", "numpy", "pandas", "xarray",
        "netcdf4", "loguru", "requests",
    },
    "timeseries-visualisation": {"ipython", "numpy", "pandas", "matplotlib"},
    "timeseries-preprocessing": {"ipython", "numpy", "pandas", "matplotlib", "scipy"},
    "timeseries-decomposition": {
        "ipython", "numpy", "pandas", "matplotlib", "statsmodels",
    },
    "timeseries-change": {
        "ipython", "numpy", "pandas", "matplotlib", "statsmodels",
    },
}


@pytest.mark.parametrize("tutorial,expected", REQUIRED_PACKAGES.items())
def test_extra_declares_required_packages(tutorial, expected):
    requirements = [
        Requirement(value) for value in metadata.requires("vito-agri-tutorials") or []
    ]
    actual = {
        item.name.lower()
        for item in requirements
        if item.marker and item.marker.evaluate({"extra": tutorial})
    }
    assert actual == expected


@pytest.mark.parametrize("tutorial", REQUIRED_PACKAGES)
def test_notebook_imports(tmp_path, tutorial):
    notebook = copy_tutorial(tutorial, tmp_path)
    namespace = {}
    for cell in json.loads(notebook.read_text())["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or source.startswith("%pip"):
            continue
        tree = ast.parse(source.replace("< YOUR CODE HERE >", "None"))
        imports = [
            statement
            for statement in tree.body
            if isinstance(statement, (ast.Import, ast.ImportFrom))
        ]
        exec(compile(ast.Module(body=imports, type_ignores=[]), str(notebook), "exec"), namespace)


@pytest.mark.parametrize(
    "tutorial",
    [name for name in REQUIRED_PACKAGES if name.startswith("timeseries-")],
)
def test_timeseries_csvs_are_readable(tmp_path, tutorial):
    notebook = copy_tutorial(tutorial, tmp_path)
    csvs = list((notebook.parent / "data").rglob("*.csv"))
    assert csvs
    for csv in csvs:
        data = pd.read_csv(csv)
        assert not data.empty
        assert len(data.columns) >= 2
