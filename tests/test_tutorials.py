"""Run against an installed wheel, not an editable source checkout."""

import json
from importlib import metadata, resources

import pytest
from packaging.requirements import Requirement

from vito_agri_tutorials import __version__
from vito_agri_tutorials.tutorials import copy_tutorial


def test_base_has_no_third_party_requirements():
    requirements = metadata.requires("vito-agri-tutorials") or []
    assert all(
        Requirement(requirement).marker is not None
        and not Requirement(requirement).marker.evaluate({"extra": ""})
        for requirement in requirements
    )
    assert metadata.version("vito-agri-tutorials") == __version__


def test_ndvi_extra_has_required_packages():
    requirements = [
        Requirement(requirement)
        for requirement in metadata.requires("vito-agri-tutorials") or []
    ]
    ndvi = {
        requirement.name
        for requirement in requirements
        if requirement.marker is not None
        and requirement.marker.evaluate({"extra": "ndvi"})
    }
    assert ndvi == {
        "openeo",
        "ipyleaflet",
        "ipywidgets",
        "ipython",
        "geopandas",
        "shapely",
        "pyproj",
        "matplotlib",
        "geojson",
        "numpy",
        "pandas",
        "scipy",
        "xarray",
        "loguru",
        "requests",
    }


def test_copy_ndvi_notebook(tmp_path):
    destination = tmp_path / "mystorage" / "ndvi"
    notebook = copy_tutorial("ndvi", destination)

    assert notebook == destination / "1_ndvi_anomalies.ipynb"
    bundled = resources.files("vito_agri_tutorials").joinpath(
        "_notebooks", "ndvi", notebook.name
    )
    assert notebook.read_bytes() == bundled.read_bytes()
    content = json.loads(notebook.read_text())
    code = [
        "".join(cell["source"])
        for cell in content["cells"]
        if cell["cell_type"] == "code"
    ]
    assert code[0].startswith('%pip install "vito-agri-tutorials[ndvi] @ https://')
    assert not any("sys.path" in cell for cell in code)
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in content["cells"]
        if cell["cell_type"] == "code"
    )


def test_copy_preserves_student_changes(tmp_path):
    notebook = copy_tutorial("ndvi", tmp_path)
    notebook.write_text("student changes")

    with pytest.raises(FileExistsError):
        copy_tutorial("ndvi", tmp_path)

    assert notebook.read_text() == "student changes"


def test_unknown_tutorial_does_not_create_destination(tmp_path):
    destination = tmp_path / "unused"

    with pytest.raises(ValueError, match="Available tutorials: ndvi"):
        copy_tutorial("../ndvi", destination)

    assert not destination.exists()


def test_destination_must_be_a_directory(tmp_path):
    destination = tmp_path / "file"
    destination.write_text("existing file")

    with pytest.raises(FileExistsError):
        copy_tutorial("ndvi", destination)

    assert destination.read_text() == "existing file"
