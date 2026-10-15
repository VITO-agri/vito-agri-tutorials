"""Run against an installed wheel, not an editable source checkout."""

import json
from importlib import metadata, resources

import pytest
from packaging.requirements import Requirement

from vito_agri_tutorials import __version__
from vito_agri_tutorials.tutorials import copy_tutorial, discover_tutorials


TUTORIALS = {
    "ndvi-anomalies": "1_ndvi_anomalies.ipynb",
    "asis": "2_asis_data_access.ipynb",
    "wapor": "3_wapor_data_access.ipynb",
    "vici": "4_vici.ipynb",
}


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
        and requirement.marker.evaluate({"extra": "ndvi-anomalies"})
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


def test_every_tutorial_has_an_installable_extra():
    extras = metadata.metadata("vito-agri-tutorials").get_all("Provides-Extra")
    assert set(TUTORIALS) <= set(extras)


@pytest.mark.parametrize("tutorial,filename", TUTORIALS.items())
def test_copy_notebook(tmp_path, tutorial, filename):
    destination = tmp_path / "mystorage"
    notebook = copy_tutorial(tutorial, destination)

    assert notebook == destination / "vito_agri_tutorials" / "drought" / filename
    bundled = resources.files("vito_agri_tutorials").joinpath(
        "_notebooks", "drought", notebook.name
    )
    assert notebook.read_bytes() == bundled.read_bytes()
    content = json.loads(notebook.read_text())
    code = [
        "".join(cell["source"])
        for cell in content["cells"]
        if cell["cell_type"] == "code"
    ]
    assert code[0].startswith("%pip install --quiet")
    assert "from vito_agri_tutorials.tutorials import install_tutorial" in code[0]
    assert code[1] == f'install_tutorial("{tutorial}")'
    assert not any("sys.path" in cell for cell in code)
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in content["cells"]
        if cell["cell_type"] == "code"
    )


def test_copy_preserves_student_changes(tmp_path):
    notebook = copy_tutorial("ndvi-anomalies", tmp_path)
    notebook.write_text("student changes")

    with pytest.raises(FileExistsError):
        copy_tutorial("ndvi-anomalies", tmp_path)

    assert notebook.read_text() == "student changes"


def test_unknown_tutorial_does_not_create_destination(tmp_path):
    destination = tmp_path / "unused"

    with pytest.raises(ValueError, match="Available tutorials: ndvi"):
        copy_tutorial("../ndvi", destination)

    assert not destination.exists()


def test_destination_must_be_a_directory(tmp_path):
    destination = tmp_path / "file"
    destination.write_text("existing file")

    with pytest.raises(NotADirectoryError):
        copy_tutorial("ndvi-anomalies", destination)

    assert destination.read_text() == "existing file"


def test_discover_tutorials(capsys):
    discover_tutorials()
    assert capsys.readouterr().out == (
        "drought:\n  - ndvi-anomalies\n  - asis\n  - wapor\n  - vici\n"
    )


def test_discovery_groups_categories(monkeypatch, capsys):
    monkeypatch.setattr(
        "vito_agri_tutorials.tutorials._TUTORIALS",
        {"example": "openeo/example.ipynb", "fields": "parcel/fields.ipynb"},
    )
    discover_tutorials()
    assert capsys.readouterr().out == "openeo:\n  - example\nparcel:\n  - fields\n"


def test_asis_includes_access_guide(tmp_path):
    notebook = copy_tutorial("asis", tmp_path)
    guide = notebook.with_name("ASIS_README.md")
    bundled = resources.files("vito_agri_tutorials").joinpath(
        "_notebooks", "drought", "ASIS_README.md"
    )
    assert guide.read_bytes() == bundled.read_bytes()


def test_supporting_file_conflict_does_not_copy_notebook(tmp_path):
    folder = tmp_path / "vito_agri_tutorials" / "drought"
    folder.mkdir(parents=True)
    guide = folder / "ASIS_README.md"
    guide.write_text("student notes")
    with pytest.raises(FileExistsError):
        copy_tutorial("asis", tmp_path)
    assert guide.read_text() == "student notes"
    assert not (folder / TUTORIALS["asis"]).exists()


def test_all_tutorials_share_drought_folder(tmp_path):
    notebooks = [copy_tutorial(tutorial, tmp_path) for tutorial in TUTORIALS]
    assert len({notebook.parent for notebook in notebooks}) == 1
