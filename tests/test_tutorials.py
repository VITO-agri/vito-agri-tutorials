"""Run against an installed wheel, not an editable source checkout."""

import ast
import json
import re
from importlib import metadata, resources
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from vito_agri_tutorials import __version__
from vito_agri_tutorials.tutorials import copy_tutorial, discover_tutorials


TUTORIALS = {
    "ndvi-anomalies": "drought/1_ndvi_anomalies.ipynb",
    "asis": "drought/2_asis_data_access.ipynb",
    "wapor": "drought/3_wapor_data_access.ipynb",
    "vici": "drought/4_vici.ipynb",
    "parcel-preprocessing": "parcel/1_Preprocessing.ipynb",
    "parcel-phenology": "parcel/2_Field_delineation_and_phenology.ipynb",
    "parcel-croptype": "parcel/3_Croptype_mapping.ipynb",
    "sentinel2-retrieval": "openeo/Sentinel-2-retrieval.ipynb",
    "timeseries-visualisation": "timeseries_decomposition/A_visualisation.ipynb",
    "timeseries-preprocessing": "timeseries_decomposition/B_preprocessing.ipynb",
    "timeseries-decomposition": "timeseries_decomposition/C_decomposition.ipynb",
    "timeseries-change": "timeseries_decomposition/D_change.ipynb",
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


@pytest.mark.parametrize("tutorial,relative_path", TUTORIALS.items())
def test_copy_notebook(tmp_path, tutorial, relative_path):
    destination = tmp_path / "mystorage"
    notebook = copy_tutorial(tutorial, destination)

    assert notebook == destination / "vito_agri_tutorials" / relative_path
    bundled = resources.files("vito_agri_tutorials").joinpath(
        "_notebooks", *relative_path.split("/")
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
    for cell in code[2:]:
        ast.parse(cell.replace("< YOUR CODE HERE >", "None"))
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
        "parcel:\n  - parcel-preprocessing\n  - parcel-phenology\n  - parcel-croptype\n"
        "openeo:\n  - sentinel2-retrieval\n"
        "timeseries_decomposition:\n  - timeseries-visualisation\n"
        "  - timeseries-preprocessing\n  - timeseries-decomposition\n"
        "  - timeseries-change\n"
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
    assert not (folder / Path(TUTORIALS["asis"]).name).exists()


def test_all_tutorials_preserve_category_folders(tmp_path):
    notebooks = [copy_tutorial(tutorial, tmp_path) for tutorial in TUTORIALS]
    assert {notebook.parent.name for notebook in notebooks} == {
        "drought", "parcel", "openeo", "timeseries_decomposition"
    }


@pytest.mark.parametrize(
    "tutorial",
    [name for name in TUTORIALS if name.startswith("timeseries-")],
)
def test_timeseries_inputs_and_illustrations_are_copied(tmp_path, tutorial):
    notebook = copy_tutorial(tutorial, tmp_path)
    content = json.loads(notebook.read_text())
    text = "\n".join("".join(cell["source"]) for cell in content["cells"])
    references = set(
        re.findall(r"\./((?:data|auxiliary)/[\w/.-]+\.(?:csv|jpg|png))", text)
    )
    assert references
    for relative_path in references:
        target = notebook.parent / relative_path
        bundled = resources.files("vito_agri_tutorials").joinpath(
            "_notebooks", "timeseries_decomposition", *relative_path.split("/")
        )
        assert target.read_bytes() == bundled.read_bytes()
    if tutorial in {"timeseries-visualisation", "timeseries-decomposition"}:
        for area in "ABCD":
            assert (notebook.parent / "data/A_visualisation" / f"area_{area}.csv").is_file()


@pytest.mark.parametrize("reverse", [False, True])
def test_timeseries_tutorials_reuse_identical_data(tmp_path, reverse):
    names = ["timeseries-visualisation", "timeseries-decomposition"]
    if reverse:
        names.reverse()
    first = copy_tutorial(names[0], tmp_path)
    shared = first.parent / "data/A_visualisation/area_A.csv"
    modified = shared.stat().st_mtime_ns
    second = copy_tutorial(names[1], tmp_path)
    assert second.is_file()
    assert shared.stat().st_mtime_ns == modified


def test_modified_shared_data_prevents_partial_copy(tmp_path):
    first = copy_tutorial("timeseries-visualisation", tmp_path)
    shared = first.parent / "data/A_visualisation/area_A.csv"
    shared.write_text("student changes")

    with pytest.raises(FileExistsError, match="area_A.csv"):
        copy_tutorial("timeseries-decomposition", tmp_path)

    assert shared.read_text() == "student changes"
    assert not (first.parent / "C_decomposition.ipynb").exists()


def test_package_does_not_include_legacy_or_generated_notebooks():
    root = resources.files("vito_agri_tutorials").joinpath("_notebooks")

    def walk(folder):
        for child in folder.iterdir():
            yield child
            if child.is_dir():
                yield from walk(child)

    entries = list(walk(root))
    assert not any(entry.name == ".ipynb_checkpoints" for entry in entries)
    assert sum(entry.name.endswith(".ipynb") for entry in entries) == len(TUTORIALS)
