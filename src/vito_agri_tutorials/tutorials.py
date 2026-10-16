"""Prepare teaching notebooks without importing scientific dependencies."""

import subprocess
import sys
from html import escape
from importlib.resources import files
from pathlib import Path
from shutil import copyfileobj

from ._version import __version__


_TUTORIALS = {
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
_LAI_FILES = tuple(
    f"timeseries_decomposition/data/A_visualisation/area_{area}.csv"
    for area in "ABCD"
)
_SUPPORTING_FILES = {
    "asis": ("drought/ASIS_README.md",),
    "timeseries-visualisation": _LAI_FILES,
    "timeseries-preprocessing": (
        "timeseries_decomposition/data/B_preprocessing/forest_senLAI.csv",
    ),
    "timeseries-decomposition": _LAI_FILES,
    "timeseries-change": (
        "timeseries_decomposition/data/D_change_detection/ts_fire.csv",
        "timeseries_decomposition/data/D_change_detection/ts_fire2.csv",
        "timeseries_decomposition/auxiliary/1610_hi-res.jpg",
        "timeseries_decomposition/auxiliary/fire.jpg",
        "timeseries_decomposition/auxiliary/seasonalbreak_TreeMort.jpg",
    ),
}


def discover_tutorials() -> None:
    """Display the available tutorial names grouped by repository category."""
    categories: dict[str, list[str]] = {}
    for tutorial, relative_path in _TUTORIALS.items():
        category = relative_path.split("/")[0]
        categories.setdefault(category, []).append(tutorial)
    for category, tutorials in categories.items():
        print(f"{category}:")
        for tutorial in tutorials:
            print(f"  - {tutorial}")


def install_tutorial(tutorial: str) -> None:
    """Install a tutorial's requirements for the already installed package version.

    Run inside a notebook after installing the base package. Pip output is hidden
    on success and shown in expandable details on failure.

    Raises:
        ValueError: The tutorial name is unknown.
        RuntimeError: Pip failed; details are displayed below the cell.
        OSError: The pip process could not be started.
    """
    if tutorial not in _TUTORIALS:
        raise ValueError(
            f"Unknown tutorial {tutorial!r}. Available tutorials: {', '.join(_TUTORIALS)}"
        )

    from IPython.display import HTML, display

    print("Installing the tutorial. This may take a few minutes...", flush=True)
    installation = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            f"vito-agri-tutorials[{tutorial}]=={__version__}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if installation.returncode != 0:
        print(
            "Installation failed. Please share the details below with your instructor."
        )
        display(
            HTML(
                "<details><summary>Show installation details</summary><pre>"
                + escape(installation.stdout)
                + "</pre></details>"
            )
        )
        raise RuntimeError(
            "Installation failed. Do not continue with the exercise yet."
        )

    print(
        "Installation succeeded! Choose Kernel > Restart Kernel, then run the next cell."
    )


def copy_tutorial(tutorial: str, destination: str | Path) -> Path:
    """Copy a tutorial under destination/vito_agri_tutorials and return its path.

    The repository's tutorial folders are preserved, for example ``drought``.
    Supporting instructions, data and illustrations retain their relative paths.
    Identical supporting files can be shared by tutorials. Existing files are
    never overwritten; choose another destination to obtain a fresh copy.
    On CDSE, use ``mystorage`` as the destination to preserve work between sessions.

    Raises:
        ValueError: The tutorial name is unknown.
        FileExistsError: The notebook exists or a supporting file differs.
        OSError: The source cannot be read or the destination cannot be written.
    """
    if tutorial not in _TUTORIALS:
        raise ValueError(
            f"Unknown tutorial {tutorial!r}. Available tutorials: {', '.join(_TUTORIALS)}"
        )

    relative_paths = (_TUTORIALS[tutorial], *_SUPPORTING_FILES.get(tutorial, ()))
    destination = Path(destination).expanduser() / "vito_agri_tutorials"
    source_root = files("vito_agri_tutorials").joinpath("_notebooks")
    to_copy = []
    for relative_path in relative_paths:
        target = destination / relative_path
        source = source_root.joinpath(*relative_path.split("/"))
        if not source.is_file():
            raise FileNotFoundError(
                f"Packaged tutorial file is missing: {relative_path}"
            )
        if target.exists():
            if (
                relative_path != _TUTORIALS[tutorial]
                and target.is_file()
                and target.read_bytes() == source.read_bytes()
            ):
                continue
            raise FileExistsError(
                f"Will not overwrite existing tutorial file: {target}"
            )
        to_copy.append(relative_path)

    for relative_path in to_copy:
        source = source_root.joinpath(*relative_path.split("/"))
        target = destination / relative_path
        with source.open("rb") as source_file:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as target_file:
                copyfileobj(source_file, target_file)

    return destination / _TUTORIALS[tutorial]
