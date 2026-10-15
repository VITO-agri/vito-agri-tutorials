"""Export packaged teaching notebooks without importing scientific dependencies."""

from importlib.resources import files
from pathlib import Path
from shutil import copyfileobj


_TUTORIALS = {"ndvi": "1_ndvi_anomalies.ipynb"}


def copy_tutorial(tutorial: str, destination: str | Path) -> Path:
    """Copy a tutorial into a working directory and return its notebook path.

    Currently only ``ndvi`` is packaged. Existing notebooks are never overwritten;
    choose another destination to obtain a fresh copy. On CDSE, use a destination
    inside ``mystorage`` so that the notebook and its results survive server stops.

    Raises:
        ValueError: The tutorial name is unknown.
        FileExistsError: The destination notebook already exists.
        OSError: The source cannot be read or the destination cannot be written.
    """
    if tutorial not in _TUTORIALS:
        raise ValueError(
            f"Unknown tutorial {tutorial!r}. Available tutorials: {', '.join(_TUTORIALS)}"
        )

    filename = _TUTORIALS[tutorial]
    source = files("vito_agri_tutorials").joinpath("_notebooks", tutorial, filename)
    destination = Path(destination).expanduser()
    notebook = destination / filename

    with source.open("rb") as source_file:
        destination.mkdir(parents=True, exist_ok=True)
        with notebook.open("xb") as target_file:
            copyfileobj(source_file, target_file)

    return notebook
