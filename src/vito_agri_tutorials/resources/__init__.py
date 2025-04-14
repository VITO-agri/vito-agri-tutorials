from pathlib import Path
import geopandas as gpd
from loguru import logger

from vito_agri_tutorials.utils.artifactory import download_file


_filenames = {
    "GAUL_0": "GAUL_0.gpkg",
    "GAUL_1": "GAUL_1.gpkg",
    "GAUL_2": "GAUL_2.gpkg",
}

layers_description = {
    "GAUL_0": "FAO GAUL Administrative regions country level",
    "GAUL_1": "FAO GAUL Administrative regions state level",
    "GAUL_2": "FAO GAUL Administrative regions district level",
}


def load_resource(layername):

    indir = Path(__file__).resolve().parent

    infile = indir / _filenames[layername]

    if not infile.exists():
        logger.info(
            f"{layername} not found in resources, downloading from Artifactory..."
        )
        srcpath = "resources/" + _filenames[layername]
        download_file(indir, srcpath)

    return gpd.read_file(infile)
