from pathlib import Path
import geopandas as gpd


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

    return gpd.read_file(indir / _filenames[layername])
