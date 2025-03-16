from datetime import datetime
import pandas as pd
from pathlib import Path
import geopandas as gpd
import numpy as np

from vito_agri_tutorials.openeo.utils import (
    openeo_timeseries_json_to_df,
    multi_index_df_to_single_index,
)
from vito_agri_tutorials.utils.geo import gdf_to_geojson
from vito_agri_tutorials.utils.upperenvelop import upper_envelop
from vito_agri_tutorials.openeo.auth import connect_openeo


SCALE_FACTOR = {
    "NDVI": 0.004,
}

OFFSET = {
    "NDVI": -0.08,
}


def _split_temporal_extent_cgls_1km(start_date: str, end_date: str) -> tuple[list[str]]:

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Check if temporal extent spans 1km period, 300m period, or both
    start_300m = datetime(2020, 7, 1)
    if (start_date < start_300m) and (end_date < start_300m):
        temporal_extent_1km = [
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        ]
        temporal_extent_300m = None
    elif (start_date < start_300m) and (end_date >= start_300m):
        temporal_extent_1km = [
            start_date.strftime("%Y-%m-%d"),
            datetime(2020, 6, 30).strftime("%Y-%m-%d"),
        ]
        temporal_extent_300m = [
            start_300m.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        ]
    elif (start_date >= start_300m) and (end_date >= start_300m):
        temporal_extent_1km = None
        temporal_extent_300m = [
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        ]

    print(f"Temporal extent 1km: {temporal_extent_1km}")
    print(f"Temporal extent 300m: {temporal_extent_300m}")

    return temporal_extent_1km, temporal_extent_300m


def _dwnld_cgls_ndvi_point(connection, collection, gdf, temporal_extent, outdir, res):

    # First determine which bands to download (NOBS not available for 1km)
    if res == "300m":
        bands = ["NDVI", "NOBS"]
    else:
        bands = ["NDVI"]

    # Check if already downloaded or not
    outfile = outdir / "timeseries.json"
    if not outfile.exists():

        # Load collection
        ndvi_cube = connection.load_collection(
            collection,
            temporal_extent=temporal_extent,
            bands=bands,
        )

        spatial_extent = gdf_to_geojson(gdf)

        ndvi_ts = ndvi_cube.aggregate_spatial(geometries=spatial_extent, reducer="mean")

        # Create and run job
        job = ndvi_ts.create_job(title=f"{collection} download point")
        job.start_and_wait()

        # Get results
        results = job.get_results()
        outdir.mkdir(exist_ok=True, parents=True)
        results.download_files(target=outdir)

    # Convert timeseries json to dataframe
    df = openeo_timeseries_json_to_df(outfile, gdf["id"].values, bands)

    if res == "300m":
        # additional masking using NOBS
        df["NDVI"] = df["NDVI"].where(df["NOBS"] != 0, np.nan)
        df = df.drop(columns=["NOBS"])

    return multi_index_df_to_single_index(df, "NDVI")


def get_ndvi_1km_point(
    gdf: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    outdir: Path,
    upperenvelop=True,
) -> pd.DataFrame:
    """
    Download 1km NDVI data for one or severel point locations.
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing point locations
    start_date : str
        Start date of the temporal extent (format YYYY-mm-dd)
    end_date : str
        End date of the temporal extent (format YYYY-mm-dd)
    outdir : Path
        Output directory for downloaded NDVI data
    Returns
    -------
    DataFrame
        DataFrame containing NDVI timeseries data
    """

    # connect to openeo backend
    connection = connect_openeo("fed")

    # Split temporal extent
    temporal_extent_1km, temporal_extent_300m = _split_temporal_extent_cgls_1km(
        start_date, end_date
    )

    data = []

    if temporal_extent_1km:
        # Get 1km data
        outdir_col = outdir / "1km"
        collection = "CGLS_NDVI_V3_GLOBAL"

        data.append(
            _dwnld_cgls_ndvi_point(
                connection, collection, gdf, temporal_extent_1km, outdir_col, res="1km"
            )
        )

    if temporal_extent_300m:
        # Get 300m data
        collection = "CGLS_NDVI300_V2_GLOBAL"
        outdir_col = outdir / "300m"

        data.append(
            _dwnld_cgls_ndvi_point(
                connection,
                collection,
                gdf,
                temporal_extent_300m,
                outdir_col,
                res="300m",
            )
        )

    # Merge data from 1km and 300m
    df = pd.concat(data)
    df = df.sort_index()

    # Apply upperenvelop smoothing or linear interpolation
    if upperenvelop:
        df[df > 250] = 0
        df[df.isna()] = 0
        df = df.astype(np.uint8)
        df = df.apply(lambda x: upper_envelop(x))
    else:
        df = df.astype(float)
        # Mask invalid data (DN > 250)
        df[df > 250] = np.nan
        # Apply interpolation
        df = df.interpolate(method="time", limit_area="inside")

    # Apply scale factor and offset
    df = df * SCALE_FACTOR["NDVI"] + OFFSET["NDVI"]

    return df
