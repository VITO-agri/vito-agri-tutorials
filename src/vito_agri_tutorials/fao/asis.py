from typing import Optional
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

ASIS_BASE_URL = "gs://fao-gismgr-asis-data/DATA/ASIS/MAPSET"

ASIS_SUPPORTED_VARIABLES = {
    "VCI-D": "Vegetation Condition Index - Dekadal",
    "VCI-M": "Vegetation Condition Index - Monthly",
    "VHI-D": "Vegetation Health Index - Dekadal",
    "VHI-M": "Vegetation Health Index - Monthly ",
    "MVHI-D": "Mean Vegetation Health Index - Dekadal",
    "MVHI-A": "Mean Vegetation Health Index - Seasonal",
    "ASI-D": "Agricultural Stress Index - Dekadal",
    "ASI-A": "Agricultural Stress Index - Seasonal",
    "DI-D": "Drought Intensity - Dekadal",
    "DI-A": "Drought Intensity - Seasonal",
    "PHE": "Phenology - Fixed",
    "POS": "Progress of season",
}

ASIS_EXAMPLE_FILES = {
    "VCI-D": "ASIS.VCI-D.1984-01-D1.tif",
    "VCI-M": "ASIS.VCI-M.1984-01.tif",
    "VHI-D": "ASIS.VHI-D.1984-01-D1.tif",
    "VHI-M": "ASIS.VHI-M.1984-01.tif ",
    "MVHI-D": "ASIS.MVHI-D.1984-01-D1.GS1.LC-C.tif",
    "MVHI-A": "ASIS.MVHI-A.1984.GS1.LC-C.tif",
    "ASI-D": "ASIS.ASI-D.1984-01-D1.GS1.LC-C.tif",
    "ASI-A": "ASIS.ASI-A.1984.GS1.LC-C.tif",
    "DI-D": "ASIS.DI-D.2010-01-D1.GS1.LC-C.tif",
    "DI-A": "ASIS.DI-A.1984.GS1.LC-C.tif",
    "PHE": "ASIS.PHE.GS1.EOS.LC-C.tif",
    "POS": "ASIS.POS.LC-C.GS1.D01.tif",
}

PHE_SUB_VARIABLES = ["SOS", "MOS", "EOS"]


def _check_asis_variable(
    variable: str,
) -> None:

    if variable not in ASIS_SUPPORTED_VARIABLES.keys():
        raise ValueError(
            f"Variable {variable} is not supported. Supported variables are {ASIS_SUPPORTED_VARIABLES.keys()}"
        )


def _check_asis_variable_args(
    variable: str,
    date: Optional[str] = None,
    season: Optional[int] = None,
    landcover: Optional[str] = None,
    sub_variable: Optional[str] = None,
) -> None:

    # check if all required arguments are provided for specific variable
    if variable in ["PHE", "POS"]:
        if variable == "PHE" and sub_variable not in PHE_SUB_VARIABLES:
            raise ValueError(
                f"Sub-variable for variable {variable} should be one of {PHE_SUB_VARIABLES}"
            )
        if landcover is None:
            raise ValueError(f"Landcover is required for variable {variable}")
        if season is None:
            raise ValueError(f"Season is required for variable {variable}")
        if variable == "POS" and date is None:
            raise ValueError(f"Date is required for variable {variable}")
    else:
        if variable.split("-")[1] in ["D", "M", "A"] and date is None:
            raise ValueError(f"Date is required for variable {variable}")
        if variable.split("-")[1] in ["A"] and season is None:
            raise ValueError(f"Season is required for variable {variable}")
        if variable.split("-")[1] in ["A"] and landcover is None:
            raise ValueError(f"Landcover is required for variable {variable}")
        if variable.split("-")[0] in ["ASI", "DI", "MVHI"] and season is None:
            raise ValueError(f"Season is required for variable {variable}")
        if variable.split("-")[0] in ["ASI", "DI", "MVHI"] and landcover is None:
            raise ValueError(f"Landcover is required for variable {variable}")


def _split_date(date: str):

    year = str(date.split("-")[0])
    month = str(date.split("-")[1])
    day = str(date.split("-")[2])
    if int(day) == 1:
        dekad = "D1"
    elif int(day) == 11:
        dekad = "D2"
    elif int(day) == 21:
        dekad = "D3"

    return year, month, dekad


def _get_asis_file_path(
    variable: str,
    sub_variable: Optional[str] = None,
    date: Optional[str] = None,
    season: Optional[int] = None,
    landcover: Optional[str] = None,
):

    # Split date
    if date is not None:
        year, month, dekad = _split_date(date)

    # Phenology
    if variable == "PHE":

        filename = f"ASIS.{variable}.GS{str(season)}.{sub_variable}.LC-{landcover}.tif"

    # Progress of season
    elif variable == "POS":

        filename = f"ASIS.{variable}.LC-{landcover}.GS{str(season)}.{dekad}.tif"

    # Dekadal track
    elif variable.split("-")[1] == "D":

        if variable.split("-")[0] in ["ASI", "DI", "MVHI"]:
            filename = f"ASIS.{variable}.{year}-{month}-{dekad}.GS{str(season)}.LC-{landcover}.tif"

        elif variable.split("-")[0] in ["VCI", "VHI"]:
            filename = f"ASIS.{variable}.{year}-{month}-{dekad}.tif"

    # Monthly track
    elif variable.split("-")[1] == "M":

        filename = f"ASIS.{variable}.{year}-{month}.tif"

    # Annual track
    elif variable.split("-")[1] == "A":

        filename = f"ASIS.{variable}.{year}.GS{str(season)}.LC-{landcover}.tif"

    else:
        raise ValueError("Something went wrong")

    return f"{ASIS_BASE_URL}/{variable}/{filename}"


def _get_lst_dates(
    start_date: str, timestep: str, end_date: Optional[str] = None
) -> list[str]:

    # Parse the dates
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if end_date is None:
        end_date = start_date
    else:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if timestep == "A":
        # Generate annual dates
        final_dates = [
            datetime(year, 1, 1).strftime("%Y-%m-%d")
            for year in range(start_date.year, end_date.year + 1)
            if start_date <= datetime(year, 1, 1) <= end_date
        ]
    else:
        final_dates = []
        # Define the days to include
        days_to_include = {"D": [1, 11, 21], "M": [1]}.get(timestep, [])
        current_date = start_date.replace(day=1)

        # Generate dates for each month
        while current_date <= end_date:
            for day in days_to_include:
                try:
                    special_date = current_date.replace(day=day)
                    if start_date <= special_date <= end_date:
                        final_dates.append(special_date.strftime("%Y-%m-%d"))
                except ValueError:
                    continue  # Ignore invalid dates (e.g., February 30)

            # Move to the next month
            current_date = (current_date.replace(day=28) + timedelta(days=4)).replace(
                day=1
            )

    return final_dates


def extract_asis_data(
    variable: str,
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    season: Optional[int] = None,
    landcover: Optional[str] = None,
    sub_variable: Optional[str] = None,
    write_tif: bool = True,
    debug: bool = False,
) -> dict:

    # Check if variable is supported
    _check_asis_variable(variable)

    # Run check to see if we have all required parameters for variable
    _check_asis_variable_args(
        variable,
        date=start_date,
        season=season,
        landcover=landcover,
        sub_variable=sub_variable,
    )

    # Get list of dates
    if variable not in ["PHE", "POS"]:
        timestep = variable.split("-")[1]
        dates = _get_lst_dates(start_date, timestep, end_date=end_date)
    else:
        dates = [None]

    results = {}

    # Loop over geometries
    for idx, row in gdf.iterrows():

        # prepare output
        geom_results = {}

        # Create output directory
        outdir = output_path / row["id"]
        outdir.mkdir(exist_ok=True, parents=True)

        # Loop over dates
        for date in dates:

            # Get the file path
            file_path = _get_asis_file_path(
                variable,
                sub_variable=sub_variable,
                date=date,
                season=season,
                landcover=landcover,
            )

            # Construct output filename
            filename = file_path.split("/")[-1]
            outfile = outdir / filename

            # Get correct timestamp for inserting result in dict
            if date is None:
                timestamp = "2000-01-01"
            else:
                timestamp = date

            if not outfile.exists():
                # Read the file from google storage
                if debug:
                    print(f"Reading file {file_path}")

                try:
                    with rasterio.open(file_path) as src:

                        # Mask the data
                        out_image, out_transform = mask(
                            src,
                            [row.geometry],
                            crop=True,
                        )

                        geom_results[timestamp] = out_image[0]

                        if write_tif:
                            # Prepare metadata
                            out_meta = src.meta.copy()

                            # Update metadata
                            out_meta.update(
                                {
                                    "driver": "GTiff",
                                    "height": out_image.shape[1],
                                    "width": out_image.shape[2],
                                    "transform": out_transform,
                                    "count": 1,
                                }
                            )

                            # Write the file
                            with rasterio.open(outfile, "w", **out_meta) as dest:
                                dest.write(out_image[0], 1)
                except Exception as e:
                    print(f"Error reading file {file_path}")
                    print(e)

            else:
                # Read previously downloaded file
                if debug:
                    print(f"File {outfile} already exists")
                with rasterio.open(outfile) as src:
                    geom_results[timestamp] = src.read(1)

        results[row["id"]] = geom_results

    return results


def compute_area_stats(
    data: dict,
) -> dict:

    stats = {}

    for geom_id, geom_data in data.items():

        # Prepare output
        geom_stats = {}

        # Loop over dates
        for date, image in geom_data.items():

            # Compute area
            masked_im = np.where(image > 250, np.nan, image)
            geom_stats[date] = {
                "mean": np.nanmean(masked_im),
                "min": np.nanmin(masked_im),
                "max": np.nanmax(masked_im),
                "std": np.nanstd(masked_im),
            }

        stats[geom_id] = geom_stats

    return stats


def plot_stats(
    stats: dict,
    name: str,
    metric: str,
    ids: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Plot area statistics for a given metric.

    Parameters
    ----------
    stats : dict
        result of compute_area_stats function
    name : str
        Name of variable
    metric : str
        metric to plot (mean, min, max, std)
    ids : list[str], optional
        list of ids to plot, by default all will be plotted

    Returns
    -------
    pd.DataFrame
        pandas dataframe with statistics
    """

    if metric not in ["mean", "min", "max", "std"]:
        raise ValueError("Metric should be one of mean, min, max, std")

    fig, ax = plt.subplots()

    if ids is not None:
        stats = {k: v for k, v in stats.items() if k in ids}

    for geom_id, geom_stats in stats.items():

        dates = list(geom_stats.keys())
        dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
        results = [v[metric] for v in geom_stats.values()]
        ax.plot(dates, results, marker="o", linestyle="-", label=geom_id)

    plt.xlabel("Date")
    plt.ylabel(f"{name} {metric}")
    plt.xticks(rotation=90)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    fig.subplots_adjust(right=0.75)  # Ensures enough space for the legend
    # add gridlines
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Export plotted stats to pandas dataframe
    stats_df = pd.DataFrame(
        {k: [v[metric] for v in geom_stats.values()] for k, geom_stats in stats.items()}
    )
    stats_df.index = list(geom_stats.keys())
    stats_df.columns = list(stats.keys())

    return stats_df
