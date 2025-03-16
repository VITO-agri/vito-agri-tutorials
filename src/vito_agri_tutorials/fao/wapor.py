import rasterio
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from typing import Optional


WAPOR3_BASE_URL = "gs://fao-gismgr-wapor-3-data/DATA/WAPOR-3/MAPSET"


WAPOR3_SUPPORTED_VARIABLES = {
    "L1-AETI-A": {
        "long_name": "Actual EvapoTranspiration and Interception",
        "units": "mm/year",
    },
    "L1-AETI-D": {
        "long_name": "Actual EvapoTranspiration and Interception",
        "units": "mm/day",
    },
    "L1-AETI-M": {
        "long_name": "Actual EvapoTranspiration and Interception",
        "units": "mm/month",
    },
    "L1-E-A": {"long_name": "Evaporation", "units": "mm/year"},
    "L1-E-D": {"long_name": "Evaporation", "units": "mm/day"},
    "L1-GBWP-A": {"long_name": "Gross Biomass Water Productivity", "units": "kg/m³"},
    "L1-I-A": {"long_name": "Interception", "units": "mm/year"},
    "L1-I-D": {"long_name": "Interception", "units": "mm/day"},
    "L1-NBWP-A": {"long_name": "Net Biomass Water Productivity", "units": "kg/m³"},
    "L1-NPP-D": {"long_name": "Net Primary Production", "units": "gC/m²/day"},
    "L1-NPP-M": {"long_name": "Net Primary Production", "units": "gC/m²/month"},
    "L1-PCP-A": {"long_name": "Precipitation", "units": "mm/year"},
    "L1-PCP-D": {"long_name": "Precipitation", "units": "mm/day"},
    "L1-PCP-E": {"long_name": "Precipitation", "units": "mm/day"},
    "L1-PCP-M": {"long_name": "Precipitation", "units": "mm/month"},
    "L1-QUAL-LST-D": {"long_name": "Quality Land Surface Temperature", "units": "d"},
    "L1-QUAL-NDVI-D": {
        "long_name": "Quality of Normalized Difference Vegetation Index",
        "units": "d",
    },
    "L1-RET-A": {"long_name": "Reference Evapotranspiration", "units": "mm/year"},
    "L1-RET-D": {"long_name": "Reference Evapotranspiration", "units": "mm/day"},
    "L1-RET-E": {"long_name": "Reference Evapotranspiration", "units": "mm/day"},
    "L1-RET-M": {"long_name": "Reference Evapotranspiration", "units": "mm/month"},
    "L1-RSM-D": {"long_name": "Relative Soil Moisture", "units": "%"},
    "L1-T-A": {"long_name": "Transpiration", "units": "mm/year"},
    "L1-T-D": {"long_name": "Transpiration", "units": "mm/day"},
    "L1-TBP-A": {"long_name": "Total Biomass Production", "units": "kg/ha"},
    "L2-AETI-A": {
        "long_name": "Actual EvapoTranspiration and Interception",
        "units": "mm/year",
    },
    "L2-AETI-D": {
        "long_name": "Actual EvapoTranspiration and Interception",
        "units": "mm/day",
    },
    "L2-AETI-M": {
        "long_name": "Actual EvapoTranspiration and Interception",
        "units": "mm/month",
    },
    "L2-E-A": {"long_name": "Evaporation", "units": "mm/year"},
    "L2-E-D": {"long_name": "Evaporation", "units": "mm/day"},
    "L2-GBWP-A": {"long_name": "Gross Biomass Water Productivity", "units": "kg/m³"},
    "L2-I-A": {"long_name": "Interception", "units": "mm/year"},
    "L2-I-D": {"long_name": "Interception", "units": "mm/day"},
    "L2-NBWP-A": {"long_name": "Net Biomass Water Productivity", "units": "kg/m³"},
    "L2-NPP-D": {"long_name": "Net Primary Production", "units": "gC/m²/day"},
    "L2-NPP-M": {"long_name": "Net Primary Production", "units": "gC/m²/month"},
    "L2-QUAL-NDVI-D": {
        "long_name": "Quality of Normalized Difference Vegetation Index",
        "units": "d",
    },
    "L2-RSM-D": {"long_name": "Relative Soil Moisture", "units": "%"},
    "L2-T-A": {"long_name": "Transpiration", "units": "mm/year"},
    "L2-T-D": {"long_name": "Transpiration", "units": "mm/day"},
    "L2-TBP-A": {"long_name": "Total Biomass Production", "units": "kg/ha"},
}


def _check_wapor_variable(
    variable: str,
) -> None:

    if variable not in WAPOR3_SUPPORTED_VARIABLES.keys():
        raise ValueError(
            f"Variable {variable} is not supported. Supported variables are {list(WAPOR3_SUPPORTED_VARIABLES.keys())}"
        )


def _split_date(date: str):

    year = str(date.split("-")[0])
    month = str(date.split("-")[1])
    day = str(date.split("-")[2])
    if int(day) >= 1 and int(day) <= 10:
        dekad = "D1"
    elif int(day) >= 11 and int(day) <= 20:
        dekad = "D2"
    else:
        dekad = "D3"

    return year, month, day, dekad


def _get_wapor_file_path(
    variable: str,
    date: str,
):

    # Split date
    year, month, day, dekad = _split_date(date)

    # Daily track
    if variable.split("-")[-1] == "E":
        filename = f"WAPOR-3.{variable}.{year}-{month}-{day}.tif"

    # Dekadal track
    elif variable.split("-")[-1] == "D":

        filename = f"WAPOR-3.{variable}.{year}-{month}-{dekad}.tif"

    # Monthly track
    elif variable.split("-")[-1] == "M":

        filename = f"WAPOR-3.{variable}.{year}-{month}.tif"

    # Annual track
    elif variable.split("-")[-1] == "A":

        filename = f"WAPOR-3.{variable}.{year}.tif"

    else:
        raise ValueError("Something went wrong")

    return f"{WAPOR3_BASE_URL}/{variable}/{filename}"


def _get_lst_dates(
    start_date: str,
    timestep: str,
    end_date: Optional[str] = None,
) -> list[str]:

    # Parse the dates
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if end_date is None:
        end_date = start_date
    else:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if timestep == "E":
        final_dates = pd.date_range(start_date, end_date).strftime("%Y-%m-%d").tolist()

    elif timestep == "A":
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


def extract_wapor_data(
    variable: str,
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    start_date: str,
    end_date: Optional[str] = None,
    write_tif: bool = True,
    debug: bool = False,
) -> dict:

    # Check if variable is supported
    _check_wapor_variable(variable)

    # Get list of dates
    if end_date is not None and datetime.strptime(end_date, "%Y-%m-%d") < datetime(
        2018, 1, 1
    ):
        raise ValueError("WaPOR V3 data only available from 2018-01-01")
    if end_date is None and datetime.strptime(start_date, "%Y-%m-%d") < datetime(
        2018, 1, 1
    ):
        raise ValueError("WaPOR V3 data only available from 2018-01-01")
    if end_date is not None and datetime.strptime(start_date, "%Y-%m-%d") < datetime(
        2018, 1, 1
    ):
        print("Your timeseries will be cut-off at 2018-01-01")
        start_date = "2018-01-01"

    timestep = variable.split("-")[-1]
    dates = _get_lst_dates(start_date, timestep, end_date=end_date)

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
            file_path = _get_wapor_file_path(
                variable,
                date=date,
            )

            # Construct output filename
            filename = file_path.split("/")[-1]
            outfile = outdir / filename

            if not outfile.exists():

                # Read the file from the cloud
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
                        scale = src.scales[0]
                        offset = src.offsets[0]

                        arr = out_image[0]
                        arr = arr.astype(np.float32)
                        arr[arr == src.meta["nodata"]] = np.nan
                        arr = (arr * scale) + offset

                        geom_results[date] = arr

                        arr[arr == np.nan] = src.meta["nodata"]

                        if write_tif:
                            # Prepare metadata
                            out_meta = src.profile.copy()

                            # Update metadata
                            out_meta.update(
                                {
                                    "driver": "GTiff",
                                    "height": out_image.shape[1],
                                    "width": out_image.shape[2],
                                    "transform": out_transform,
                                    "count": 1,
                                    "dtype": rasterio.float32,
                                    "scales": [1.0],
                                    "offsets": [0.0],
                                }
                            )

                            # Write the file
                            with rasterio.open(outfile, "w", **out_meta) as dest:
                                dest.write(arr, 1)
                except Exception as e:
                    print(f"Error reading file {file_path}")
                    print(e)
            else:
                # Read previously downloaded file
                if debug:
                    print(f"File {outfile} already exists")
                with rasterio.open(outfile) as src:
                    image = src.read(1)
                    nodata = src.profile["nodata"]
                    image[image == nodata] = np.nan
                    geom_results[date] = image

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

            # Compute area statistics
            geom_stats[date] = {
                "mean": np.nanmean(image),
                "min": np.nanmin(image),
                "max": np.nanmax(image),
                "std": np.nanstd(image),
            }

        stats[geom_id] = geom_stats

    return stats


def plot_stats(
    stats: dict,
    name: str,
    metric: str,
    ids: Optional[list[str]] = None,
    plot: bool = True,
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
    plot : bool, optional
        whether to plot the results, by default True

    Returns
    -------
    pd.DataFrame
        pandas dataframe with statistics
    """

    if metric not in ["mean", "min", "max", "std"]:
        raise ValueError("Metric should be one of mean, min, max, std")

    if ids is not None:
        stats = {k: v for k, v in stats.items() if k in ids}

    if plot:

        fig, ax = plt.subplots()

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
    stats_df.index = list(stats[next(iter(stats))].keys())
    stats_df.columns = list(stats.keys())

    return stats_df
