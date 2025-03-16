import rasterio
from rasterio.mask import mask
import geopandas as gpd
from pathlib import Path
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from typing import Optional


WPOP_BASE_URL = "gs://fao-gismgr-wpop-data/DATA/WPOP/MAPSET"


WPOP_SUPPORTED_VARIABLES = {
    "WPOP-T": {"unit": "people", "description": "Total population"},
    "WPOP-D": {"unit": "people/km²", "description": "Population density"},
}


def _check_wpop_variable(
    variable: str,
) -> None:

    if variable not in WPOP_SUPPORTED_VARIABLES.keys():
        raise ValueError(
            f"Variable {variable} is not supported. Supported variables are {list(WPOP_SUPPORTED_VARIABLES.keys())}"
        )


def _get_wpop_file_path(
    variable: str,
    year: int,
):

    filename = f"WPOP.{variable}.{year}.tif"

    return f"{WPOP_BASE_URL}/{variable}/{filename}"


def _get_lst_dates(
    start_date: str,
    end_date: Optional[str] = None,
) -> list[str]:

    # Parse the dates
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if end_date is None:
        end_date = start_date
    else:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Generate annual dates
    final_dates = [
        datetime(year, 1, 1).strftime("%Y-%m-%d")
        for year in range(start_date.year, end_date.year + 1)
        if start_date <= datetime(year, 1, 1) <= end_date
    ]

    return final_dates


def extract_wpop_data(
    variable: str,
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    start_date: str,
    end_date: Optional[str] = None,
    write_tif: bool = True,
    debug: bool = False,
) -> dict:

    # Check if variable is supported
    _check_wpop_variable(variable)

    # Get list of dates
    if end_date is not None and datetime.strptime(end_date, "%Y-%m-%d") < datetime(
        2000, 1, 1
    ):
        raise ValueError("WPOP data only available between 2000 and 2020")
    if end_date is None and datetime.strptime(start_date, "%Y-%m-%d") < datetime(
        2000, 1, 1
    ):
        raise ValueError("WPOP data only available between 2000 and 2020")
    if end_date is not None and datetime.strptime(start_date, "%Y-%m-%d") < datetime(
        2000, 1, 1
    ):
        print("Your timeseries will be cut-off at 2000-01-01")
        start_date = "2000-01-01"
    if datetime.strptime(start_date, "%Y-%m-%d") > datetime(2020, 1, 1):
        raise ValueError("WPOP data only available between 2000 and 2020")
    if end_date is not None and datetime.strptime(end_date, "%Y-%m-%d") > datetime(
        2020, 1, 1
    ):
        print("Your timeseries will be cut-off at 2020-01-01")
        end_date = "2020-01-01"

    dates = _get_lst_dates(start_date, end_date=end_date)

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
            year = datetime.strptime(date, "%Y-%m-%d").year
            file_path = _get_wpop_file_path(
                variable,
                year,
            )

            # Construct output filename
            filename = file_path.split("/")[-1]
            outfile = outdir / filename

            if not outfile.exists():

                # Read the file from google cloud storage
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
                "sum": np.nansum(image),
            }

        stats[geom_id] = geom_stats

    return stats


def plot_stats(
    stats: dict,
    name: str,
    metric: str,
    ids: Optional[list[str]] = None,
) -> None:
    """Plot area statistics for a given metric.

    Parameters
    ----------
    stats : dict
        result of compute_area_stats function
    name : str
        Name of variable
    metric : str
        metric to plot (mean, min, max, std, sum)
    ids : list[str], optional
        list of ids to plot, by default all will be plotted

    Returns
    -------
    None
    """

    if metric not in ["mean", "min", "max", "std", "sum"]:
        raise ValueError("Metric should be one of mean, min, max, std, sum")

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

    return None
