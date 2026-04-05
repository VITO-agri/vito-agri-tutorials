import pandas as pd
import numpy as np
import csv
import os
from pathlib import Path
import geopandas as gpd
import datetime
from dateutil.relativedelta import relativedelta
from typing import Literal, List
import glob
import xarray as xr
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.enums import Resampling as ResamplingEnum
from tqdm import tqdm
from scipy import stats
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

from vito_agri_tutorials.utils.geotiff import (
    write_geotiff,
    read_geotiff,
)
from vito_agri_tutorials.utils.upperenvelop import upper_envelop


# Path to the resources folder
current_dir = os.path.dirname(__file__)
resources_folder = os.path.join(current_dir, "resources", "vici")

NYEARS_ARCHIVE = 20

TIME_BUFFER_MONTHS = 2

NDVI_SCALE = 0.004
NDVI_OFFSET = -0.08

PERCENTILE_THRESHOLDS = [5, 15]

QUALITY_FLAGS = {
    252: "Smoothed data",
    253: "Out of season",
    254: "Invalid (trend/no data)",
    255: "Other invalid",
}

############################################################
# SPATIAL EXTENT FUNCTIONS
############################################################


def get_spatial_extent(
    aoi_gpkg: Path,
):
    """
    Compute the extents of the AOI for 1km and 300m resolution,
    ensuring data downloaded at both resolutions match perfectly.

    Parameters
    ----------
    aoi_gpkg : Path
        Path to GeoPackage file containing a single geometry defining the AOI boundaries

    Returns
    -------
    Path
        Path to the output AOI file (csv)
    """
    # Set output file and check for its existence
    outdir = aoi_gpkg.parent
    outfile_csv = outdir / "AOI_adjusted.csv"

    if not outfile_csv.exists():

        # Get coordinates of pixel centroids
        COP_X1km = pd.read_csv(f"{resources_folder}/Centroids/lon1km_cop.txt")
        COP_Y1km = pd.read_csv(f"{resources_folder}/Centroids/lat1km_cop.txt")
        COP_X300m = pd.read_csv(f"{resources_folder}/Centroids/lon300m_cop.txt")
        COP_Y300m = pd.read_csv(f"{resources_folder}/Centroids/lat300m_cop.txt")

        # Load AOI
        aoi_gdf = gpd.read_file(aoi_gpkg)

        # Get bounds
        bounds = aoi_gdf.total_bounds
        assert bounds[0] <= bounds[2], (
            "min Longitude is bigger than correspond Max, "
            "pls change position or check values."
        )
        assert bounds[3] >= bounds[1], (
            "min Latitude is bigger than correspond Max, "
            "pls change position or check values."
        )

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx][0]

        # The following provides the NEW 1km centroids of the AOI-Extent
        AOI_adj_1k = np.zeros(4)
        AOI_adj_1k[0] = find_nearest(COP_X1km, bounds[0])
        AOI_adj_1k[1] = find_nearest(COP_X1km, bounds[2])
        AOI_adj_1k[2] = find_nearest(COP_Y1km, bounds[3])
        AOI_adj_1k[3] = find_nearest(COP_Y1km, bounds[1])

        # The following provides the NEW 300m centroids of the AOI-Extent
        AOI_adj_300m = np.zeros(4)
        AOI_adj_300m[0] = find_nearest(COP_X300m, bounds[0])
        AOI_adj_300m[1] = find_nearest(COP_X300m, bounds[2])
        AOI_adj_300m[2] = find_nearest(COP_Y300m, bounds[3])
        AOI_adj_300m[3] = find_nearest(COP_Y300m, bounds[1])

        # The following provides the NEW 300m centroids of the AOI-Extent - EWNS
        AOI_adj_1k300m = np.zeros(4)
        AOI_adj_1k300m[0] = find_nearest(COP_X300m, AOI_adj_1k[0] - 1 / 336)
        AOI_adj_1k300m[1] = find_nearest(COP_X300m, AOI_adj_1k[1] + 1 / 336)
        AOI_adj_1k300m[2] = find_nearest(COP_Y300m, AOI_adj_1k[2] + 1 / 336)
        AOI_adj_1k300m[3] = find_nearest(COP_Y300m, AOI_adj_1k[3] - 1 / 336)

        AOI_Header = [
            "type",
            "west",
            "east",
            "north",
            "south",
            "StartCol",
            "StartRow",
            "Columns",
            "Rows",
        ]

        AOI_data = [
            ["SHP", bounds[0], bounds[2], bounds[3], bounds[1], 0, 0, 0, 0],
            [
                "COP_1k",
                AOI_adj_1k[0],
                AOI_adj_1k[1],
                AOI_adj_1k[2],
                AOI_adj_1k[3],
                round(((180 + AOI_adj_1k[0]) * 112)),
                round(((80 - AOI_adj_1k[2]) * 112)),
                round(((AOI_adj_1k[1] - AOI_adj_1k[0]) * 112) + 1),
                round(((AOI_adj_1k[2] - AOI_adj_1k[3]) * 112) + 1),
            ],
            [
                "COP_300m",
                AOI_adj_300m[0],
                AOI_adj_300m[1],
                AOI_adj_300m[2],
                AOI_adj_300m[3],
                round(((180 + AOI_adj_300m[0]) * 336)),
                round(((80 - AOI_adj_300m[2]) * 336)),
                round(((AOI_adj_300m[1] - AOI_adj_300m[0]) * 336) + 1),
                round(((AOI_adj_300m[2] - AOI_adj_300m[3]) * 336) + 1),
            ],
            [
                "COP_1k300m",
                AOI_adj_1k300m[0],
                AOI_adj_1k300m[1],
                AOI_adj_1k300m[2],
                AOI_adj_1k300m[3],
                round(((180 + AOI_adj_1k300m[0]) * 336)),
                round(((80 - AOI_adj_1k300m[2]) * 336)),
                round(((AOI_adj_1k300m[1] - AOI_adj_1k300m[0]) * 336) + 1),
                round(((AOI_adj_1k300m[2] - AOI_adj_1k300m[3]) * 336) + 1),
            ],
        ]

        # Write result to csv file
        outdir = aoi_gpkg.parent
        outfile_csv = outdir / "AOI_adjusted.csv"
        with open(outfile_csv, "w") as file:
            writer = csv.writer(file)
            writer.writerow(AOI_Header)
            writer.writerows(AOI_data)

    else:
        logger.info("Spatial extent already computed, skipping!")

    return outfile_csv


############################################################
# TEMPORAL EXTENT FUNCTIONS
############################################################
def get_vici_archive_dates(end_year: int):
    """Get the VICI archive dates for a given end year.

    Parameters
    ----------
    end_year : int
        The final year of the 20-year reference period.

    Returns
    -------
    start_date_archive : str
        The start date of the archive period.
    end_date_archive : str
        The end date of the archive period.
    """
    end_date_archive = f"{end_year}-12-31"

    # get start date of archive
    e_date = pd.to_datetime(end_date_archive)
    s_date = e_date - pd.DateOffset(years=NYEARS_ARCHIVE) + pd.DateOffset(days=1)
    start_date_archive = s_date.strftime("%Y-%m-%d")
    logger.info(f"Archive period: {start_date_archive} - {end_date_archive}")

    return start_date_archive, end_date_archive


def get_temporal_extent(start_date: str, end_date: str):
    """
    Determine the temporal extent (for upper envelope filtering).

    Parameters
    ----------
    start_date : str
        Start date of temporal extent, format 'YYYY-MM-DD'
    end_date : str
        End date of temporal extent, format 'YYYY-MM-DD'

    Returns
    -------
    temporal_extent_1km : list of dates
        Start and end date of extended temporal range for 1km period
    temporal_extent_300km : list of dates
        Start and end date of extended temporal range for 300m period
    """
    # Define extended temporal range for upper envelope filtering
    start_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    if start_datetime - relativedelta(months=TIME_BUFFER_MONTHS) < datetime.datetime(
        1999, 1, 1
    ):
        start_date_extended = datetime.datetime(1999, 1, 1)
    else:
        start_date_extended = start_datetime - relativedelta(months=TIME_BUFFER_MONTHS)

    if (
        end_datetime + relativedelta(months=TIME_BUFFER_MONTHS)
        > datetime.datetime.today()
    ):
        end_date_extended = datetime.datetime.today()
    else:
        end_date_extended = end_datetime + relativedelta(months=TIME_BUFFER_MONTHS)

    # Check if temporal extent spans 1km period, 300m period, or both
    start_300m = datetime.datetime(2020, 7, 1)
    if (start_date_extended < start_300m) and (end_date_extended < start_300m):
        temporal_extent_1km = [
            start_date_extended.strftime("%Y-%m-%d"),
            end_date_extended.strftime("%Y-%m-%d"),
        ]
        temporal_extent_300m = None
    elif (start_date_extended < start_300m) and (end_date_extended >= start_300m):
        temporal_extent_1km = [
            start_date_extended.strftime("%Y-%m-%d"),
            datetime.datetime(2020, 6, 30).strftime("%Y-%m-%d"),
        ]
        temporal_extent_300m = [
            start_300m.strftime("%Y-%m-%d"),
            end_date_extended.strftime("%Y-%m-%d"),
        ]
    elif (start_date_extended >= start_300m) and (end_date_extended >= start_300m):
        temporal_extent_1km = None
        temporal_extent_300m = [
            start_date_extended.strftime("%Y-%m-%d"),
            end_date_extended.strftime("%Y-%m-%d"),
        ]
    else:
        raise ValueError("End date before start date, cannot continue!")

    logger.info(f"Temporal extent 1km: {temporal_extent_1km}")
    logger.info(f"Temporal extent 300m: {temporal_extent_300m}")

    return temporal_extent_1km, temporal_extent_300m


############################################################
# HELPER FUNCTIONS
############################################################
def calculateGeotransformXY(x, y):
    """
    Compute the geotransform for the given x and y coordinates.

    Parameters
    ----------
    x : array
        x coordinates
    y : array
        y coordinates

    Returns
    -------
    geotransform : tuple
        Geotransform for output Geotiff based on x and y
    """
    xmin = x.values.min()
    xmax = x.values.max()
    ymin = y.values.min()
    ymax = y.values.max()
    xres = (xmax - xmin) / np.float64(len(x) - 1)
    yres = (ymax - ymin) / np.float64(len(y) - 1)

    # Reference is the topleft corner of the first pixel
    geotransform = (xmin - 1 / 2 * xres, xres, 0.0, ymax + 1 / 2 * yres, 0.0, -yres)
    return geotransform


def resample_raster_rasterio(
    input_file: Path,
    output_file: Path,
    new_resolution: float,
    resampling_method: Literal[
        "average", "bilinear", "nearest", "cubic", "mode", "max", "min"
    ],
):
    """
    Resample a raster to a new resolution using rasterio.

    Parameters
    ----------
    input_file : Path
        Path to input raster file
    output_file : Path
        Path to output resampled raster file
    new_resolution : float
        New pixel resolution (assuming square pixels)
    resampling_method : str
        Resampling method ('average', 'bilinear', 'nearest', etc.)

    Returns
    -------
    None
    """
    # Map resampling methods
    resampling_map = {
        "average": ResamplingEnum.average,
        "bilinear": ResamplingEnum.bilinear,
        "nearest": ResamplingEnum.nearest,
        "cubic": ResamplingEnum.cubic,
        "mode": ResamplingEnum.mode,
        "max": ResamplingEnum.max,
        "min": ResamplingEnum.min,
    }

    resampling_alg = resampling_map.get(resampling_method, ResamplingEnum.average)

    with rasterio.open(input_file) as src:
        # Calculate new dimensions based on resolution
        old_res = src.transform.a  # assuming square pixels
        scale_factor = old_res / new_resolution

        new_width = int(src.width * scale_factor)
        new_height = int(src.height * scale_factor)

        # Calculate new transform
        new_transform = Affine(
            new_resolution, 0.0, src.transform.c, 0.0, -new_resolution, src.transform.f
        )

        # Create output profile
        out_profile = src.profile.copy()
        out_profile.update(
            {"width": new_width, "height": new_height, "transform": new_transform}
        )

        # Read and resample data
        data = src.read()

        # Create output array
        out_data = np.zeros((src.count, new_height, new_width), dtype=data.dtype)

        # Resample each band
        for i in range(src.count):
            reproject(
                data[i],
                out_data[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=src.crs,
                resampling=resampling_alg,
            )

        # Write output
        with rasterio.open(output_file, "w", **out_profile) as dst:
            dst.write(out_data)


def read_raster_metadata_rasterio(raster_file):
    """
    Read raster metadata using rasterio.

    Parameters
    ----------
    raster_file : str
        Path to raster file

    Returns
    -------
    dict
        Dictionary containing data array, projection (CRS), and geotransform
    """
    with rasterio.open(raster_file) as src:
        data = src.read()
        if data.shape[0] == 1:  # Single band
            data = data[0]

        # Convert rasterio transform to GDAL-style geotransform for compatibility
        transform = src.transform
        geotransform = (
            transform.c,  # xmin
            transform.a,  # xres
            transform.b,  # 0
            transform.f,  # ymax
            transform.d,  # 0
            transform.e,  # -yres
        )

        return {
            "data": data,
            "projection": src.crs.to_wkt(),
            "geotransform": geotransform,
            "crs": src.crs,
            "transform": src.transform,
        }


############################################################
# DOWNLOAD FUNCTIONS
############################################################
def get_data_local_folder(
    AOI_adjusted_file: Path,
    res: Literal["1km", "300m"],
    temporal_extent: List[str],
    outdir: Path,
    overwrite: bool = False,
):
    """
    Copy NDVI data from local directory for given temporal range and
    crop to spatial extents given in the AOI_adjusted_file.

    Parameters
    ----------
    AOI_adjusted_file : Path
        CSV file with AOI boundaries for 1 km and 300 m
    res : Literal['1km', '300m']
        Resolution (1km or 300m)
    temporal_extent : list[str]
        Start and end date of period that is downloaded
    outdir : Path
        Output directory for retrieved NDVI data
    overwrite : bool
        Whether to overwrite existing files

    Returns
    -------
    None
    """

    # Get AOI boundaries
    AOI_Adjusted = pd.read_csv(AOI_adjusted_file)

    # Select local folder depending on resolution (1km or 300m)
    if res == "1km":
        NDVI_src_dir = (
            "/data/MTDA/Copernicus/Land/global/netcdf/ndvi/ndvi_1km_v3_10daily"
        )
        rows_cols = AOI_Adjusted[AOI_Adjusted["type"] == "COP_1k"][
            ["StartCol", "StartRow", "Columns", "Rows"]
        ]

    elif res == "300m":
        NDVI_src_dir = (
            "/data/MTDA/Copernicus/Land/global/netcdf/ndvi/ndvi_300m_v2_10daily"
        )
        rows_cols = AOI_Adjusted[AOI_Adjusted["type"] == "COP_1k300m"][
            ["StartCol", "StartRow", "Columns", "Rows"]
        ]
        new_res = 3.0 * 0.00297619047619

    # Get spatial extent (which rows and columns)
    xmin = rows_cols["StartCol"].values[0]
    ymin = rows_cols["StartRow"].values[0]
    xmax = rows_cols["StartCol"].values[0] + rows_cols["Columns"].values[0]
    ymax = rows_cols["StartRow"].values[0] + rows_cols["Rows"].values[0]

    # Determine all dates within temporal extent
    all_dates = pd.date_range(temporal_extent[0], temporal_extent[1], freq="D")
    dates = []
    for d in all_dates:
        if d.day in [1, 11, 21]:
            dates.append(d.strftime("%Y-%m-%d"))

    # Crop data to spatial extent and save in output directory
    for i in range(len(dates)):
        date = dates[i]
        year = date[:4]

        logger.debug(f"Processing {date}...")

        outfile = outdir / f"{date}.tif"
        if not outfile.exists() or overwrite:
            filename = glob.glob(f"{NDVI_src_dir}/{year}/{date.replace('-', '')}/*.nc")
            if filename:
                f = xr.open_dataset(
                    filename[0],
                    mask_and_scale=False,
                    decode_times=False,
                    engine="netcdf4",
                )

                if res == "1km":
                    ndvi = f["NDVI"]
                    ndvi_data = ndvi[0, ymin:ymax:1, xmin:xmax:1]

                    # Mask invalid pixels (DN > 250)
                    ndvi_data.values[ndvi_data.values > 250] = 255

                    output_file = outdir / f"{date}.tif"
                    geotransform = calculateGeotransformXY(
                        x=ndvi_data.lon, y=ndvi_data.lat
                    )
                    write_geotiff(
                        ndvi_data.values,
                        output_file,
                        projection=f.crs.attrs["spatial_ref"],
                        geotransform=geotransform,
                        datatype="uint8",
                        nodata=255,
                    )

                elif res == "300m":
                    # NDVI
                    ndvi = f["NDVI"]
                    ndvi_data = ndvi[0, ymin:ymax:1, xmin:xmax:1]
                    ndvi_data.values[ndvi_data.values > 250] = 255
                    output_file_ndvi = outdir / f"{date}_ndvi.tif"
                    output_file_ndvi_resampled = Path(
                        f"{output_file_ndvi.stem}_resampled.tif"
                    )
                    geotransform = calculateGeotransformXY(
                        x=ndvi_data.lon, y=ndvi_data.lat
                    )
                    write_geotiff(
                        ndvi_data.values,
                        output_file_ndvi,
                        projection=f.crs.attrs["spatial_ref"],
                        geotransform=geotransform,
                        datatype="uint8",
                        nodata=255,
                    )

                    # Resample NDVI to 1 km using rasterio
                    resample_raster_rasterio(
                        output_file_ndvi, output_file_ndvi_resampled, new_res, "average"
                    )

                    # NOBS
                    nobs = f["NOBS"]
                    nobs_data = nobs[0, ymin:ymax:1, xmin:xmax:1]
                    output_file_nobs = outdir / f"{date}_nobs.tif"
                    output_file_nobs_resampled = Path(
                        f"{output_file_nobs.stem}_resampled.tif"
                    )
                    geotransform = calculateGeotransformXY(
                        x=nobs_data.lon, y=nobs_data.lat
                    )
                    write_geotiff(
                        nobs_data.values,
                        output_file_nobs,
                        projection=f.crs.attrs["spatial_ref"],
                        geotransform=geotransform,
                        datatype="uint8",
                        nodata=255,
                    )

                    # Resample NOBS to 1km using rasterio
                    resample_raster_rasterio(
                        output_file_nobs, output_file_nobs_resampled, new_res, "average"
                    )

                    f.close()
                    ndvi = ndvi_data = nobs = nobs_data = None

                    # Load resampled datasets using rasterio
                    ndvi_meta = read_raster_metadata_rasterio(
                        output_file_ndvi_resampled
                    )
                    ndvi_data = ndvi_meta["data"]
                    nobs_meta = read_raster_metadata_rasterio(
                        output_file_nobs_resampled
                    )
                    nobs_data = nobs_meta["data"]

                    # Mask pixels without observations (NOBS = 0)
                    ndvi_data = np.where(nobs_data > 0, ndvi_data, 255)

                    output_file = outdir / f"{date}.tif"
                    write_geotiff(
                        ndvi_data,
                        output_file,
                        projection=ndvi_meta["projection"],
                        geotransform=ndvi_meta["geotransform"],
                        datatype="uint8",
                        nodata=255,
                    )

                    # Remove intermediate files
                    os.remove(output_file_ndvi)
                    os.remove(output_file_ndvi_resampled)
                    os.remove(output_file_nobs)
                    os.remove(output_file_nobs_resampled)

    return


def get_ndvi_data_terrascope(
    aoi_gpkg: Path,
    outdir: Path,
    start_date: str,
    end_date: str,
    overwrite: bool = False,
):
    """
    Get 1 km NDVI data from global Copernicus Land collection available on Terrascope.

    Parameters
    ----------
    aoi_gpkg : Path
        Path to the geopackage file defining the AOI
    outdir : Path
        Path to the output directory
    start_date : str
        Start date of the temporal range, format 'YYYY-MM-DD'
    end_date: str
        End date of the temporal range, format 'YYYY-MM-DD'
    overwrite: bool
        Whether to overwrite existing files

    """
    logger.info("Getting NDVI data...")

    # Set up output dir
    outdir.mkdir(parents=True, exist_ok=True)

    # Temporal extent
    temporal_extent_1km, temporal_extent_300m = get_temporal_extent(
        start_date=start_date, end_date=end_date
    )

    # Spatial extent
    aoi_adjusted_file = get_spatial_extent(aoi_gpkg)

    # Download NDVI data (openEO or copy from local folder)
    if temporal_extent_1km:
        res = "1km"
        get_data_local_folder(
            aoi_adjusted_file,
            res,
            temporal_extent_1km,
            outdir,
            overwrite=overwrite,
        )

    if temporal_extent_300m:
        res = "300m"
        get_data_local_folder(
            aoi_adjusted_file,
            res,
            temporal_extent_300m,
            outdir,
            overwrite=overwrite,
        )

    logger.success(f"All NDVI data downloaded to {outdir}")

    return


def create_invalid_pixel_mask(ndvi_smoothed_file: Path, outfile: Path):
    """
    Determine historically invalid pixels based on long term trends.

    Parameters
    ----------
    ndvi_smoothed_file : Path
        File with upper envelope filtered NDVI data
    outfile : Path
        Output file path of invalid pixels mask

    Returns
    -------
    invalid_pixel_mask : 2D array
        Array with invalid pixel mask, also saved in archive_dir/invalid_pixels_historical.tif
    """
    logger.info("Building map of invalid pixels...")

    # Get the NDVI data
    ndvi_smoothed, metadata = read_geotiff(
        ndvi_smoothed_file, apply_scaling=True, return_metadata=True
    )
    ndates, nx, ny = ndvi_smoothed.shape

    logger.info(f"Total number of pixels: {nx * ny}")

    # Mask water pixels (check if all values are nan along first axis)
    water_mask = np.all(np.isnan(ndvi_smoothed), axis=0)

    logger.info(
        f"Number of pixels masked due to no valid NDVI data: {np.sum(water_mask)}"
    )

    # Mask pixels with extreme long term trends
    ndvi_smoothed = np.reshape(ndvi_smoothed, (ndates, nx * ny))
    x = np.arange(len(ndvi_smoothed))
    slope_all_years = np.zeros(nx * ny)
    for i in tqdm(range(nx * ny)):
        pixel = ndvi_smoothed[:, i]
        pixel = pixel[~np.isnan(pixel)]

        if len(pixel) > 0:
            x1 = x[~np.isnan(pixel)]
            slope = stats.linregress(x1, pixel).slope
            slope_all_years[i] = slope
        else:
            slope_all_years[i] = 1

    slope_all_years = np.reshape(slope_all_years, [nx, ny])
    invalid_all_years = np.array(slope_all_years < -0.084883047) | np.array(
        slope_all_years > 0.0814077858
    )
    # ignore the ones that were already masked out
    invalid_all_years[water_mask] = False

    logger.info(
        f"Number of pixels masked due to long term trends: {np.sum(invalid_all_years)}"
    )

    # Combine water mask and long term trends mask into one masked pixel file
    invalid_pixels_mask = invalid_all_years | water_mask

    logger.info(f"Number of valid pixels remaining: {np.sum(~invalid_pixels_mask)}")

    # Create separate layer with distinction between two criteria
    mask_classified = invalid_all_years.astype(np.uint8)
    mask_classified[mask_classified == 1] = 2
    mask_classified[water_mask] = 1

    # make sure nodata is set to 255
    invalid_pixels_mask = invalid_pixels_mask.astype(np.uint8)
    mask_classified[mask_classified == 0] = 255

    # Write final results to file
    write_geotiff(
        invalid_pixels_mask,
        outfile,
        epsg=metadata["epsg"],
        bounds=metadata["bounds"],
        band_names=["invalid_pixel_mask"],
        datatype="uint8",
        nodata=255,
    )
    outfile_classified = str(outfile).replace(".tif", "_classified.tif")
    write_geotiff(
        mask_classified,
        outfile_classified,
        epsg=metadata["epsg"],
        bounds=metadata["bounds"],
        band_names=["invalid_pixel_mask_classified"],
        datatype="uint8",
        nodata=255,
    )

    logger.success(f"Invalid pixels mask saved to {outfile}")

    return invalid_pixels_mask


def determine_clusters_kmeans(
    stats_per_dekad_file: Path,
    cpsz_file: Path,
    min_zones: int,
    max_zones: int,
    sub_sample: int,
):
    """
    Perform clustering algorithm to determine between a number of
    min_zones and max_zones clusters using scikit-learn K-means.

    Parameters
    ----------
    stats_per_dekad_file : Path
        File with statistics datacube
    cpsz_file : Path
        Output filename for CPS zones
    min_zones : int
        Minimal number of zones that may be determined
    max_zones : int
        Maximal number of zones that may be determined
    sub_sample : int
        Subsampling factor (1 = use all pixels, 10 = use every 10th pixel)
    Returns
    -------
    None (clustering is saved in cpsz_file)
    """
    logger.info("Clustering with scikit-learn implementation of k-means...")

    # Read the input data
    data, meta = read_geotiff(
        stats_per_dekad_file, apply_scaling=False, return_metadata=True
    )
    bands, height, width = data.shape

    # Reshape data to 2D array (pixels x bands)
    data_reshaped = data.reshape(bands, height * width).T

    # Remove pixels with all zeros (invalid pixels)
    valid_mask = ~np.all(data_reshaped == 0, axis=1)
    valid_pixels = data_reshaped[valid_mask]

    logger.info(f"Total pixels: {len(data_reshaped)}")
    logger.info(f"Valid pixels: {len(valid_pixels)}")

    # Apply subsampling if specified
    if sub_sample > 1:
        n_samples = len(valid_pixels) // sub_sample
        sample_indices = np.random.choice(len(valid_pixels), n_samples, replace=False)
        sample_pixels = valid_pixels[sample_indices]
        logger.info(f"Subsampled to: {len(sample_pixels)} pixels")
    else:
        sample_pixels = valid_pixels
        sample_indices = np.arange(len(valid_pixels))

    # Standardize the data for better clustering
    scaler = StandardScaler()
    sample_pixels_scaled = scaler.fit_transform(sample_pixels)

    # Try different numbers of clusters and find the best one
    best_n_clusters = min_zones
    best_score = -1
    best_inertia = float("inf")

    logger.info(f"Testing cluster numbers from {min_zones} to {max_zones}")

    cluster_scores = []
    for n_clusters in tqdm(range(min_zones, max_zones + 1)):
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)

        try:
            cluster_labels = kmeans.fit_predict(sample_pixels_scaled)

            # Calculate silhouette score (higher is better)
            if len(np.unique(cluster_labels)) > 1:
                sil_score = silhouette_score(sample_pixels_scaled, cluster_labels)
                inertia = kmeans.inertia_

                # Combine silhouette score and inertia for selection criterion
                # We want high silhouette score and reasonable inertia
                combined_score = sil_score - (inertia / 1000000)  # Scale inertia down

                cluster_scores.append((n_clusters, sil_score, inertia, combined_score))

                # Update best if this is better
                if combined_score > best_score:
                    best_score = combined_score
                    best_n_clusters = n_clusters
                    best_inertia = inertia

        except Exception as e:
            logger.error(f"Error with {n_clusters} clusters: {e}")
            continue

    logger.info(
        f"Selected {best_n_clusters} clusters with silhouette score: {best_score:.3f}"
    )

    # Perform final clustering with the best number of clusters
    final_kmeans = KMeans(
        n_clusters=best_n_clusters, random_state=42, n_init=10, max_iter=300
    )

    if sub_sample > 1:
        # Fit on sample, then predict on all valid pixels
        final_kmeans.fit(sample_pixels_scaled)
        valid_pixels_scaled = scaler.transform(valid_pixels)
        all_labels = final_kmeans.predict(valid_pixels_scaled)
    else:
        all_labels = final_kmeans.fit_predict(sample_pixels_scaled)

    # Create output array
    output_array = np.zeros(height * width, dtype=np.int16)
    output_array.fill(0)  # Background/invalid pixels = 0

    # Assign cluster labels to valid pixels (add 1 so clusters start from 1)
    output_array[valid_mask] = all_labels + 1

    # Reshape back to 2D
    output_array = output_array.reshape(height, width)

    # Save the result
    write_geotiff(
        output_array,
        cpsz_file,
        bounds=meta["bounds"],
        epsg=meta["epsg"],
        datatype="int16",
        nodata=0,
    )

    logger.success(f"Clustering saved to: {cpsz_file}")
    logger.info(f"Final number of clusters: {best_n_clusters}")

    return


def compute_stats_per_dekad(
    ndvi_smoothed_file: Path, invalid_pixel_mask_file: Path, outfile: Path
):
    """
    Compute percentiles p10, p50, p90, and the standard deviation of data stored
    in filtered_cropped_file per dekad and combine all into one datacube that
    will be used in the IsoData clustering.
    Pixels in invalid_pixels_mask and pixels with more than 17/20 years of invalid data are masked.

    Parameters
    ----------
    ndvi_smoothed_file : Path
        File with upper envelope smoothed NDVI data
    invalid_pixel_mask_file : Path
        File path for invalid pixel mask
    outfile : Path
        Output file path for statistics datacube

    Returns
    -------
    None (datacube containing masked p10, p50, p90 and std values
         is saved to tiff file)
    """
    logger.info("Deriving statistics to be used in clustering...")

    # Get the NDVI data
    ndvi_smoothed, metadata = read_geotiff(
        ndvi_smoothed_file, apply_scaling=True, return_metadata=True
    )
    dekads, nx, ny = ndvi_smoothed.shape

    # Get invalid pixels mask
    invalid_pixels_mask = read_geotiff(invalid_pixel_mask_file, apply_scaling=False)
    # convert to boolean
    invalid_pixels_mask = invalid_pixels_mask.astype(bool)

    # initiate the output
    p10_year = np.zeros((36, nx, ny)).astype(float)
    p50_year = np.zeros((36, nx, ny)).astype(float)
    p90_year = np.zeros((36, nx, ny)).astype(float)
    std_year = np.zeros((36, nx, ny)).astype(float)
    for i in range(36):
        dekad_bands = np.arange(i, dekads, 36)
        cube = ndvi_smoothed[dekad_bands, :, :]

        # Check if more than 17/20 dekads/pixel are invalid (invalid=True, valid=False)
        dekad_mask = np.sum(np.isnan(cube), axis=0) > 17

        # Compute percentiles and stdev
        p10 = np.nanpercentile(cube, 10, axis=0)
        p50 = np.nanpercentile(cube, 50, axis=0)
        p90 = np.nanpercentile(cube, 90, axis=0)
        std = np.std(cube, axis=0)

        # Apply invalid pixel mask and dekad mask
        total_mask = invalid_pixels_mask | dekad_mask
        p10 = np.where(total_mask, np.nan, p10)
        p50 = np.where(total_mask, np.nan, p50)
        p90 = np.where(total_mask, np.nan, p90)
        std = np.where(total_mask, np.nan, std)

        p10_year[i, :, :] = p10
        p50_year[i, :, :] = p50
        p90_year[i, :, :] = p90
        std_year[i, :, :] = std

    # Combine all layers into one cube (for clustering later on)
    cube_to_cluster = np.concatenate((p10_year, p50_year, p90_year, std_year), axis=0)
    cube_to_cluster[np.isnan(cube_to_cluster)] = (
        0  # Set nan equal to 0 (needed for clustering algorithm)
    )

    # Write result to file
    write_geotiff(
        cube_to_cluster,
        outfile,
        bounds=metadata["bounds"],
        epsg=metadata["epsg"],
        datatype="float32",
        nodata=0,
    )

    logger.success(f"Dekadal statistics saved to {outfile}")

    return


def get_dekads():
    """Get list of unique dekads in a year."""

    days = ["01", "11", "21"]
    months = [month for month in range(1, 13)]
    dekads = [f"{b:0>2d}{a}" for b in months for a in days]
    return dekads


def compute_payout_thresholds(
    ndvi_smoothed_file: Path,
    cpsz_file: Path,
    outdir_percentiles: Path,
    percentile_numbers: list = PERCENTILE_THRESHOLDS,
):
    """
    Compute the payout thresholds, i.e. the limiting percentiles of smoothed NDVI values
    per dekad and per CPS zone.

    Parameters
    ----------
    ndvi_smoothed_file : Path
        Filename for smoothed NDVI values
    cpsz_file : Path
        File with CPS zones
    outdir_percentiles : Path
        Output directory to save percentiles
    percentile_numbers : list
        List of integers for which percentile should be computed (example: [5,15] -> p5 and p15)

    Returns
    -------
    percentiles : dict
        Percentiles (given in percentile_numbers) per dekad and per CPS zone
    p50_array : 2D array
        p50 percentile per dekad and per CPS zone
    """
    logger.info("Computing payout thresholds per CPS zone...")

    # Make sure output dir exists
    outdir_percentiles.mkdir(parents=True, exist_ok=True)

    # Get smoothed NDVI data
    ndvi_smoothed, metadata = read_geotiff(
        ndvi_smoothed_file, apply_scaling=True, return_metadata=True
    )
    ndekads, nx, ny = ndvi_smoothed.shape
    dekads = get_dekads()

    # Load cpszs
    cpszs = read_geotiff(cpsz_file, apply_scaling=False, return_metadata=False)
    zones = np.unique(cpszs)
    # ignore zero
    if 0 in zones:
        zones = zones[zones != 0]
    nzones = len(zones)

    # Get percentiles arrays
    percentiles = {}
    for p in percentile_numbers:
        percentiles[f"p{p:02}"] = np.full((36, nx, ny), 255).astype(int)
    p50_array = np.zeros((36, nzones))

    for i in range(36):
        dekad_bands = np.arange(i, ndekads, 36)
        cube = ndvi_smoothed[dekad_bands, :, :]

        lta = np.full((nx, ny), 255).astype(int)
        for zone in zones:
            for p in percentile_numbers:
                perc = np.nanpercentile(cube[:, cpszs == zone], q=p)
                if np.isnan(perc):
                    perc = 255
                percentiles[f"p{p:02}"][i, cpszs == zone] = int(np.round(perc))

            mean = np.nanmean(cube[:, cpszs == zone])
            if np.isnan(mean):
                mean == 255
            lta[cpszs == zone] = int(np.round(mean))

            perc50 = np.nanpercentile(cube[:, cpszs == zone], q=50)
            p50_array[i, int(zone) - 1] = perc50

        # Output percentiles files
        for p in percentile_numbers:
            im = percentiles[f"p{p:02}"][i, :, :]
            im[np.isnan(im)] = 255
            im = im.astype(np.uint8)
            outfile = outdir_percentiles / f"percentiles{p}_{dekads[i]}.tif"
            write_geotiff(
                im,
                outfile,
                epsg=metadata["epsg"],
                bounds=metadata["bounds"],
                datatype="uint8",
                nodata=255,
            )

        # Output LTA (mean per zone)
        outfile = outdir_percentiles / f"lta_{dekads[i]}.tif"
        write_geotiff(
            lta,
            outfile,
            epsg=metadata["epsg"],
            bounds=metadata["bounds"],
            datatype="uint8",
            nodata=255,
        )

    logger.success(f"NDVI LTA and percentiles saved to {outdir_percentiles}")

    # Save p50_array in csv file
    outfile = outdir_percentiles / "p50_array.csv"
    with open(outfile, "w") as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(zones)
        csvWriter.writerows(p50_array)

    logger.success(f"p50 values saved to {outfile}")

    return percentiles, p50_array


def define_growing_seasons(p50_array, cpsz_file, outdir_seasons):
    """
    Determine growing seasons per CPS zone from median NDVI values.

    Parameters
    ----------
    p50_array : 2D array
        p50 percentile per dekad and per CPS zone
    cpsz_file : str
        File with CPS zones
    outdir_seasons : str
        Output folder for season data

    Returns
    -------
    seasons : 2D array
        Boolean array with growing seasons per CPS zone
    """
    logger.info("Computing seasons...")

    # Make sure output directory exists
    outdir_seasons.mkdir(parents=True, exist_ok=True)

    def moving_average(x):
        w = np.ones(9)
        return np.convolve(x, w, "valid") / 9

    # Get the zones
    cpszs, meta = read_geotiff(cpsz_file, apply_scaling=False, return_metadata=True)
    zone_labels = np.unique(cpszs)
    # ignore zero
    if 0 in zone_labels:
        zone_labels = zone_labels[zone_labels != 0]

    # Get number of dekads and zones from lta data
    ndekads, nzones = p50_array.shape

    # Derive growing seasons
    seasons = np.zeros((ndekads, nzones))
    NDVI_thr = 70
    ratio_thr = 0.95
    for i in range(nzones):
        zone_profile = p50_array[:, i]
        max_val = np.max(zone_profile)
        zone_profile2 = np.tile(zone_profile, 2)  # Repeat data
        ma9 = moving_average(zone_profile2)
        zone_profile = zone_profile2[36:]
        ma9 = ma9[28:]
        flag_50perc_ma9 = zone_profile > ma9
        flag_50perc_NDVIthr = zone_profile > NDVI_thr
        flag_50perc_maxval = zone_profile / max_val > ratio_thr

        flag_season = (flag_50perc_ma9 & flag_50perc_NDVIthr) | (
            flag_50perc_maxval & flag_50perc_NDVIthr
        )

        seasons[:, i] = np.array(flag_season).astype(
            int
        )  # 1 = in season, 0 = out of season

    # Write seasons to file
    out_csv_file = outdir_seasons / "seasons.csv"
    with open(out_csv_file, "w") as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(zone_labels)
        csvWriter.writerows(seasons)

    # Write seasons to geotiff files
    seasons_all_dekads = pd.read_csv(out_csv_file)

    i, j = 1, 1
    for dekad in range(36):
        zones_dekad = seasons_all_dekads.iloc[dekad]
        seasons_dekad = np.zeros(cpszs.shape)
        seasons_dekad = np.where(cpszs == 0, 255, seasons_dekad)
        for zone_idx, zone_value in zones_dekad.items():
            seasons_dekad = np.where(
                cpszs == int(zone_idx), int(zone_value), seasons_dekad
            )

        dekad_date = f"{i:02}{j:02}"
        if j != 21:
            j += 10
        else:
            i += 1
            j = 1

        outfile = outdir_seasons / f"seasons_dekad_{dekad_date}.tif"
        write_geotiff(
            seasons_dekad,
            outfile,
            epsg=meta["epsg"],
            bounds=meta["bounds"],
            nodata=255,
            datatype="uint8",
        )

    logger.success(f"Seasons saved to {outdir_seasons}")

    return seasons


def _get_percentiles_dekad(
    final_thresholds_dir, lower_percentile, upper_percentile, dekad
):
    """
    Get the lower and upper percentiles for a specific dekad.

    Parameters
    ----------
    final_thresholds_dir : Path
        Directory containing the final thresholds
    lower_percentile : int
        Value of the lower percentile
    upper_percentile : int
        Value of the upper percentile
    dekad : str
        Dekad identifier (e.g. "2021-01-01")

    Returns
    -------
    lower_percentile_data : np.ndarray
        Lower percentile data
    upper_percentile_data : np.ndarray
        Upper percentile data
    """
    lower_percentile_file = (
        final_thresholds_dir / f"percentiles{lower_percentile}_{dekad}.tif"
    )
    lower_percentile_data = read_geotiff(lower_percentile_file, apply_scaling=True)
    upper_percentile_file = (
        final_thresholds_dir / f"percentiles{upper_percentile}_{dekad}.tif"
    )
    upper_percentile_data = read_geotiff(upper_percentile_file, apply_scaling=True)

    return lower_percentile_data, upper_percentile_data


def compute_vici(
    basedir: Path,
    ndvi_smoothed_file: Path,
    start_date: str,
    end_date: str,
    lower_percentile: int,
    upper_percentile: int,
    outdir: Path,
    overwrite: bool = False,
):
    """
    Compute the VICI values from the smoothed NDVI and percentiles.

    Parameters
    ----------
    basedir : Path
        Base directory containing the NDVI and archival data
    ndvi_smoothed_file : Path
        Path to the smoothed NDVI file
    start_date : str
        Start date of temporal extent
    end_date : str
        End date of temporal extent
    lower_percentile : int
        Value of the lower percentile
    upper_percentile : int
        Value of the upper percentile
    outdir : Path
        Path of the output directory
    overwrite : bool
        Whether to overwrite existing output files

    Returns
    -------
    None (all results are save automatically as .tif files per dekad)
    """
    archive_dir = basedir / "NDVI_archive"

    # Get the smoothed NDVI data
    ndvi_smoothed, meta = read_geotiff(
        ndvi_smoothed_file, apply_scaling=True, return_metadata=True
    )

    # Load cpsz to set quality flags later on
    cpsz_file = archive_dir / "cpsz.tif"
    cpsz = read_geotiff(cpsz_file, apply_scaling=False, return_metadata=False)
    in_cpsz = cpsz > 0  # 0=False, 1=True

    # Load invalid pixel mask
    invalid_pixel_mask_file = archive_dir / "invalid_pixel_mask.tif"
    invalid_pixels = read_geotiff(
        invalid_pixel_mask_file, apply_scaling=False, return_metadata=False
    )

    # Folders with historical data
    final_thresholds_dir = archive_dir / "final_thresholds"

    # All dates within the temporal extent
    dates = []
    for d in pd.date_range(start_date, end_date):
        if d.day in [1, 11, 21]:
            dates.append(d.strftime("%Y-%m-%d"))

    # Loop over all dates and compute VICI
    for i in range(len(dates)):
        date = dates[i]
        dekad = date[5:].replace("-", "")
        outfile_vici = outdir / f"VICI_{date.replace('-', '')}.tif"

        if not outfile_vici.exists() or overwrite:
            # Get NDVI data
            ndvi_dekad = ndvi_smoothed[i, :, :]

            # Get percentiles for this dekad
            lower_percentile_data, upper_percentile_data = _get_percentiles_dekad(
                final_thresholds_dir, lower_percentile, upper_percentile, dekad
            )

            # Compute VICI
            vici = (upper_percentile_data - ndvi_dekad) / (
                upper_percentile_data - lower_percentile_data
            )

            # Clip values to [0,1] range and multiply by 100 to get percentage
            vici[vici < 0] = 0
            vici[vici > 1] = 1
            vici = vici * 100
            vici = np.where(np.isnan(vici), 255, vici)
            vici = vici.astype(np.uint8)

            # Save VICI to file
            write_geotiff(
                vici,
                outfile_vici,
                epsg=meta["epsg"],
                bounds=meta["bounds"],
                datatype="uint8",
                nodata=255,
            )

            # Construct quality flags
            quality_flags = np.zeros_like(vici, dtype=np.uint8)
            # Missing data (252)
            ndvi_ori_dir = basedir / "NDVI" / "NDVI_original"
            ndvi_orig_file = glob.glob(str(ndvi_ori_dir / f"*{date}*.tif"))[0]
            ndvi_orig = read_geotiff(ndvi_orig_file, apply_scaling=False)
            not_in_ndvi = ndvi_orig > 250
            missing_data = not_in_ndvi & in_cpsz
            quality_flags[missing_data == 1] = 252
            # Out of season (253)
            seasons_dir = archive_dir / "seasons"
            season_dekad_file = seasons_dir / f"seasons_dekad_{dekad}.tif"
            season = read_geotiff(season_dekad_file, apply_scaling=False)
            out_of_season = (season == 0) & in_cpsz
            quality_flags[out_of_season == 1] = 253
            # Outside cpszs (255)
            quality_flags[cpsz == 0] = 255
            # Invalid pixels (254)
            quality_flags[invalid_pixels == 1] = 254

            # Save quality flags layer
            quality_flags_outfile = (
                outdir / f"quality_flags_{date.replace('-', '')}.tif"
            )

            write_geotiff(
                quality_flags,
                quality_flags_outfile,
                epsg=meta["epsg"],
                bounds=meta["bounds"],
                datatype="uint8",
                nodata=0,
            )

    logger.success(f"VICI and quality flags saved to {outdir}")

    return


def run_vici(
    basedir: Path,
    start_date,
    end_date,
    percentile_numbers: list = PERCENTILE_THRESHOLDS,
):
    """
    Compute VICI for the given country and temporal period.

    Parameters
    ----------
    basedir : Path
        Base directory where all results are stored
    start_date : str
        Start date of temporal extent
    end_date : str
        End date of temporal extent
    percentile_numbers :  list
        List of percentile numbers to use for VICI computation,
        by default 5 and 15

    Returns
    -------
    None (all results are save automatically as .tif files per dekad)

    """
    logger.info(f"Period: {start_date} - {end_date}")

    # Percentiles that will be used in VICI computation
    lower_percentile = percentile_numbers[0]
    upper_percentile = percentile_numbers[1]

    # Output directory
    outdir = basedir / "VICI" / f"p{lower_percentile}_p{upper_percentile}"
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {outdir}")

    # Make sure we have all NDVI data that we need
    aoi_gpkg_file = basedir / "AOI.gpkg"
    ndvi_dir = basedir / "NDVI"
    ndvi_ori_dir = ndvi_dir / "NDVI_original"
    get_ndvi_data_terrascope(aoi_gpkg_file, ndvi_ori_dir, start_date, end_date)

    # Run upper envelope filtering on the NDVI data
    ndvi_smoothed_file = ndvi_dir / f"NDVI_smoothed_{start_date}_{end_date}.tif"
    upper_envelope_smoothing(ndvi_ori_dir, start_date, end_date, ndvi_smoothed_file)

    # Compute VICI and quality flags
    compute_vici(
        basedir,
        ndvi_smoothed_file,
        start_date,
        end_date,
        lower_percentile,
        upper_percentile,
        outdir,
    )

    return


def upper_envelope_smoothing(
    ndvi_dir: Path, start_date: str, end_date: str, outfile: Path
):
    """Apply upper envelope smoothing to NDVI data.

    Parameters
    ----------
    ndvi_dir : Path
        Directory containing NDVI data files.
    start_date : str
        Start date for the smoothing period, format YYYY-MM-DD.
    end_date : str
        End date for the smoothing period, format YYYY-MM-DD.
    outfile : Path
        Output file path for the smoothed NDVI data.

    Returns
    -------
    None
    """
    logger.info("Applying upper envelope smoothing...")

    # Read all available NDVI data into single array
    ndvi_data = []
    infiles = sorted(glob.glob(str(ndvi_dir / "*.tif")))
    logger.info(f"Found {len(infiles)} NDVI files.")
    list_of_dates = [Path(f).stem for f in infiles]
    for file_path in infiles:
        data, ndvi_meta = read_geotiff(
            file_path, apply_scaling=True, return_metadata=True
        )
        ndvi_data.append(data)
    ndvi_data = np.array(ndvi_data)

    # Apply upper envelope filtering
    smoothed = np.apply_along_axis(upper_envelop, 0, ndvi_data)

    # Ensure the data is still in range [0 - 250]
    smoothed[smoothed > 250] = 250
    smoothed[smoothed < 0] = 1

    # Restore nodata value to 255
    smoothed[smoothed == 0] = 255
    smoothed = smoothed.astype(np.uint8)

    # Now get rid of any additional data that was added
    def find_date_in_list(date_str, date_list, mode="smaller"):

        # convert to datetime
        date = pd.to_datetime(date_str)
        date_list = pd.to_datetime(date_list)
        if mode == "smaller":
            for i, d in enumerate(date_list):
                if d >= date:
                    return i
        if mode == "larger":
            # iterate list backwards
            for i, d in enumerate(reversed(date_list)):
                if d <= date:
                    return len(date_list) - i - 1
        return None

    start_index = find_date_in_list(start_date, list_of_dates, mode="smaller")
    end_index = find_date_in_list(end_date, list_of_dates, mode="larger")
    ndvi_data = ndvi_data[start_index : end_index + 1]
    smoothed = smoothed[start_index : end_index + 1]
    new_list_of_dates = list_of_dates[start_index : end_index + 1]
    band_names = [f"NDVI_{i}" for i in new_list_of_dates]
    nbands = len(band_names)

    # Write to one geotiff file
    write_geotiff(
        smoothed,
        outfile,
        epsg=ndvi_meta["epsg"],
        bounds=ndvi_meta["bounds"],
        band_names=band_names,
        scales=ndvi_meta["scales"] * nbands,
        offsets=ndvi_meta["offsets"] * nbands,
        datatype=ndvi_meta["dtype"],
        nodata=ndvi_meta["nodata"],
    )

    logger.success(f"Smoothed NDVI timeseries saved to {outfile}")

    return


def compute_vici_zonal_stats(basedir: Path, percentile_numbers=PERCENTILE_THRESHOLDS):

    # Get VICI data
    lower_percentile = percentile_numbers[0]
    upper_percentile = percentile_numbers[1]
    vici_dir = basedir / "VICI" / f"p{lower_percentile}_p{upper_percentile}"
    vici_files = sorted(glob.glob(str(vici_dir / "VICI_*.tif")))
    vici = []
    for vici_file in vici_files:
        vici.append(read_geotiff(vici_file))
    vici = np.array(vici)

    # Get zones
    cpsz_file = basedir / "NDVI_archive" / "cpsz.tif"
    cpsz = read_geotiff(cpsz_file, apply_scaling=False)
    zones = np.unique(cpsz)
    # ignore zero
    if 0 in zones:
        zones = zones[zones != 0]

    # Compute zonal statistics
    zonal_stats = {}
    for zone in zones:
        zonal_stats[str(zone)] = {}
        zonal_mask = np.tile(cpsz == zone, (vici.shape[0], 1, 1)).astype(float)
        zonal_mask[zonal_mask == 0] = np.nan
        zonal_vici = vici * zonal_mask
        npixels = np.sum(cpsz == zone)
        zonal_stats[str(zone)]["mean"] = np.nanmean(zonal_vici)
        zonal_stats[str(zone)]["min"] = np.nanmin(zonal_vici)
        zonal_stats[str(zone)]["max"] = np.nanmax(zonal_vici)
        larger_than_zero = np.sum(zonal_vici > 0)
        zonal_stats[str(zone)]["freq"] = (
            larger_than_zero / (vici.shape[0] * npixels) * 100
        )

    return zonal_stats


def show_dekadal_vici_result(
    basedir: Path, dekad: str, percentile_numbers=PERCENTILE_THRESHOLDS
):
    """Show VICI results for a specific dekad as raster.

    Parameters
    ----------
    basedir : Path
        Path to where VICI results are stored.
    dekad : str
        The dekad to visualize in format %Y%m%d (e.g. 20210101)
    percentile_numbers : tuple
        The lower and upper percentile thresholds that were used for VICI computation.

    Raises
    ------
    FileNotFoundError
        If the VICI or quality flags files are not found.
    """

    # Find the correct files and read them
    lower_percentile = percentile_numbers[0]
    upper_percentile = percentile_numbers[1]
    vici_dir = basedir / "VICI" / f"p{lower_percentile}_p{upper_percentile}"
    vici_file = Path(glob.glob(str(vici_dir / f"VICI_{dekad}.tif"))[0])
    if not vici_file.exists():
        raise FileNotFoundError("VICI file not found.")
    vici = read_geotiff(vici_file)
    quality_flags_file = vici_file.parent / f"quality_flags_{dekad}.tif"
    if not quality_flags_file.exists():
        raise FileNotFoundError("Quality flags file not found.")
    quality_flags = read_geotiff(quality_flags_file)

    # Create figure with proper subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Plot VICI raster
    im1 = axes[0].imshow(vici, cmap="viridis", vmin=0, vmax=100)
    axes[0].set_title("VICI Raster - Drought Severity")
    axes[0].axis("off")  # Remove axis ticks for cleaner look
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Create a custom colormap for quality flags
    flag_values = sorted(list(QUALITY_FLAGS.keys()))
    min_flag_value = flag_values[0]
    max_flag_value = flag_values[-1]
    n_flags = len(flag_values)
    colors = cm.tab10(np.linspace(0, 1, 10))[:n_flags]  # Take only first n_flags colors
    custom_cmap = ListedColormap(colors)

    # Plot Quality Flags with proper colormap and labels
    im2 = axes[1].imshow(
        quality_flags, cmap=custom_cmap, vmin=min_flag_value, vmax=max_flag_value
    )
    axes[1].set_title("Quality Flags Raster")
    axes[1].axis("off")  # Remove axis ticks for cleaner look
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_ticks(flag_values)
    cbar2.set_ticklabels([QUALITY_FLAGS.get(flag, "Unknown") for flag in flag_values])

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print("\nVICI Statistics:")
    print(f"Min: {np.nanmin(vici):.2f}")
    print(f"Max: {np.nanmax(vici):.2f}")
    print(f"Mean: {np.nanmean(vici):.2f}")
    print(
        f"Pixels with drought (VICI > 0): {np.sum(vici > 0)} ({np.sum(vici > 0)/np.sum(~np.isnan(vici))*100:.1f}%)"
    )

    print("\nQuality Flags Distribution:")
    quality_unique = np.unique(quality_flags)
    # remove nan
    quality_unique = quality_unique[~np.isnan(quality_unique)]
    for flag in sorted(quality_unique):
        count = np.sum(quality_flags == flag)
        percentage = count / quality_flags.size * 100
        meaning = QUALITY_FLAGS.get(flag, "Unknown")
        print(f"Flag {flag} ({meaning}): {count} pixels ({percentage:.1f}%)")


def show_vici_result_pixel(
    basedir, x, y, start_date, end_date, percentile_numbers=PERCENTILE_THRESHOLDS
):
    """Show VICI results for a specific pixel.

    Parameters
    ----------
    basedir : Path
        Path to where VICI results are stored.
    x : int
        The x-coordinate of the pixel.
    y : int
        The y-coordinate of the pixel.
    start_date : str
        The start date for the analysis.
    end_date : str
        The end date for the analysis.
    percentile_numbers : tuple, optional
        The lower and upper percentile thresholds that were used for VICI computation.
    """

    # Get zone
    cpsz_file = basedir / "NDVI_archive" / "cpsz.tif"
    cpsz = read_geotiff(cpsz_file, apply_scaling=False)
    zone = str(cpsz[x, y])
    if zone == 0:
        logger.warning(f"Pixel {x}, {y} is not part of any zone")
        return
    else:
        logger.info(f"Pixel {x}, {y} is part of zone {zone}")

    # Get zonal median
    final_thresholds_dir = basedir / "NDVI_archive" / "final_thresholds"
    p50_array_file = final_thresholds_dir / "p50_array.csv"
    p50_array = pd.read_csv(p50_array_file, header=0)
    p50 = p50_array.iloc[:, int(zone) - 1].values

    # Get percentiles
    lower_percentile = percentile_numbers[0]
    upper_percentile = percentile_numbers[1]
    dekads = get_dekads()
    lower_percentiles = []
    upper_percentiles = []
    for dekad in dekads:
        lower_percentile_data, upper_percentile_data = _get_percentiles_dekad(
            final_thresholds_dir, lower_percentile, upper_percentile, dekad
        )
        lower_percentiles.append(lower_percentile_data[x, y])
        upper_percentiles.append(upper_percentile_data[x, y])
    lower_percentiles = np.array(lower_percentiles)
    upper_percentiles = np.array(upper_percentiles)

    # Get VICI data
    vici_dir = basedir / "VICI" / f"p{lower_percentile}_p{upper_percentile}"
    vici_files = sorted(glob.glob(str(vici_dir / "VICI_*.tif")))
    vici = []
    for vici_file in vici_files:
        vici.append(read_geotiff(vici_file)[x, y])
    vici = np.array(vici)

    # Get smoothed NDVI data for this pixel
    ndvi_dir = basedir / "NDVI"
    ndvi_smoothed_file = ndvi_dir / f"NDVI_smoothed_{start_date}_{end_date}.tif"
    smoothed_ndvi = read_geotiff(ndvi_smoothed_file)[:, x, y]

    # Plot time series
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    axes = [axes]  # Ensure axes is a list for consistency
    ax1 = axes[0]
    ax1.plot(p50, label="Zonal Median (p50)", color="green", linestyle="--")
    ax1.plot(
        lower_percentiles,
        label=f"Exit (p{lower_percentile})",
        color="red",
        linestyle="--",
    )
    ax1.plot(
        upper_percentiles,
        label=f"Trigger (p{upper_percentile})",
        color="orange",
        linestyle="--",
    )
    ax1.plot(smoothed_ndvi, label="Smoothed NDVI", color="blue", linewidth=2)
    ax1.set_title(f"VICI Time Series for Pixel ({x}, {y})")
    ax1.set_xlim(-14, 41)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("NDVI")
    plt.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.bar(np.arange(len(vici)), vici, label="VICI", color="black", alpha=0.3)
    ax2.set_ylabel("VICI")
    ax2.set_ylim(0, 100)
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()
