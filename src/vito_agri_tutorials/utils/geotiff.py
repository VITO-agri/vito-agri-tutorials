import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.profiles import Profile
from rasterio.transform import Affine


class DefaultProfile(Profile):
    """Tiled, band-interleaved, LZW-compressed, 8-bit GTiff."""

    defaults = {
        "driver": "GTiff",
        "interleave": "band",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
        "dtype": "float32",
    }


def get_blocksize(val):
    """
    Blocksize needs to be a multiple of 16
    """
    if val % 16 == 0:
        return val
    else:
        return (val // 16) * 16


def get_rasterio_profile(arr, bounds, epsg, blockxsize=None, blockysize=None, **params):

    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, axis=0)

    base_profile = DefaultProfile()
    shape = arr.shape

    count, height, width = shape

    if blockxsize is None:
        blockxsize = get_blocksize(width)

    if blockysize is None:
        blockysize = get_blocksize(height)

    crs = CRS.from_epsg(epsg)

    base_profile.update(
        transform=rasterio.transform.from_bounds(*bounds, width=width, height=height),
        width=width,
        height=height,
        blockxsize=blockxsize,
        blockysize=blockysize,
        dtype=arr.dtype,
        crs=crs,
        count=count,
    )

    base_profile.update(**params)

    return base_profile


def get_rasterio_profile_shape(
    shape, bounds, epsg, dtype, blockxsize=1024, blockysize=1024, **params
):

    base_profile = DefaultProfile()

    if len(shape) == 2:
        shape = [1] + shape

    count, height, width = shape

    crs = CRS.from_epsg(epsg)

    base_profile.update(
        transform=rasterio.transform.from_bounds(*bounds, width=width, height=height),
        width=width,
        height=height,
        blockxsize=blockxsize,
        blockysize=blockysize,
        dtype=dtype,
        crs=crs,
        count=count,
    )

    base_profile.update(**params)

    return base_profile


def write_geotiff(
    arr: np.ndarray,
    filename: Path,
    profile=None,
    bounds=None,
    epsg=None,
    projection=None,
    geotransform=None,
    nodata=None,
    datatype=None,
    band_names=None,
    colormap=None,
    tags=None,
    bands_tags=None,
    scales=None,
    offsets=None,
    blockxsize=None,
    blockysize=None,
    **kwargs,
):
    """
    Unified function to write arrays to GeoTIFF files with flexible input options.

    This function allows multiple ways to specify spatial reference:
    1. Direct rasterio profile
    2. Bounds + EPSG code
    3. Projection + geotransform (GDAL-style)

    Parameters
    ----------
    arr : np.ndarray
        Data array to write (2D or 3D)
    filename : Path
        Output file path
    profile : dict, optional
        Rasterio profile. If provided, other spatial parameters are ignored.
    bounds : tuple, optional
        Bounding box (minx, miny, maxx, maxy). Used with epsg parameter.
    epsg : int, optional
        EPSG code. Used with bounds parameter.
    projection : int or str, optional
        EPSG code (int) or projection string (WKT). Used with geotransform.
    geotransform : tuple, optional
        GDAL-style geotransform (xmin, xres, 0, ymax, 0, -yres)
    nodata : int or float, optional
        No data value
    datatype : str, optional
        Data type ('uint8', 'int16', 'uint16', 'int32', 'float32', etc.)
    band_names : list, optional
        List of band names
    colormap : dict, optional
        Colormap for first band
    tags : dict, optional
        Dataset-level metadata tags
    bands_tags : list of dict, optional
        List of per-band metadata tags
    scales : list, optional
        Scale values for bands
    offsets : list, optional
        Offset values for bands
    blockxsize : int, optional
        Tile block width (default: 256)
    blockysize : int, optional
        Tile block height (default: 256)
    **kwargs
        Additional profile parameters

    Examples
    --------
    # Method 1: Using bounds and EPSG
    write_geotiff_unified(arr, 'output.tif', bounds=(0, 0, 10, 10), epsg=4326)

    # Method 2: Using projection and geotransform (GDAL-style)
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    write_geotiff_unified(arr, 'output.tif', projection=4326, geotransform=geotransform)

    # Method 3: Using existing profile
    profile = get_rasterio_profile(arr, bounds, epsg)
    write_geotiff_unified(arr, 'output.tif', profile=profile)
    """

    # Handle array dimensions
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)

    # Create or use provided profile
    if profile is not None:
        # Use provided profile
        out_profile = profile.copy()
    elif bounds is not None and epsg is not None:
        # Method 1: Create profile from bounds and EPSG
        out_profile = get_rasterio_profile(
            arr, bounds, epsg, blockxsize, blockysize, **kwargs
        )
    elif projection is not None and geotransform is not None:
        # Method 2: Create profile from projection and geotransform (GDAL-style)

        # Convert GDAL geotransform to rasterio transform
        transform = Affine(
            geotransform[1],  # xres
            geotransform[2],  # 0
            geotransform[0],  # xmin
            geotransform[4],  # 0
            geotransform[5],  # -yres
            geotransform[3],  # ymax
        )

        # Handle CRS
        if isinstance(projection, int):
            crs = CRS.from_epsg(projection)
        else:
            crs = CRS.from_wkt(projection)

        # Get array dimensions
        count, height, width = arr.shape

        # Create profile
        out_profile = DefaultProfile()
        out_profile.update(
            transform=transform,
            width=width,
            height=height,
            count=count,
            crs=crs,
            **kwargs,
        )
    else:
        raise ValueError(
            "Must provide either:\n"
            "1. profile parameter, or\n"
            "2. bounds + epsg parameters, or\n"
            "3. projection + geotransform parameters"
        )

    # Override profile settings with explicit parameters
    if nodata is not None:
        out_profile.update(nodata=nodata)
    if datatype is not None:
        out_profile.update(dtype=datatype)
    if blockxsize is not None:
        out_profile.update(blockxsize=blockxsize)
    if blockysize is not None:
        out_profile.update(blockysize=blockysize)

    # Ensure bands_tags is a list
    bands_tags = bands_tags if bands_tags is not None else []

    # Remove existing file
    if os.path.isfile(filename):
        os.remove(filename)

    # Write the file
    with rasterio.open(filename, "w", **out_profile) as dst:
        dst.write(arr)

        # Write dataset-level tags
        if tags is not None:
            dst.update_tags(**tags)

        # Write band names
        if band_names is not None:
            dst.update_tags(bands=band_names)
            for i, b in enumerate(band_names):
                dst.update_tags(i + 1, band_name=b)

        # Write per-band tags
        for i, bt in enumerate(bands_tags):
            if i < arr.shape[0]:  # Ensure we don't exceed number of bands
                dst.update_tags(i + 1, **bt)

        # Write colormap (for first band)
        if colormap is not None:
            dst.write_colormap(1, colormap)

        # Write scales and offsets
        if scales is not None:
            dst.scales = scales
        if offsets is not None:
            dst.offsets = offsets


def read_geotiff(
    infile, bandnr=None, apply_scaling: bool = True, return_metadata: bool = False
):
    """
    Read GeoTIFF file into a numpy array with optional metadata.

    Parameters
    ----------
    infile : str
        Path to the input GeoTIFF file
    bandnr : int or None, optional
        Band number to read (1-indexed). If None, reads all bands.
        Default is None (reads all bands).
    apply_scaling : bool, optional
        Whether to apply scale and offset values and convert nodata to NaN.
        Default is True.
    return_metadata : bool, optional
        Whether to return metadata along with the array.
        Default is False.

    Returns
    -------
    array : numpy.ndarray
        The raster data as numpy array. Shape is:
        - (height, width) for single band when bandnr is specified
        - (bands, height, width) for multi-band when bandnr is None
    metadata : dict, optional
        Metadata dictionary (only returned if return_metadata=True)
        Contains: epsg, bounds, geotransform, sizeX, sizeY, bands,
                 bandnames, scales, offsets, nodata

    Examples
    --------
    # Read single band
    array = read_geotiff('image.tif', bandnr=1)

    # Read all bands
    array = read_geotiff('image.tif')

    # Read with metadata
    array, metadata = read_geotiff('image.tif', return_metadata=True)
    """

    # Read data
    with rasterio.open(infile) as src:
        if bandnr is not None:
            # Read single band
            array = src.read(bandnr)
            nodata = src.nodata
            scale = src.scales[bandnr - 1] if bandnr <= len(src.scales) else 1.0
            offset = src.offsets[bandnr - 1] if bandnr <= len(src.offsets) else 0.0

            if apply_scaling:
                # Scale and apply nodata value
                if nodata is not None:
                    array = array.astype(np.float32)
                    array[array == nodata] = np.nan
                array = (array * scale) + offset

        else:
            # Read all bands
            array = src.read()  # Shape: (bands, height, width)
            # in case of single band, reduce dimensionality
            if array.shape[0] == 1:
                array = array.squeeze(0)  # Remove band dimension
            nodata = src.nodata
            scales = list(src.scales) if src.scales else [1.0] * src.count
            offsets = list(src.offsets) if src.offsets else [0.0] * src.count

            if apply_scaling:
                # Apply scaling to each band
                array = array.astype(np.float32)
                for i in range(array.shape[0]):
                    band_data = array[i]
                    scale = scales[i] if i < len(scales) else 1.0
                    offset = offsets[i] if i < len(offsets) else 0.0

                    # Apply nodata mask
                    if nodata is not None:
                        band_data[band_data == nodata] = np.nan

                    # Apply scaling
                    array[i] = (band_data * scale) + offset

        # Get metadata if requested
        if return_metadata:
            metadata = get_geotiff_metadata(infile)
            return array, metadata

    return array


def get_geotiff_metadata(infile):
    """
    Extract comprehensive metadata from a GeoTIFF file.

    Parameters
    ----------
    infile : str
        Path to the input GeoTIFF file

    Returns
    -------
    dict
        Dictionary containing raster metadata with keys:
        - epsg: EPSG code as string
        - crs: rasterio CRS object
        - bounds: BoundingBox (minx, miny, maxx, maxy)
        - geotransform: Affine transformation
        - sizeX, sizeY: Pixel resolution in X and Y
        - width, height: Raster dimensions in pixels
        - bands: Number of bands
        - bandnames: List of band names
        - scales: List of scale factors per band
        - offsets: List of offset values per band
        - nodata: No data value
        - dtype: Data type
        - tags: Dataset-level tags
    """

    with rasterio.open(infile) as src:
        proj = src.crs
        epsg = str(proj).split(":")[-1] if proj else None  # Safer EPSG extraction
        geotransform = src.transform
        sizeX, sizeY = src.res
        bounds = src.bounds
        bands = src.count
        width, height = src.width, src.height

        # Safely extract band names
        tags = src.tags()
        if "bands" in tags:
            try:
                # Try to safely evaluate the bands tag
                import ast

                bandnames = ast.literal_eval(tags["bands"])
            except (ValueError, SyntaxError):
                # If evaluation fails, treat as string or use default
                bandnames = tags["bands"] if isinstance(tags["bands"], list) else []
        else:
            bandnames = []

        scales = list(src.scales) if src.scales else [1.0] * bands
        offsets = list(src.offsets) if src.offsets else [0.0] * bands
        nodata = src.nodata
        dtype = str(src.dtypes[0]) if src.dtypes else None

    return {
        "epsg": epsg,
        "crs": proj,
        "bounds": bounds,
        "geotransform": geotransform,
        "sizeX": sizeX,
        "sizeY": sizeY,
        "width": width,
        "height": height,
        "bands": bands,
        "bandnames": bandnames,
        "scales": scales,
        "offsets": offsets,
        "nodata": nodata,
        "dtype": dtype,
        "tags": tags,
    }
