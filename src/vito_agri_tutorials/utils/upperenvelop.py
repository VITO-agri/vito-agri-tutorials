from openeo.udf import XarrayDataCube
from openeo.udf.debug import inspect
from typing import Dict
import xarray as xr
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# from numpy.typing import NDArray
# from typing import List, Union

NPTPERYEAR = 36
FORCEUPPERENVELOPE = True
LASTITERATIONLIKETIMESATFIT = True
SPIKECUTOFF = 0.5
WINDOW = [3, 5, 7, 9]


def create_mask_outliers(time_series, n=SPIKECUTOFF):
    """
    Creates a mask for ignoring outlier values in a time series.

    Parameters:
        time_series (array-like): The input time series data.
        n (float): The number of standard deviations from the mean to define an outlier.

    Returns:
        numpy.ndarray: A boolean mask where True indicates values to keep, and False indicates outliers.
    """
    # Convert to numpy array if not already
    time_series = np.array(time_series)
    time_series_no_zeroes = time_series[time_series != 0]

    # Calculate standard deviation (ignoring 0 values)
    std_dev = np.std(time_series_no_zeroes)
    dist = n * std_dev

    # Create a mask for values within n standard deviations of the sliding median
    weights = np.where(time_series == 0, 0, 1)
    swinmax = 4
    nb = len(time_series) - 2 * swinmax
    for i in range(swinmax, nb + swinmax):
        m1 = i - swinmax
        m2 = i + swinmax + 1

        idx_weights_nonzero = weights[m1:m2].nonzero()
        index = m1 + idx_weights_nonzero[0]
        med = np.median(time_series[index])
        if abs(time_series[i] - med) >= dist and (
            (
                time_series[i]
                < (float(time_series[i - 1]) + float(time_series[i + 1])) / 2 - dist
            )
            or (time_series[i] > max(time_series[i - 1], time_series[i + 1]) + dist)
        ):
            weights[i] = 0

    mask = np.where(weights == 1, True, False)
    return mask


def iterative_masked_savgol_filter(
    time_series, window_lengths, polyorder=2, upper_envelope_filtering=False
):
    """
    Iteratively applies a Savitzky-Golay filter to a masked time series,
    ignoring data points with a value of 0 (interpolating over them).

    Parameters:
        time_series (array-like): The input time series data.
        window_lengths (array-like): A list of window lengths for iterative filtering.
        polyorder (int): The order of the polynomial to fit.
        mask (array-like): mask for data points, value of 0 to ignore them and 1 to take them into account.
        upper_envelope_filtering (bool): Whether to retain the maximum of the fit or the original data at each step.

    Returns:
        numpy.ndarray: The final smoothed time series.
    """
    # Convert inputs to numpy arrays
    time_series = np.array(time_series, dtype=int)
    mask = create_mask_outliers(time_series, n=SPIKECUTOFF)
    time_series = time_series * mask
    # if len(time_series) != len(mask):
    #     raise ValueError("The time series and mask must have the same length.")

    # Interpolate over data points with value 0
    valid_indices = time_series != 0
    if valid_indices.sum() < 2:
        return np.zeros(len(time_series))

    # Create an interpolated version of the time series where 0s are replaced
    interp_func = interp1d(
        np.flatnonzero(valid_indices),
        time_series[valid_indices],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_series = interp_func(np.arange(len(time_series)))

    smoothed_series = interpolated_series.copy()  # Start with the interpolated series

    while window_lengths[-1] > len((time_series - 1) / 2):
        window_lengths.pop(-1)

    for window_length in window_lengths:
        # Perform Savitzky-Golay filtering on the series
        try:
            next_smoothed = savgol_filter(smoothed_series, window_length, polyorder)
        # If Savitzky-Golay filtering not possible, use interpolated value
        except:
            next_smoothed = smoothed_series
        # If upper envelope filtering is enabled, retain the maximum of smoothed or original (interpolated) data
        if upper_envelope_filtering and window_length != window_lengths[-1]:
            smoothed_series = np.maximum(next_smoothed, smoothed_series)
        else:
            smoothed_series = next_smoothed
    return smoothed_series.astype(np.uint8)


def isSeqValid(arr: np.ndarray, limit: int) -> bool:
    """Look for runs consecutive zeros. Return False if any of the
    runs is longer than the limit. True otherwise

    Args:
        arr (np.ndarray): timeseries data (1D)
        limit (int): max allowed length of zeros

    Returns:
        bool: False if at least one sequence of zeros is longer than limit,
        True if none of the zero sequences exceeds the limit
    """
    mask_ = np.concatenate(([False], np.r_[np.diff(arr) == 0, False], [False]))
    idx = np.flatnonzero(mask_[1:] != mask_[:-1])
    ivlist = [
        arr[idx[i] : idx[i + 1] + 1] for i in range(0, len(idx), 2) if arr[idx[i]] == 0
    ]
    return len([a for a in ivlist if len(a) > limit]) == 0


def upper_envelop(series: np.ndarray) -> np.ndarray:
    """Perform upper-envelope cleanup on a 1D timeseries.
    The cleanup uses a filter window, and therefore the beginning and end
    of the timeseries can not be cleaned properly. To counter this extend
    the timeseries with additional data at end and begin. The amount is equal
    to the window size of the filter

    The series is expected to have values in the range 0-255.
    The type is forced to byte (numpy.uint8)

    Args:
        series (np.ndarray): 1D-array containing timeseries data

    Returns:
        np.ndarray: cleaned timeseries. The lenght of the timeseries is
        identical to the lenght of the input series
    """
    if series.dtype != np.uint8:
        series = series.astype(np.uint8)
    (nb,) = series.shape
    w = np.ones(nb)
    invalid = series < 2
    w[invalid] = 0
    # indicate missing data if less than 25% of the data is valid
    missingdata = np.count_nonzero(w) < np.floor(nb / 4)
    # find longest run of invalid data. if longer than maximum length
    # allowed (in allow_run) then indicate series as missing data
    allow_run = int(np.floor(NPTPERYEAR / 3))
    missingdata = missingdata or (not isSeqValid(w, allow_run))
    if missingdata:
        return np.zeros((nb,))

    filtered = iterative_masked_savgol_filter(
        series, window_lengths=WINDOW, polyorder=2, upper_envelope_filtering=True
    )
    return filtered


def upper_envelop_cube(cube: xr.DataArray) -> xr.DataArray:
    """Apply the upper-envelop on the entire datacube

    Args:
        cube (xarray.DataArray): 3d-cube with timeseries data

    Returns:
        xarray.DataArray: cleanup timeseries data
    """
    return xr.DataArray(np.apply_along_axis(upper_envelop, 0, cube))


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    """OpenEO entry point
        OpenEO signature function to run upperenvelop as UDF
        Dimensions and coordinates have not changed, but are
        lost during the calculation, so are reattached
    @param cube: Openeo provided data cube (timeseries)
    @param context: user provided data (parameters)
    @return: Corrected datacube
    """
    inspect(message="Start upper envelope filtering")
    indata = cube.get_array()
    dims = indata.dims
    coords = indata.coords
    calc = upper_envelop_cube(indata)
    return XarrayDataCube(
        xr.DataArray(
            calc, dims=dims, coords=coords, attrs=indata.attrs, name=indata.name
        )
    )
