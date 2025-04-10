import scipy
import numpy as np
import xarray as xr
import pandas as pd


def tsteps(x, n_steps=6):
    return scipy.signal.resample(x, n_steps, axis=0)


def percentile_iqr(x, q=[10, 50, 90], iqr=[25, 75]):

    q_all = list(set(q + iqr))
    q_ids = [q_all.index(qv) for qv in q]
    iqr_ids = [q_all.index(qv) for qv in iqr]

    perc = np.percentile(x, q=q_all, axis=0)

    perc_arr = perc[q_ids]
    iqr_arr = perc[iqr_ids[1]] - perc[iqr_ids[0]]

    iqr_arr = np.expand_dims(iqr_arr, axis=0)

    arr = np.concatenate([perc_arr, iqr_arr], axis=0)

    return arr

def band_percentile(inarr: xr.DataArray, quantiles: dict) -> xr.DataArray:
    """Computes the band percentiles and return a array containing for each
    pixel, the specified percentiles value for a band.
    
    Parameters
    ----------
    inarr: xr.DataArray
        Input array with expected dimensions 'variables', 'timestamp', 'y', 'x'
        or 'variable', 'timestamp', 'sample'
    band_mapping: dict
        A dictionnary containing for each band name the percentiles values to
        compute. For example:
        ```
        {
            'B02': [0.1, 0.5, 0.9],
            'ndvi': [0.1, 0.25, 0.5, 0.75, 0.9]
        }
        ```
    Returns
    -------
    output: xr.DataArray
        A xarray.DataArray that has the 'variable' and 'timestamp' dimensions
        reduced into a 'features' dimension containing the quantiles.
    """
    all_quantiles = []
    for variable, quantile_values in quantiles.items():
        quantile_array = inarr.sel(variable=[variable]).quantile(
            quantile_values, dim='timestamp'
        )
        quantile_names = [f'{variable}-{val}' for val in quantile_values]

        quantile_array = quantile_array.stack(
            features=('variable', 'quantile')
        ).drop(['variable', 'quantile']).assign_coords(
            {'features': quantile_names}
        )

        all_quantiles.append(quantile_array)

    return xr.concat(all_quantiles, dim='features')
