import numba
import numpy as np


@numba.jit(nopython=True, error_model='numpy', fastmath=True)
def interpolate_ts_linear(arr):
    out = np.zeros_like(arr)
    for band in range(arr.shape[0]):
        for py in range(arr.shape[2]):
            for px in range(arr.shape[3]):

                t = arr[band, :, py, px].copy()

                nans_ids = (t == 0)
                x_valid = np.where(~nans_ids)[0]
                y_valid = t[x_valid]
                x_invalid = np.where(nans_ids)[0]

                re_init = False
                if t[0] == 0:
                    t[0] = np.mean(y_valid)
                    re_init = True

                if t[-1] == 0:
                    t[-1] = np.mean(y_valid)
                    re_init = True

                if re_init:
                    nans_ids = (t == 0)
                    x_valid = np.where(~nans_ids)[0]
                    y_valid = t[x_valid]
                    x_invalid = np.where(nans_ids)[0]

                y_new = t
                m = len(x_invalid)
                for i in range(m):
                    j = np.searchsorted(
                        x_valid, x_invalid[i], side='right') - 1

                    q = (x_valid[j + 1] - x_valid[j])
                    s = (x_invalid[i] - x_valid[j]) / q

                    y_new[x_invalid[i]] = (
                        1 - s) * y_valid[j] + s * y_valid[j + 1]

                out[band, :, py, px] = y_new

    return out


def interpolate_ts(ts):

    newdata = interpolate_ts_linear(ts.data)
    out = ts.copy(data=newdata)
    return out
