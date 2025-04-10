import numpy as np


def mask_ts(ts, mask):

    new_mask = np.broadcast_to(mask.data, ts.shape)
    ts_masked = ts.where(new_mask, np.nan)
    return ts.copy(data=ts_masked)
