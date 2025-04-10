import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
from numba import njit, float32, int64, int16, float64
from numba.types import Tuple


@njit(int64[:](float32[:]))
def find_peaks(x):

    peak_index = []

    for i, val in enumerate(x[1:-1], 1):

        if val >= x[i-1] and val > x[i+1]:
            peak_index.append(i)

    if x[-1] > x[-2]:
        peak_index.append(len(x) - 1)

    if x[0] > x[1]:
        peak_index.append(0)

    return np.array(peak_index)


def time_converter(x):
    if x == -1:
        return np.nan
    else:
        return pd.to_datetime(x, unit='s')


def nearest_date(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def detect_seasons(data, times,
                   max_seasons=5,
                   amp_thr1=0.1,
                   amp_thr2=0.35,
                   min_window=10,
                   max_window=185,
                   partial_start=False,
                   partial_end=False):
    '''
    Computes peak, SOS and EOS of all seasons from a given timeseries

    SOS, MOS and EOS are stored as datetime.timestamp()

    :param data: a 3D numpy data array (time, x, y)
        from which to derive the seasons (should only contain one band!).
        The data cannot contain any NaN's!
    :param times: a 1D data array containing the timestamps of the
        input time series. 
    :param max_seasons: maximum number of seasons to be detected
    :param amp_thr1: minimum threshold for amplitude of a season
    :param amp_thr2: factor with which to multiply total
        amplitude of signal to derive second minimum amplitude threshold
    :param min_window: search window for finding start and end
        before/after peak - minimum distance from peak in days
    :param max_window: search window for finding start and end
        before/after peak - maximum distance from peak in days
    :param partial_start and partial_end: whether or not to
        detect partial seasons
        (if partial_start = True, season is allowed to have
        SOS prior to start of time series;
        if partial_end = True, season is allowed to have
        EOS after end of time series)

    returns a collection of arrays: SOS, MOS, EOS and nSeasons for each pixel
    '''

    if data.ndim != 3:
        raise ValueError('Timeseries as input for detect_seasons '
                         'should be 3-dimensional!')

    # convert datetime objects to floats
    times = np.array([pd.Timestamp(t).timestamp() for t in times])

    ########## definition of function actually doing the work ##########
    @njit(Tuple((int16[:, :], float32[:, :, :],
                 float32[:, :, :], float32[:, :, :]))(
                     float32[:, :, :], float64[:]))
    def _detect_seasons_fast(data, times):

        def _day_to_second(days):
            return days * 24 * 3600

        def _find_nearest(array, value):
            array = np.asarray(array)
            return np.abs(array - value).argmin()

        # prepare outputs
        nx = data.shape[1]
        ny = data.shape[2]

        nseasons = np.zeros((nx, ny), dtype=np.int16)
        sos = np.zeros((max_seasons, nx, ny), dtype=np.float32)
        mos = np.zeros((max_seasons, nx, ny), dtype=np.float32)
        eos = np.zeros((max_seasons, nx, ny), dtype=np.float32)

        sos[...] = np.nan
        mos[...] = np.nan
        eos[...] = np.nan

        # loop over each individual pixel to define
        # start, peak and end of season(s)

        for i in range(nx):
            for j in range(ny):
                data_pix = data[:, i, j]
                # find all local maxima
                localmax_idx = find_peaks(data_pix)
                if localmax_idx.size == 0:
                    # no peaks found, proceed to next pixel
                    continue
                localmax = data_pix[localmax_idx]

                # sort local maxima according to VI amplitude
                sort_idx = localmax.argsort()
                localmax_idx_sorted = localmax_idx[sort_idx]

                # define outputs
                npeaks = localmax_idx_sorted.shape[0]
                valid = np.ones(npeaks, dtype=np.uint8)
                start = np.zeros(npeaks, dtype=np.int32)
                end = np.zeros(npeaks, dtype=np.int32)

                # setting some threshold values
                totalrange = np.max(data_pix) - np.min(data_pix)
                amp_thr2_fin = amp_thr2 * totalrange

                # find for each peak the associated local minima
                # and decide whether
                # the peak is valid or not
                for p in range(npeaks):

                    skip_sos = False

                    idx = localmax_idx_sorted[p]
                    # define search window for SOS
                    t_idx = times[idx]
                    t_min = t_idx - _day_to_second(max_window)
                    idx_min = _find_nearest(times, t_min)
                    t_max = t_idx - _day_to_second(min_window)
                    idx_max = _find_nearest(times, t_max)

                    # if peak is very close to start of TS...
                    if idx_max == 0:
                        if partial_start:
                            # and partial season mapping is allowed
                            # -> skip SOS detection
                            skip_sos = True
                        else:
                            # else, peak is invalid
                            valid[p] = 0
                            continue

                    # do SOS check if necessary
                    if not skip_sos:
                        # adjust search window in case there is a valid
                        # peak within the window
                        # find all intermediate VALID peaks
                        val_peaks = localmax_idx_sorted.copy()
                        val_peaks[valid == 0] = -1
                        int_peaks_idx = localmax_idx_sorted[(
                            val_peaks > idx_min) & (val_peaks < idx)]
                        # if any, find the peak nearest to original peak
                        # and set t_min to that value
                        if int_peaks_idx.shape[0] > 0:
                            idx_min = np.max(int_peaks_idx)
                            # if, by adjusting the window, idx_max <
                            # idx_min -> label peak as invalid
                            if idx_max < idx_min:
                                valid[p] = 0
                                continue

                        # identify index of local minimum in search window
                        win = data_pix[idx_min:idx_max+1]
                        start[p] = np.where(win == np.amin(win))[
                            0][0] + idx_min

                        # check if amplitude conditions of the identified
                        # starting point are met
                        amp_dif = data_pix[idx] - data_pix[start[p]]
                        if not (amp_dif >= amp_thr1) & (amp_dif >= amp_thr2_fin):
                            # if partial season mapping is allowed,
                            # and search window includes start of TS,
                            # the true SOS could be before start of TS.
                            # So we skip sos check, meaning eos check
                            # should definitely be done
                            if partial_start and (idx_min == 0):
                                skip_sos = True
                            else:
                                valid[p] = 0
                                continue

                    # define search window for EOS
                    t_min = t_idx + _day_to_second(min_window)
                    idx_min = _find_nearest(times, t_min)
                    t_max = t_idx + _day_to_second(max_window)
                    idx_max = _find_nearest(times, t_max)
                    # adjust search window in case there is a valid
                    # peak within the window
                    # find all intermediate VALID peaks
                    val_peaks = localmax_idx_sorted.copy()
                    val_peaks[valid == 0] = -1
                    int_peaks_idx = localmax_idx_sorted[(val_peaks > idx) & (
                        val_peaks < idx_max)]
                    # if any, find the peak nearest to original peak
                    # and set t_max to that value
                    if int_peaks_idx.shape[0] > 0:
                        idx_max = np.min(int_peaks_idx)
                        # if, by adjusting the window, idx_max
                        # < idx_min -> label peak as invalid
                        if idx_max < idx_min:
                            valid[p] = 0
                            continue

                    # in case you've reached the end of the timeseries,
                    # adjust idx_max
                    if idx_max == data_pix.shape[0] - 1:
                        idx_max -= 1
                    # identify index of local minimum in search window
                    if idx_max < idx_min:
                        end[p] = data_pix.shape[0] - 1
                    else:
                        win = data_pix[idx_min:idx_max+1]
                        end[p] = np.where(win == np.amin(win))[0][0] + idx_min

                    # if partial season mapping is allowed
                    # AND sos check was not skipped
                    # AND search window includes end of TS
                    # THEN the end of season check can be skipped

                    if (partial_end and (not skip_sos) and
                            (idx_max == data_pix.shape[0] - 2)):
                        continue
                    else:
                        # check if amplitude conditions of the identified
                        # end point are met
                        amp_dif = data_pix[idx] - data_pix[end[p]]
                        if not (amp_dif >= amp_thr1) & (
                                amp_dif >= amp_thr2_fin):
                            valid[p] = 0

                # now delete invalid peaks
                idx_valid = np.where(valid == 1)[0]
                peaks = localmax_idx_sorted[idx_valid]
                start = start[valid == 1]
                end = end[valid == 1]
                npeaks = peaks.shape[0]

                # if more than max_seasons seasons detected ->
                # select the seasons with highest amplitudes
                if npeaks > max_seasons:
                    toRemove = npeaks - max_seasons
                    maxSeason = data_pix[peaks]

                    baseSeason = np.mean(np.stack((data_pix[start],
                                                   data_pix[end])))
                    amp = maxSeason - baseSeason
                    idx_remove = np.zeros_like(amp)
                    for r in range(toRemove):
                        idx_remove[np.where(amp == np.min(amp))[0][0]] = 1
                        amp[np.where(amp == np.min(amp))[0][0]] = np.max(amp)
                    # check whether enough seasons will be removed
                    check = toRemove - np.sum(idx_remove)
                    if check > 0:
                        # remove random seasons
                        for r in range(int(check)):
                            idx_remove[np.where(idx_remove == 0)[0][0]] = 1
                    # remove the identified peaks
                    peaks = peaks[idx_remove != 1]
                    start = start[idx_remove != 1]
                    end = end[idx_remove != 1]
                    npeaks = max_seasons

                # convert indices to actual corresponding dates
                peaktimes = times[peaks]
                starttimes = times[start]
                endtimes = times[end]

                # if less than max_seasons seasons detected -> add
                # dummy seasons
                if peaktimes.shape[0] < max_seasons:
                    toAdd = np.ones(max_seasons - peaktimes.shape[0],
                                    dtype=np.float32) * -1
                    starttimes = np.concatenate((starttimes, toAdd))
                    endtimes = np.concatenate((endtimes, toAdd))
                    peaktimes = np.concatenate((peaktimes, toAdd))

                # transfer to output
                mos[:, i, j] = peaktimes
                sos[:, i, j] = starttimes
                eos[:, i, j] = endtimes
                nseasons[i, j] = npeaks
        return nseasons, sos, mos, eos

    ########## end of function definition ##########

    (nseasons, sos, mos, eos) = _detect_seasons_fast(data, times)

    # convert sos, mos and eos back to datetime objects
    cfunc = np.vectorize(time_converter)
    sos = cfunc(sos)
    mos = cfunc(mos)
    eos = cfunc(eos)

    return nseasons, sos, mos, eos


def visualize_seasons(nseasons, sos, mos, eos,
                      x, y, ts, times, outfile=None):
    '''
    Plot seasons for a given pixel x,y

    :param nseasons (2D array): number of seasons per pixel
        (first output from detect_seasons function)
    :param sos (3D array): start of each season per pixel 
        (second output from detect_seasons function)
    :param mos (3D array): peak of each season per pixel 
        (third output from detect_seasons function)
    :param eos (3D array): end of each season per pixel 
        (last output from detect_seasons function)
    :param x: row index of pixel to be plotted
    :param y: column index of pixel to be plotted
    :param ts: timeseries to be used for plotting (timeseries
        which was used as input for season detection)
    :param times: timestamps matching the timeseries used as input
        for season detection
    '''

    if ts.ndim != 3:
        raise ValueError('Timeseries as input for detect_seasons'
                         'should only contain one band!')

    # plot timeseries
    f, ax = plt.subplots()
    ts = np.squeeze(ts[:, x, y])
    ax.plot(times, ts)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Growing seasons for pixel ({x}, {y})')

    # plot all seasons for particular pixel
    npeaks = nseasons[x, y]
    for p in range(npeaks):
        startdate = sos[p, x, y]
        startidx = np.where(times == nearest_date(times, startdate))[0][0]
        peakdate = mos[p, x, y]
        peakidx = np.where(times == nearest_date(times, peakdate))[0][0]
        enddate = eos[p, x, y]
        endidx = np.where(times == nearest_date(times, enddate))[0][0]
        ax.plot(times[startidx], ts[startidx], 'go')
        ax.plot(times[peakidx], ts[peakidx], 'k+')
        ax.plot(times[endidx], ts[endidx], 'ro')

    plt.show()

    # save resulting plot if requested
    if outfile is not None:
        plt.savefig(outfile)
