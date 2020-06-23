from matplotlib import pyplot as plt
import numpy as np
from scipy import stats, interpolate
import h5py

def read_header(filename):
    header = {}
    h5_file = h5py.File(filename, "r")
    for key, val in h5_file['data'].attrs.items():
        header[key] = val
    h5_file.close()
    return header

def norm_test(arr):
    return stats.normaltest(arr.flatten())

def fit_channel_bandpass(channel, integrated_channel, channel_width=1033216, spl_order=16):
    x = np.arange(channel_width)
    knots = np.arange(0, channel_width, channel_width//spl_order+1)
    spl = interpolate.splrep(x, integrated_channel, t=knots[1:])
    chan_fit = interpolate.splev(x, spl)
    return chan_fit

def remove_channel_bandpass(channel, integrated_channel, channel_width=1033216, spl_order=16):
    fit = fit_channel_bandpass(channel, integrated_channel, channel_width, spl_order)
    return channel - fit
