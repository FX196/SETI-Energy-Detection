import numpy as np
from scipy import stats
from blimpy import Waterfall
from blimpy.utils import rebin
from matplotlib import pyplot as plt
from bisect import bisect_left
from tqdm import tqdm
import dask.array as da
import h5py
from time import time

from utils import *
import dask.array as da
import h5py
import sys
import os

fil_path = "data/filterbanks/"
h5_path = "data/h5/"

test_fil = fil_path + "blc20_guppi_57991_48899_3C161_0007.gpuspec.0000.fil"

fri_obs = h5_path + "GBT_57532_09539_HIP56445_fine.h5"

plt_args = {
    'aspect': 'auto',
    'origin': 'lower',
    'rasterized': True,
    'interpolation': 'nearest',
    'cmap': 'viridis'
}

from dask.distributed import Client


def to_npy_stack(source_h5_path, dest_path, verbose=False, channel_len=1033216):
    """
    Convert original h5 file to npy stack

    :param source_h5_path:
    :param dest_path:
    :param channel_len:
    :return:
    """
    if verbose:
        start = time()
    h5_file = h5py.File(source_h5_path, "r")
    arr = da.from_array(h5_file["data"], chunks=(2, 1, channel_len * 14))
    da.to_npy_stack(dest_path, arr, axis=2)
    if verbose:
        end = time()
        print("Converted to npy stack in %.4f seconds." % (end-start))


def remove_broadband(source_npy_path, dest_npy_path, verbose=False):
    client = Client(processes=False, threads_per_worker=3, n_workers=os.cpu_count()//3, memory_limit='8GB')

    if verbose:
        start = time()

    a = da.from_npy_stack(source_npy_path)
    means = da.mean(a, axis=2)
    means = da.reshape(means, (16,1,1))     # reshape to fit original data dimensions
    normalized_a = da.divide(a, means)      # divide by mean
    da.to_npy_stack(dest_npy_path, normalized_a, axis=2)    # write to npy stack

    if verbose:
        end = time()
        print("Converted to npy stack in %.4f seconds." % (end-start))

def remove_bandpass():
    pass


def gaussianity_thresholding():
    pass

if __name__ == "__main__":
    input_file, out_dir = sys.argv[1:3]
    to_npy_stack(input_file, out_dir, True)
    remove_broadband(out_dir, out_dir+"normalized", True)
