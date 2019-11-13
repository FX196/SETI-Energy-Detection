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
        print("Converting to npy stack")
    h5_file = h5py.File(source_h5_path, "r")
    arr = da.from_array(h5_file["data"], chunks=(2, 1, channel_len * 14))
    if not os.path.isdir(dest_path+"/original"):
        os.mkdir(dest_path)
        os.mkdir(dest_path+"/original")
    da.to_npy_stack(dest_path+"/original", arr, axis=2)
    if verbose:
        end = time()
        print("Converted to npy stack in %.4f seconds." % (end-start))


def remove_broadband(source_npy_path, dest_npy_path, verbose=False):
    client = Client(processes=True, threads_per_worker=3, n_workers=os.cpu_count()//3, memory_limit='8GB')

    if verbose:
        start = time()
        print("Removing broadband signals")

    a = da.from_npy_stack(source_npy_path)
    means = da.mean(a, axis=2)
    means = da.reshape(means, (16,1,1))     # reshape to fit original data dimensions
    normalized_a = da.divide(a, means)      # divide by mean
    da.to_npy_stack(dest_npy_path, normalized_a, axis=2)    # write to npy stack

    if verbose:
        end = time()
        print("Removed broadband signals in %.4f seconds." % (end-start))


# def remove_bandpass(source_npy_path, coarse_channel_width=1033216):
#     # source_npy_path is the directory containing the npy stack
#     block_files = os.listdir(source_npy_path)
#
#     for block_file in block_files:
#         print("loading %s from %s" % (block_file, source_npy_path))
#         block_data = np.load(source_npy_path+"/"+block_file)
#         block_data = block_data[:, 0, :]
#         integrated = np.mean(block_data, axis=0)
#         for n in np.nonzero(integrated > 800):                          # remove DC spike
#             integrated[n] = (integrated[n-1] + integrated[n+1]) /2
#         channels = np.reshape(integrated, (-1, coarse_channel_width))
#
#
#         from multiprocessing import Pool, current_process
#         def clean(channel_ind):
#             print("%s processing channel %d of block %d" % (current_process().name, channel_ind, block_file))
#             return remove_channel_bandpass(block_data[:, coarse_channel_width*(channel_ind):coarse_channel_width*(channel_ind+1)],
#                            channels[channel_ind], coarse_channel_width)
#
#         def normalize_block():
#             with Pool(min(14, os.cpu_count())) as p:
#                 cleaned = p.map(clean, range(14))
#             return cleaned
#         normalized = normalize_block()
#         normalized = np.concatenate(normalized, axis=1)
#         np.save(source_npy_path+"/"+block_file.split(".")[0]+"_cleaned.npy", normalized)


def gaussianity_thresholding():
    pass

if __name__ == "__main__":
    input_file, out_dir = sys.argv[1:3]
    to_npy_stack(input_file, out_dir, True)
    remove_broadband(out_dir+"/original", out_dir+"/normalized", True)
    # remove_bandpass(out_dir+"_normalized")

    source_npy_path = out_dir+"/normalized"
    block_files = [file for file in os.listdir(out_dir+"/normalized") if file.endswith(".npy")]
    coarse_channel_width=1033216

    if not os.path.isdir(out_dir+"/cleaned"):
        os.mkdir(out_dir+"/cleaned")

    for block_file in tqdm(block_files):
        print("loading %s from %s" % (block_file, source_npy_path))
        block_data = np.load(source_npy_path+"/"+block_file)
        block_data = block_data[:, 0, :]
        integrated = np.mean(block_data, axis=0)
        for n in np.nonzero(integrated > 800):                          # remove DC spike
            integrated[n] = (integrated[n-1] + integrated[n+1]) /2
        channels = np.reshape(integrated, (-1, coarse_channel_width))


        from multiprocessing import Pool, current_process
        def clean(channel_ind):
            print("%s processing channel %d of %s" % (current_process().name, channel_ind, block_file))
            return remove_channel_bandpass(block_data[:, coarse_channel_width*(channel_ind):coarse_channel_width*(channel_ind+1)],
                           channels[channel_ind], coarse_channel_width)

        def normalize_block():
            with Pool(12) as p:
                cleaned = p.map(clean, range(14))
            return cleaned
        normalized = normalize_block()
        normalized = np.concatenate(normalized, axis=1)
        np.save(out_dir+"/cleaned/" + block_file, normalized)

    # import warnings
    # warnings.filterwarnings("ignore")
    #
    # cleaned_block_files = []
    #
    #
    # def threshold_hits(chan):
    #     res = list()
    #     window = data[:, channel_len*(chan):channel_len*(chan+1)]
    #     # window_f = freqs[channel_len*(chan):channel_len*(chan+1)]
    #     for i in range(0, (len(window[0])//200*200), 100):
    #         test_data = window[:, i:i+200]
    #         s, p = norm_test(test_data)
    #         if p < 1e-25:
    #             res.append([channel_len*(chan) + i, s, p])
    #     return res
    #
    # start = time()
    # with Pool(12) as p:
    #     chan_hits = p.map(threshold_hits, range(14))
    # end = time()
    # print(end-start)
