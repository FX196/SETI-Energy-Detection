import numpy as np
import pandas as pd
from scipy import stats
from blimpy import Waterfall
from blimpy.utils import rebin
from matplotlib import pyplot as plt
from bisect import bisect_left
from tqdm import tqdm
import dask.array as da
import h5py
from time import time
from multiprocessing import Pool, current_process
from dask.distributed import Client
import pickle

from utils import *
import dask.array as da
import sys
import os

# if "cupy" in sys.modules:
#     import cupy as np
#     print("Using cupy")

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


# Hyperparameters
coarse_channel_width=1048576
threshold = 1e-80
num_chans_per_block = 28


def to_npy_stack(source_h5_path, dest_path, verbose=False, channel_len=1048576):
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
    arr = da.from_array(h5_file["data"], chunks=(2, 1, channel_len * num_chans_per_block))
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)
    if not os.path.isdir(dest_path+"/original"):
        os.mkdir(dest_path+"/original")
    da.to_npy_stack(dest_path+"/original", arr, axis=2)
    if verbose:
        end = time()
        print("Converted to npy stack in %.4f seconds." % (end-start))


# def remove_broadband(source_npy_path, dest_npy_path, verbose=False):
#     client = Client(processes=True, threads_per_worker=3, n_workers=os.cpu_count()//3, memory_limit='8GB')
#
#     if verbose:
#         start = time()
#         print("Removing broadband signals")
#
#     a = da.from_npy_stack(source_npy_path)
#     means = da.mean(a, axis=2)
#     means = da.reshape(means, (16,1,1))     # reshape to fit original data dimensions
#     normalized_a = da.divide(a, means)      # divide by mean
#     da.to_npy_stack(dest_npy_path, normalized_a, axis=2)    # write to npy stack
#
#     if verbose:
#         end = time()
#         print("Removed broadband signals in %.4f seconds." % (end-start))
#     client.close()


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

if __name__ == "__main__":
    input_file = sys.argv[1]
    if len(sys.argv) == 2:
        out_dir = input_file[:-3]
    else:
        out_dir = sys.argv[2]
    header = read_header(input_file)
    n_chans = header["nchans"]
    i_vals = np.arange(n_chans)
    freqs = header["foff"] * i_vals + header["fch1"]
    block_width = num_chans_per_block * coarse_channel_width
    to_npy_stack(input_file, out_dir, True)
    with open(out_dir+"/header.pkl", "wb") as f:
        pickle.dump(header, f)
        print("Header saved to "+out_dir+"/header.pkl")
    # remove_broadband(out_dir+"/original", out_dir+"/normalized", True)

    source_npy_path = out_dir+"/original"
    block_files = [file for file in os.listdir(source_npy_path) if file.endswith(".npy")]
    cleaned_dir = out_dir+"/cleaned"

    if not os.path.isdir(cleaned_dir):
        os.mkdir(cleaned_dir)

    start = time()

    for block_file in tqdm(block_files):
        print("loading %s from %s" % (block_file, source_npy_path))
        block_data = np.load(source_npy_path+"/"+block_file)
        print("%s loaded, processing" % block_file)
        block_data = block_data[:, 0, :]
        half_chan = coarse_channel_width/2
        for i in range(num_chans_per_block):      # remove dc spike
            dc_ind = int(i*coarse_channel_width + half_chan)
            block_data[:, dc_ind] = (block_data[:, dc_ind+1] + block_data[:, dc_ind-3])/2
            block_data[:, dc_ind-1] = (block_data[:, dc_ind+2] + block_data[:, dc_ind-2])/2

        integrated = np.mean(block_data, axis=0)
        channels = np.reshape(integrated, (-1, coarse_channel_width))


        def clean(channel_ind):
            # print("%s processing channel %d of %s" % (current_process().name, channel_ind, block_file))
            cleaned_block =  remove_channel_bandpass(block_data[:, coarse_channel_width*(channel_ind):coarse_channel_width*(channel_ind+1)],
                           channels[channel_ind], coarse_channel_width)
            return cleaned_block / np.mean(cleaned_block, axis=1, keepdims=True)

        def normalize_block():
            with Pool(min(num_chans_per_block, os.cpu_count())) as p:
                cleaned = p.map(clean, range(num_chans_per_block))
            return cleaned
        normalized = normalize_block()
        normalized = np.concatenate(normalized, axis=1)
        np.save(out_dir+"/cleaned/" + block_file, normalized)

    end = time()
    print("Bandpass cleaned in %.4f seconds." % (end - start))

    import warnings
    warnings.filterwarnings("ignore")

    cleaned_block_files = [file for file in os.listdir(out_dir+"/cleaned") if file.endswith(".npy")]
    filtered_dir = out_dir+"/filtered/"
    if not os.path.isdir(filtered_dir):
        os.mkdir(filtered_dir)

    frame_list = []


    for block_file in tqdm(cleaned_block_files):
        print("Loading %s from %s" % (block_file, out_dir+"/cleaned"))
        data = np.load(out_dir+"/cleaned/" +block_file)
        print("Processing %s" % block_file)
        block_num = int(block_file.split(".")[0])
        if not os.path.isdir(filtered_dir+str(block_num)):
            os.mkdir(filtered_dir+str(block_num))
        def threshold_hits(channel_ind):
            res = list()
            window = data[:, coarse_channel_width*(channel_ind):coarse_channel_width*(channel_ind+1)]
            # window_f = freqs[coarse_channel_width*(chan):coarse_channel_width*(chan+1)]
            for i in range(0, (len(window[0])//200*200) - 100, 100):
                test_data = window[:, i:i+200]
                s, p = norm_test(test_data)
                if p < threshold:
                    res.append([coarse_channel_width*(channel_ind) + i, s, p])
            return res

        # def calc_stats(channel_ind):
        #     res = list()
        #     window = data[:, coarse_channel_width*(channel_ind):coarse_channel_width*(channel_ind+1)]
        #     # window_f = freqs[coarse_channel_width*(chan):coarse_channel_width*(chan+1)]
        #     for i in range(0, (len(window[0])//200*200), 100):
        #         test_data = window[:, i:i+200]
        #         s, p = norm_test(test_data)
        #         res.append([coarse_channel_width*(channel_ind) + i, s, p])
        #     return res

        start = time()
        with Pool(min(num_chans_per_block, os.cpu_count())) as p:
            chan_hits = p.map(threshold_hits, range(num_chans_per_block))
        end = time()
        print("%s Processed in %.4f seconds" %(block_file, end-start))

        vals_frame = pd.DataFrame(sum(chan_hits, []), columns=["index", "statistic", "pvalue"])
        vals_frame["block_num"] = block_num
        vals_frame["index"] += block_num*block_width
        vals_frame["freqs"] = vals_frame["index"].map(lambda x: freqs[x])
        frame_list.append(vals_frame)

        # print("Saving results")
        # def save_stamps(channel_ind):
        #     # print("%s processing channel %d of %s" % (current_process().name, channel_ind, block_file))
        #     for res in chan_hits[channel_ind]:
        #         i, s, p = res
        #         # plt.imsave((filtered_dir+"%d/%d.png" % (block_num, block_num*block_width + i)), data[:, i:i+200])
        #         np.save((filtered_dir+"%d/%d.npy" % (block_num, block_num*block_width + i)), data[:, i:i+200])
        # start = time()
        # with Pool(min(num_chans_per_block, os.cpu_count())) as p:
        #     p.map(save_stamps, range(num_chans_per_block))
        # end = time()
        # print("Results saved in %.4f seconds" % (end - start))

    full_df = pd.concat(frame_list, ignore_index=True)
    full_df.set_index("index")
    full_df.to_pickle(out_dir + "/info_df.pkl")
