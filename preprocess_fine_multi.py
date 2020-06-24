import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from bisect import bisect_left
from tqdm import tqdm
import h5py
from time import time
from multiprocessing import Pool, current_process
import pickle
import multiprocessing
from utils import *
import sys
import os
from functools import partial
# if "cupy" in sys.modules:
#     import cupy as np
#     print("Using cupy")

# Hyperparameters
coarse_channel_width=1048576
threshold = 1e-80
stat_threshold = 2048
parallel_coarse_chans = 28 # number of coarse channels operated on in parallel
num_blocks = 308 // parallel_coarse_chans
block_width = coarse_channel_width * parallel_coarse_chans
save_png = False
save_npy = True




def read_h5_block_sub(block_num, sub):
    #Each process that a core will read 
    #the data is indexed with the help of the routine function. 
    coarse_channel_width=1048576
    threshold = 1e-80
    stat_threshold = 2048
    parallel_coarse_chans = 28 
    num_blocks = 308 // parallel_coarse_chans
    #Creates a set of indexes that will be loaded by each core. 
    routine = create_routine(cores =7, block_num =block_num , coarse_channel_width = coarse_channel_width)
    block_width = coarse_channel_width * parallel_coarse_chans
    h5 = h5py.File("GBT_58010_50176_HIP61317_fine.h5",'r')
    return h5.get('data')[:, 0, routine[block_num][sub]:routine[block_num][sub]+(4*coarse_channel_width)]
    
    
def create_routine(cores, block_num, coarse_channel_width):
    # Creates a list of numpy index which are routines
    #that the cores will individually read
    cores = 7
    block_num = 11
    routine_temp = []
    routine= []
    for i in range(77): 
        routine_temp.append(i*4*coarse_channel_width)
    for k in range(block_num):
        temp = routine_temp[k*cores:(k+1)*cores]
        routine.append(temp)
    return routine

def multiprocess_reading(block_num):
    # Wrapped all the multiprocessing functions in one big function
    a_pool = multiprocessing.Pool()
    core = 7
    sub = [range(core)]
    func = partial(read_h5_block_sub,block_num)
    data = a_pool.map(func, range(core))
    a_pool.close()
    a_pool.join()
    return data

def re_shape_input(data):
    #Reshapes the stacked 3d Tensor and spreads it to a 2d matrix 
    result_data = np.zeros((data.shape[1],data.shape[0]*data.shape[2] ))
    for i in range(data.shape[0]):
        result_data[:,i*data.shape[2]:data.shape[2]*(i+1)]=data[i,:,:]
    return result_data


if __name__ == "__main__":
    g_start = time()
    input_file = sys.argv[1]
    if len(sys.argv) == 2:
        out_dir = input_file.split(".")[0]
    else:
        out_dir = sys.argv[2]

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # read and store the header
    header = read_header(input_file)
    n_chans = header["nchans"]
    i_vals = np.arange(n_chans)
    freqs = header["foff"] * i_vals + header["fch1"]
    with open(out_dir+"/header.pkl", "wb") as f:
        pickle.dump(header, f)
        print("Header saved to "+out_dir+"/header.pkl")

    hf = h5py.File(input_file, "r")

    frame_list = []
    stack_list = []

    for block_num in tqdm(range(num_blocks)):
        print(f"Processing coarse channels {block_num * parallel_coarse_chans}-{(block_num + 1) * parallel_coarse_chans}")
        start = time()
        #Replaced the original read with a multiprocess_reading
        #I had wrapped it around multiple functions to make it one line
        print("Reading in a chunk of data...")
        block_data = re_shape_input(np.array(multiprocess_reading(block_num)))
        end = time()
        print(f"Data loaded in {end - start:.4f} seconds, processing")

        start = time()
        half_chan = coarse_channel_width/2
        for i in range(parallel_coarse_chans):      # remove dc spike
            dc_ind = int(i*coarse_channel_width + half_chan)
            block_data[:, dc_ind] = (block_data[:, dc_ind+1] + block_data[:, dc_ind-3])/2
            block_data[:, dc_ind-1] = (block_data[:, dc_ind+2] + block_data[:, dc_ind-2])/2

        integrated = np.mean(block_data, axis=0)
        channels = np.reshape(integrated, (-1, coarse_channel_width))


        def clean(channel_ind):
            # print("%s processing channel %d of %s" % (current_process().name, channel_ind, block_file))
            cleaned_block =  remove_channel_bandpass(block_data[:, coarse_channel_width*(channel_ind):coarse_channel_width*(channel_ind+1)],
                           channels[channel_ind], coarse_channel_width)
            return cleaned_block

        def clean_block_bandpass():
            with Pool(min(parallel_coarse_chans, os.cpu_count())) as p:
                cleaned = p.map(clean, range(parallel_coarse_chans))
            return cleaned


        cleaned_block_data = clean_block_bandpass()
        cleaned_block_data = np.concatenate(cleaned_block_data, axis=1)
        # np.save(out_dir+"/cleaned/" + block_file, normalized)

        end = time()
        print("Bandpass cleaned in %.4f seconds." % (end - start))

        del block_data

        # actual energy detection
        def threshold_hits(channel_ind):
            res = list()
            channel_data = cleaned_block_data[:, coarse_channel_width*(channel_ind):coarse_channel_width*(channel_ind+1)]
            for i in range(0, coarse_channel_width - 128, 128):
                test_window = channel_data[:, i:i+256]
                s, p = norm_test(test_window)
                if s > stat_threshold:
                    res.append([coarse_channel_width*(channel_ind) + i, s, p])
            return res

        start = time()
        with Pool(min(parallel_coarse_chans, os.cpu_count())) as p:
            chan_hits = p.map(threshold_hits, range(parallel_coarse_chans))
        end = time()
        print("Stamps filtered in %.4f seconds" %(end-start))

        vals_frame = pd.DataFrame(sum(chan_hits, []), columns=["index", "statistic", "pvalue"])
        vals_frame["index"] += block_num*block_width
        vals_frame["freqs"] = vals_frame["index"].map(lambda x: freqs[x])
        frame_list.append(vals_frame)

        print("Saving results")
        def aggregate_npy(channel_ind):
            inds = map(lambda x: x[0], chan_hits[channel_ind])
            return np.array([cleaned_block_data[:, ind:ind+256] for ind in inds])

        start = time()
        with Pool(min(parallel_coarse_chans, os.cpu_count())) as p:
            # if save_png:
            #     p.map(save_stamps, range(parallel_coarse_chans))
            if save_npy:
                stack = p.map(aggregate_npy, range(parallel_coarse_chans))
                stack = [e for e in stack if e.size != 0]
                if stack:
                    stack_list.append(np.concatenate(stack, axis=0))
        end = time()
        print("Results aggregated in %.4f seconds" % (end - start))
        del integrated
        del channels
        del cleaned_block_data

    full_df = pd.concat(frame_list, ignore_index=True)
    full_df.set_index("index")
    full_df.to_pickle(out_dir + "/info_df.pkl")

    if stack_list:
        full_stack = np.concatenate(stack_list)
        np.save(out_dir + "/filtered.npy", full_stack)

    g_end = time()
    print("Finished Energy Detection on %s in %.4f seconds" % (os.path.basename(input_file), g_end - g_start))
