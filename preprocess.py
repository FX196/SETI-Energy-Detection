import dask.array as da
import h5py

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

client = Client(processes=False, threads_per_worker=3,
                n_workers=4, memory_limit='4GB')


def to_npy_stack(source_h5_path, dest_path, channel_len=1033216):
    """
    Convert original h5 file to npy stack

    :param source_h5_path:
    :param dest_path:
    :param channel_len:
    :return:
    """
    h5_file = h5py.File(source_h5_path, "r")
    arr = da.from_array(h5_file["data"], chunks=(2, 1, channel_len * 14))
    da.to_npy_stack(dest_path, arr, axis=2)


def remove_broadband(source_h5_path):
    pass

def remove_bandpass():
    pass


def gaussianity_thresholding():
    pass
