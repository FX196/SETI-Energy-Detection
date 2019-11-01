from blimpy.utils import rebin
from matplotlib import pyplot as plt

MAX_PLT_POINTS      = 65536                  # Max number of points in matplotlib plot
MAX_IMSHOW_POINTS   = (8192, 4096)           # Max number of points in imshow plot

plt_args = {
            'aspect':'auto',
            'origin':'lower',
            'rasterized':True,
            'interpolation':'nearest',
            'cmap':'viridis'
            }

def norm_test(arr):
    return stats.normaltest(arr.flatten())

def show_stamp(window, i):
    test_data = window[:, i:i+200]
    plt.figure()
    plt.imshow(test_data, **plt_args)
    
def show_stamp_f(freqs, data, f):
    ind = bisect_left(freqs, f)
    test_data = data[:, ind:ind+200]
    plt.figure()
    plt.imshow(test_data, **plt_args)
    
def plot_segment(plot_data):
    dec_fac_x, dec_fac_y = 1, 1
    if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:
        dec_fac_x = int(plot_data.shape[0] / MAX_IMSHOW_POINTS[0])

    if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:
        dec_fac_y = int(plot_data.shape[1] / MAX_IMSHOW_POINTS[1])

    print('Downsampling by a factor of (%d, %d)' %(dec_fac_x, dec_fac_y))
    plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)
    plt.figure(figsize=(10, 6))
    plt.imshow(plot_data, **plt_args)
