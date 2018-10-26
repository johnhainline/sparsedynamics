
from skimage.io import imread, imsave
import numpy as np
import os
import pathlib

from load_img.baseimage import PETImage, CTImage, normalize
from skimage.restoration import (denoise_wavelet, estimate_sigma)


# put filepath to image to open
filepath = os.path.join("data", "mpet3715b_em1_v1.pet.img")

# load image into memory
my_img = PETImage(filepath=filepath)
my_img.load_image()


def save(filename, data):
    sum_over_time = np.sum(data, axis=3)

    # ensure the outputs directory exists
    outputs_path = pathlib.Path('outputs/')
    outputs_path.mkdir(parents=True, exist_ok=True)

    xy_axis = np.sum(sum_over_time, axis=2)
    imsave(pathlib.Path('outputs/xy_' + filename + '.png'), xy_axis / xy_axis.max())
    xz_axis = np.sum(sum_over_time, axis=1)
    imsave(pathlib.Path('outputs/xz_' + filename + '.png'), xz_axis / xz_axis.max())
    yz_axis = np.sum(sum_over_time, axis=0)
    imsave(pathlib.Path('outputs/yz_' + filename + '.png'), yz_axis / yz_axis.max())


# data is [x,y,z,t] array of float64
data = np.swapaxes(my_img.img_data, 0, 2)
my_img.unload_image()

save("original", data)

# See http://scikit-image.org/docs/dev/auto_examples/filters/plot_denoise_wavelet.html

# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(data, multichannel=True, average_sigmas=True)

# Due to clipping in random_noise, the estimate will be a bit smaller than the specified sigma.
print("Estimated Gaussian noise standard deviation = {}".format(sigma_est))

denoised = denoise_wavelet(data, sigma=sigma_est, multichannel=True, mode='soft')

save("denoised", denoised)

