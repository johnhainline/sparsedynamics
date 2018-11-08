
from skimage.io import imread, imsave
import numpy as np
import os
import pathlib

from load_img.baseimage import PETImage, CTImage, normalize
from skimage.restoration import (denoise_wavelet, estimate_sigma)


def add_gaussian_noise(image, mean, variance, seed=1234):
    np.random.seed(seed)
    sigma = np.sqrt(variance)
    noise = np.random.normal(mean, sigma, np.size(image)).reshape(np.shape(image))
    return np.clip(image + noise, a_min=0.0, a_max=1.0)


# see http://wiki.stat.ucla.edu/socr/index.php/AP_Statistics_Curriculum_2007_Limits_Norm2Poisson
def add_poisson_noise(image, durations, seed=1234):
    # remove negative numbers
    norm_image = np.where(image > 0.0, image, 0.0)

    # the amount of (poisson) noise is inversely proportional to the image duration
    norm_image = np.divide(durations, norm_image[:, :, :], out=np.zeros_like(norm_image), where=norm_image[:, :, :] != 0)

    np.random.seed(seed)
    result = np.zeros(image.shape)
    greater = norm_image > 1000
    less = np.invert(greater)
    greater_values = norm_image[greater]
    result[greater] = np.random.normal(greater_values, greater_values)
    result[less] = np.random.poisson(norm_image[less])
    result = np.divide(durations, result[:, :, :], out=np.zeros_like(result), where=result[:, :, :] != 0)

    # remove negative numbers
    result[result < 0.0] = 0.0

    # NOTE!!! If np.random.poisson could handle arbitrarily large integers we'd be fine, but we end up having to scale,
    # which means we end up calling poisson of a number less than 1, which is almost always 0, which effectively erases
    #  a ton of data.
    return result


# put filepath to image to open
filepath = os.path.join("data", "mpet3715b_em1_v1.pet.img")

print("Loading data...")

# load image into memory
my_img = PETImage(filepath=filepath)
my_img.load_image()


def save(filename, data):
    sum_over_time = np.sum(data, axis=3)

    # ensure the outputs directory exists
    outputs_path = pathlib.Path('outputs/')
    outputs_path.mkdir(parents=True, exist_ok=True)

    axis = np.sum(sum_over_time, axis=2)
    imsave(pathlib.Path('outputs/xy_' + filename + '.png'), axis / axis.max())
    axis = np.sum(sum_over_time, axis=1)
    imsave(pathlib.Path('outputs/xz_' + filename + '.png'), axis / axis.max())
    axis = np.sum(sum_over_time, axis=0)
    imsave(pathlib.Path('outputs/yz_' + filename + '.png'), axis / axis.max())


# frame_durations is the amount of time (in seconds) each frame took
frame_durations = my_img.params.frame_duration
# data is [x,y,z,t] array of float64
data = np.swapaxes(my_img.img_data, 0, 2)
my_img.unload_image()

print("Saving original images...")
save("original", data)

print("Generating noise...")
noisy = add_poisson_noise(data, frame_durations)

print("Saving noisy images...")
nice_noisy = np.where(noisy < np.percentile(noisy, 95), noisy, 0.0)
save("noisy", nice_noisy)

print("Estimating sigma...")
# See http://scikit-image.org/docs/dev/auto_examples/filters/plot_denoise_wavelet.html
# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(noisy, multichannel=True, average_sigmas=True)

# Due to clipping in random_noise, the estimate will be a bit smaller than the specified sigma.
print("Estimated Gaussian noise standard deviation = {}".format(sigma_est))

denoised = denoise_wavelet(noisy, sigma=sigma_est, multichannel=True, mode='soft')

print("Saving denoised images...")
save("denoised", denoised)
