import numpy as np
import math

import os
import shutil


def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))


def rgb2gray(rgb_image, keep_dim=False):
    gray = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])
    return np.repeat(gray[:, :, np.newaxis], 3, axis=2) if keep_dim else gray


def empty_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.mkdir(directory_path)




