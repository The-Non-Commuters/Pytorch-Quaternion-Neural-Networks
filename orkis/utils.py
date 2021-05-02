import numpy as np
import math

import os
import shutil

import torch
from   torch.autograd   import Variable, Function
import torch.nn         as nn
import torchvision

from   .qnn.ops       import *
from   .qnn    import *


def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))


# def rgb2gray(rgb_image, keep_dim=False):
#     gray = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])
#     return np.repeat(gray[:, :, np.newaxis], 3, axis=2) if keep_dim else gray
rgb2gray = torchvision.transforms.Grayscale(num_output_channels=1)


def empty_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.mkdir(directory_path)


def convert_data_for_quaternion(batch):
    """
    converts batches of RGB images in 4 channels for QNNs
    """
    assert all(batch[i][0].size(0) == 3 for i in range(len(batch)))
    inputs, labels = [], []
    for i in range(len(batch)):
        inputs.append(torch.cat([batch[i][0], rgb2gray(batch[i][0])], 0))
        labels.append(batch[i][1])
    
    return torch.stack(inputs), torch.LongTensor(labels)
    