import numpy as np
import math

import os
import shutil

import torch
from   torch.autograd   import Variable, Function
import torch.nn         as nn
from   ..core.ops       import *
from   ..core.layers    import *


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


def _replace_module(model, orig_module_cls, new_module_cls, module_attrs={}):
    for idx, module in model.features._modules.items():
        if isinstance(module, orig_module_cls):
            if issubclass(new_module_cls, Function):
                model.features._modules[idx] = new_module_cls.apply
            else:
                module_kwargs = {k: v(getattr(module, k)) for k, v in module_attrs.items()}
                new_module = new_module_cls(**module_kwargs)
                model.features._modules[idx] = new_module
        else:
            for idx_c, child in model.features[int(idx)].named_children():
                if isinstance(child, orig_module_cls):
                    if issubclass(new_module_cls, Function):
                        model.features[int(idx)]._modules[idx_c] = new_module_cls.apply
                    else:
                        module_kwargs = {k: v(getattr(module, k)) for k, v in module_attrs.items()}
                        new_module = new_module_cls(**module_kwargs)
                        model.features[int(idx)]._modules[idx_c] = new_module


def convert_model_inplace(model):
    ident = lambda x: x
    ch_clip = lambda x: max(4, x)
    exists = lambda x: x is not None
    # _replace_module(model, nn.Linear, QuaternionLinear, module_attrs=
    #     {
    #         "in_channels": ch_clip,
    #         "out_channels": ch_clip,
    #         "bias": exists
    #     }
    # )
    _replace_module(model, nn.Conv2d, QuaternionConv, module_attrs=
        {
            "in_channels": ch_clip,
            "out_channels": ch_clip,
            "kernel_size": ident,
            "stride": ident,
            "padding": ident,
            "dilation": ident,
            "groups": ident,
            "bias": exists
        }
    )