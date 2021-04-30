import torch
import torchvision
from .utils import rgb2gray

class ToQuaternion:

    def __init__(self, channel_dim=0, fill="grayscale"):
        assert isinstance(channel_dim, int)
        self.channel_dim = channel_dim
        assert fill in ("grayscale", "zeros")
        self.fill = fill

    def __call__(self, image):
        C = image.size(self.channel_dim)
        if C < 4:
            if self.fill == "grayscale":
                rfill = rgb2gray(image)
            else:
                rfill = torch.zeros(*image.shape[:self.channel_dim], 4 - C, *image.shape[self.channel_dim+1:])
            
            image = torch.cat((image, rfill), dim=self.channel_dim)

        return image