import torch

class ToQuaternion:

    def __init__(self, channel_dim=0):
        assert isinstance(channel_dim, int)
        self.channel_dim = channel_dim

    def __call__(self, image):
        C = image.size(self.channel_dim)
        if C < 4:
            zpad = torch.zeros(*image.shape[:self.channel_dim], 4 - C, *image.shape[self.channel_dim+1:])
            image = torch.cat((zpad, image), dim=self.channel_dim)

        return image