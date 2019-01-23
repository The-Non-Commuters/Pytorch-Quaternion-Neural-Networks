##########################################################
# pytorch-qnn v1.0                                     
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

from skimage.measure import compare_ssim as ssim
import imageio
import numpy
import math
import sys


def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))


original = imageio.imread(sys.argv[1])
contrast = imageio.imread(sys.argv[2])

p = psnr(original, contrast)  # Peak signal-to-noise ratio
s = ssim(original, contrast, multichannel=True)  # Structural similarity

print("PSNR : " + str(p))
print("SSIM : " + str(s))
