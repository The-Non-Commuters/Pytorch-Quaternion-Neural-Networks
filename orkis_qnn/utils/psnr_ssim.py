##########################################################
# pytorch-qnn v1.0                                     
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

from skimage.measure import compare_ssim as ssim
from utils.misc import psnr
import imageio
import sys


if len(sys.argv) != 3:
    raise Exception("Request 2 image-path arguments. Found: " + str(len(sys.argv)-1))

original = imageio.imread(sys.argv[1])
contrast = imageio.imread(sys.argv[2])

p = psnr(original, contrast)  # Peak signal-to-noise ratio
s = ssim(original, contrast, multichannel=True)  # Structural similarity

print("PSNR : " + str(p))
print("SSIM : " + str(s))
