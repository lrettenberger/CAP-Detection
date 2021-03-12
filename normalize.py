import glob

from PIL import Image
from skimage.exposure import equalize_adapthist
import numpy as np
import matplotlib.pyplot as plt

for img_name in glob.glob('*.png'):
    img = Image.open(img_name)
    im_normal = equalize_adapthist(np.array(img))
    pillow_normal = Image.fromarray(im_normal * 255)
    pillow_normal = pillow_normal.convert('L')
    pillow_normal.save(img_name, 'PNG')



#equalize_adapthist(image.numpy())