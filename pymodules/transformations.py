from PIL import Image
from skimage.exposure import equalize_adapthist
from torch.fft import Tensor
from torchvision.transforms import functional as F
import torch
import random
import numpy as np


class EqualizeAdaptiveHistogramEqualization(object):
    '''
    https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
    '''

    def __call__(self, image, target):
        if isinstance(image, Tensor):
            return F.to_tensor(equalize_adapthist(image.numpy())), target
        else:
            return equalize_adapthist(image), target


class ToTensor(object):
    '''
    Source: https://github.com/pytorch/vision/tree/master/references/segmentation
    '''

    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class RandomHorizontalFlip(object):
    '''
    Source: https://github.com/pytorch/vision/tree/master/references/segmentation
    '''

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class Compose(object):
    '''
    Source: https://github.com/pytorch/vision/tree/master/references/segmentation
    '''

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    '''
    Source: https://github.com/pytorch/vision/tree/master/references/segmentation
    '''

    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=Image.NEAREST)
        return image, target
