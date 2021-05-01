from PIL import Image
from skimage.exposure import equalize_adapthist
from torch.fft import Tensor
import torch
import random
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torchvision


def pad_circular(x, pad):
    """
    :param x: shape [H, W]
    :param pad: int >= 0
    :return:
    """
    x = torch.cat([x, x[0:pad]], dim=0)
    x = torch.cat([x, x[:, 0:pad]], dim=1)
    x = torch.cat([x[-2 * pad:-pad], x], dim=0)
    x = torch.cat([x[:, -2 * pad:-pad], x], dim=1)

    return x


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


class RandomGaussianNoise(object):

    def __init__(self, noise_ammount):
        self.noise_ammount = noise_ammount

    def __call__(self, image, target):
        return image + (torch.randn_like(image)*self.noise_ammount), target


class RandomRotation(object):

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, target):
        rotation = T.RandomRotation.get_params([-self.degrees, self.degrees])
        image = F.rotate(image, rotation)
        target = F.rotate(target, rotation)
        return image, target


class RandomCrop(object):
    '''
    Source: https://github.com/pytorch/vision/tree/master/references/segmentation
    '''

    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.padder = torch.nn.ReflectionPad2d(crop_size // 2)

    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, (image.shape[1] - self.crop_size, image.shape[2] - self.crop_size))
        image = self.padder((F.crop(image, *crop_params)).reshape(image.shape[0], 1, image.shape[1] - self.crop_size, image.shape[2] - self.crop_size))[:, 0, :, :]
        target = self.padder((F.crop(target, *crop_params)).reshape(1, target.shape[0], target.shape[1] - self.crop_size, target.shape[2] - self.crop_size).double())[0].int()
        return image, target
