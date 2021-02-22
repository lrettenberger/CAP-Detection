from skimage.exposure import equalize_hist, equalize_adapthist
from torch.utils.data import Dataset
import random
import os
import imageio
import numpy as np
import glob
import torch
import tensorflow as tf

from pymodules.model import SegNet
from pymodules.trainloop import train
from pymodules.transformations import Compose, ToTensor, RandomHorizontalFlip, EqualizeAdaptiveHistogramEqualization, Resize


class TeslaSiemensDataset(Dataset):

    def __init__(self, root_dir, transform=None, seed=0):
        self.root_dir = root_dir
        self.transform = transform
        self.cap_labels = os.path.join(self.root_dir, 'labels', 'cap')
        self.cg_labels = os.path.join(self.root_dir, 'labels', 'cg')
        self.prostate_labels = os.path.join(self.root_dir, 'labels', 'prostate')
        self.pz_labels = os.path.join(self.root_dir, 'labels', 'pz')
        self.samples = os.path.join(self.root_dir, 'samples')
        sorted_patients = sorted(glob.glob(f'{self.samples}/*.png'), key=lambda x: list(int(i) for i in x.split(f'{self.samples}\\')[1].split('.png')[0].split('_')))
        self.sorted_names = [i.split(f'{self.samples}\\')[1] for i in sorted_patients]
        random.seed(seed)

    def __len__(self):
        return len(self.sorted_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cap_img = np.array(imageio.imread(f'{self.cap_labels}/{self.sorted_names[idx]}'))
        cg_img = np.array(imageio.imread(f'{self.cg_labels}/{self.sorted_names[idx]}'))
        prostate_img = np.array(imageio.imread(f'{self.prostate_labels}/{self.sorted_names[idx]}'))
        pz_img = np.array(imageio.imread(f'{self.pz_labels}/{self.sorted_names[idx]}'))
        combined_labels = np.concatenate((cap_img.reshape(1, cap_img.shape[0], cap_img.shape[1]),
                                          cg_img.reshape(1, cg_img.shape[0], cg_img.shape[1]),
                                          prostate_img.reshape(1, prostate_img.shape[0], prostate_img.shape[1]),
                                          pz_img.reshape(1, pz_img.shape[0], pz_img.shape[1])),
                                         axis=0)
        sample_img = np.array(imageio.imread(f'{self.samples}/{self.sorted_names[idx]}'))

        if self.transform:
            sample_img, combined_labels = self.transform(sample_img, combined_labels)

        return sample_img, combined_labels


# transform = Compose([
#    EqualizeAdaptiveHistogramEqualization(),
#    ToTensor(),
#    RandomHorizontalFlip()
# ])

from torch.nn import functional as F


def dice(input, target):
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def dice_loss(input, target):
    for i in range(input.shape[0]):
        scores = []
        input_i = input[i,:,:,:]
        targe_t = target[i,:,:,:]
        for j in range(input_i.shape[0]):
            scores.append(dice(input_i[j,:,:],targe_t[j,:,:]))
        print(scores)




dataset = TeslaSiemensDataset(root_dir='../data/siemens_reduced/train')

# dice_loss(torch.from_numpy(dataset[0][1].reshape(1,4,360,448)),torch.from_numpy(dataset[0][1].reshape(1,4,360,448)))

dice_loss(torch.from_numpy(tf.ones([1, 4, 360, 448]).numpy()), torch.from_numpy(dataset[0][1].reshape(1, 4, 360, 448)))

print()

transform = Compose([
    EqualizeAdaptiveHistogramEqualization(),
    ToTensor(),
    Resize((255, 255)),
    RandomHorizontalFlip()
])
