from typing import Optional, Sequence

from skimage.exposure import equalize_hist, equalize_adapthist
from torch.utils.data import Dataset, Sampler
import random
import os
import imageio
import numpy as np
import glob
import torch
import tensorflow as tf
import torch.optim as optim

from PIL import Image
from torch.utils.data._utils.fetch import _BaseDatasetFetcher, _MapDatasetFetcher
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter, DataLoader, T_co, _collate_fn_t, _worker_init_fn_t, _DatasetKind

from pymodules.DiceLoss import GeneralizedDiceLoss
from pymodules.LossFunctions import weighted_dice, weighted_cross_entropy
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
        self.sequence_boundaries = []
        for i in range(len(self.sorted_names)-1):
            patient_number_current = int(self.sorted_names[i][0])
            patient_number_next = int(self.sorted_names[i+1][0])
            if patient_number_current != patient_number_next:
                self.sequence_boundaries.append(i)

        random.seed(seed)

    def __len__(self):
        return len(self.sorted_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cap_img = np.array(imageio.imread(f'{self.cap_labels}/{self.sorted_names[idx]}'))
        cg_img = np.array(imageio.imread(f'{self.cg_labels}/{self.sorted_names[idx]}'))
        pz_img = np.array(imageio.imread(f'{self.pz_labels}/{self.sorted_names[idx]}'))
        combined_labels = np.concatenate((cap_img.reshape(1, cap_img.shape[0], cap_img.shape[1]),
                                          cg_img.reshape(1, cg_img.shape[0], cg_img.shape[1]),
                                          pz_img.reshape(1, pz_img.shape[0], pz_img.shape[1])),
                                         axis=0)
        combined_labels = np.concatenate((combined_labels, ((np.sum(combined_labels, axis=0) == 0) * 255).reshape(1, combined_labels.shape[1], combined_labels.shape[2])))

        if np.min(np.sum(combined_labels, axis=0)) == 0:
            print('ERROR: non hot encoded vector in gt mask')

        # enforce one hot
        combined_labels = np.transpose(np.eye(4)[combined_labels.argmax(axis=0)], axes=[2, 0, 1])

        sample_img = np.array(imageio.imread(f'{self.samples}/{self.sorted_names[idx]}'))

        if self.transform:
            sample_img, combined_labels = self.transform(sample_img, combined_labels)

        return sample_img, combined_labels


class TeslaSiemensDataLoader(DataLoader):

    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1, shuffle: bool = False, sampler: Optional[Sampler[int]] = None, batch_sampler: Optional[Sampler[Sequence[int]]] = None, num_workers: int = 0, collate_fn: _collate_fn_t = None, pin_memory: bool = False, drop_last: bool = False, timeout: float = 0, worker_init_fn: _worker_init_fn_t = None, multiprocessing_context=None, generator=None, *, prefetch_factor: int = 2, persistent_workers: bool = False):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, False, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return TeslaFrameDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)


class TeslaFrameDataLoaderIter(_SingleProcessDataLoaderIter):

    def __init__(self, loader):
        super().__init__(loader)

        self._dataset_fetcher = TeslaDatasetFetcher(self._dataset, self._auto_collation, self._collate_fn, self._drop_last)

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        return data


class TeslaDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super().__init__(dataset, auto_collation, collate_fn, drop_last)

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)


rgb_map = [
    [255, 0, 0],  # cap
    [181, 70, 174],  # cg
    [61, 184, 102],  # pz
    [0, 0, 0]  # background
]

import matplotlib.pyplot as plt


def vizualize_labels(labels):
    maxes = torch.argmax(labels, dim=0)
    rgb_values = [rgb_map[p] for p in maxes.numpy().flatten()]
    plt.imshow(np.array(rgb_values).reshape(255, 255, 3))
    plt.show()


def main():
    transform = Compose([
        ToTensor(),
        Resize((368, 448)),
        RandomHorizontalFlip()
    ])

    net = SegNet(1, 4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.load_state_dict(torch.load('../dice2.pt'))
    criterion = weighted_dice
    criterion_2 = weighted_cross_entropy
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    BATCH_SIZE = 3

    trainset = TeslaSiemensDataset(root_dir='../data/siemens_reduced/train', transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)



    testset = TeslaSiemensDataset(root_dir='../data/siemens_reduced/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    train(model=net, optimizer=optimizer, loss_fn=criterion, train_loader=trainloader, val_loader=testloader, epochs=100, device=device, best_model_name='dice2', early_stopping_patience=12)


if __name__ == "__main__":
    main()
