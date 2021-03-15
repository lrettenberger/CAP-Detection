from torch.utils.data import Dataset
import random
import os
import imageio
import numpy as np
import glob
import torch
import matplotlib.pyplot as plt


class TeslaSiemensDataset(Dataset):

    def __init__(self, root_dir, transform=None, seed=0, num_of_surrouding_imgs=0, include_cap=True):
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
        self.num_of_surrouding_imgs = num_of_surrouding_imgs
        self.include_cap = include_cap
        for i in range(len(self.sorted_names) - 1):
            patient_number_current = int(self.sorted_names[i][0])
            patient_number_next = int(self.sorted_names[i + 1][0])
            if patient_number_current != patient_number_next:
                self.sequence_boundaries.append(i)
        random.seed(seed)

    def __len__(self):
        return len(self.sorted_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cg_img = np.array(imageio.imread(f'{self.cg_labels}/{self.sorted_names[idx]}'))
        pz_img = np.array(imageio.imread(f'{self.pz_labels}/{self.sorted_names[idx]}'))

        if self.include_cap:
            cap_img = np.array(imageio.imread(f'{self.cap_labels}/{self.sorted_names[idx]}'))
            combined_labels = np.concatenate((cap_img.reshape(1, cap_img.shape[0], cap_img.shape[1]),
                                              cg_img.reshape(1, cg_img.shape[0], cg_img.shape[1]),
                                              pz_img.reshape(1, pz_img.shape[0], pz_img.shape[1])),
                                             axis=0)
            combined_labels = np.concatenate((combined_labels, ((np.sum(combined_labels, axis=0) == 0) * 255).reshape(1, combined_labels.shape[1], combined_labels.shape[2])))
            if np.min(np.sum(combined_labels, axis=0)) == 0:
                print('ERROR: non hot encoded vector in gt mask')
            # enforce one hot
            combined_labels = np.transpose(np.eye(4)[combined_labels.argmax(axis=0)], axes=[2, 0, 1])
        else:
            combined_labels = np.concatenate((cg_img.reshape(1, cg_img.shape[0], cg_img.shape[1]),
                                              pz_img.reshape(1, pz_img.shape[0], pz_img.shape[1])),
                                             axis=0)
            combined_labels = np.concatenate((combined_labels, ((np.sum(combined_labels, axis=0) == 0) * 255).reshape(1, combined_labels.shape[1], combined_labels.shape[2])))
            if np.min(np.sum(combined_labels, axis=0)) == 0:
                print('ERROR: non hot encoded vector in gt mask')
            # enforce one hot
            combined_labels = np.transpose(np.eye(3)[combined_labels.argmax(axis=0)], axes=[2, 0, 1])

        if self.num_of_surrouding_imgs == 0:
            sample_img = np.array(imageio.imread(f'{self.samples}/{self.sorted_names[idx]}'))
            if self.transform:
                sample_img, combined_labels = self.transform(sample_img, combined_labels)
            return sample_img, combined_labels
        elif self.num_of_surrouding_imgs == 1:
            # Corner cases are resolved with index shifts, hacky solution.
            # 0 is not valid, just shift index (hacky solution)
            if idx == 0:
                idx += 1
            # last item is also not valid
            if idx == (len(self) - 1):
                idx -= 1

            sample_patient_number = int(self.sorted_names[idx].split('_')[0])
            next_patient_number = int(self.sorted_names[idx + 1].split('_')[0])
            previous_patient_number = int(self.sorted_names[idx - 1].split('_')[0])

            if next_patient_number != sample_patient_number:
                idx -= 1
            if previous_patient_number != sample_patient_number:
                idx += 1

            sample_patient_number = int(self.sorted_names[idx].split('_')[0])
            next_patient_number = int(self.sorted_names[idx + 1].split('_')[0])
            previous_patient_number = int(self.sorted_names[idx - 1].split('_')[0])

            if sample_patient_number != next_patient_number or sample_patient_number != previous_patient_number:
                print('ERROR IN SEQUENCE CREATION')

            current_img = np.array(imageio.imread(f'{self.samples}/{self.sorted_names[idx]}'))
            next_img = np.array(imageio.imread(f'{self.samples}/{self.sorted_names[idx + 1]}'))
            previous_img = np.array(imageio.imread(f'{self.samples}/{self.sorted_names[idx - 1]}'))

            width = current_img.shape[0]
            height = current_img.shape[1]
            sample_img = np.concatenate((previous_img.reshape(width, height, 1), current_img.reshape(width, height, 1), next_img.reshape(width, height, 1)), axis=2)

            if self.transform:
                sample_img, combined_labels = self.transform(sample_img, combined_labels)
            return sample_img, combined_labels
