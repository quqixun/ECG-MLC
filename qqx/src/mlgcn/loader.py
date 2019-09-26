import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader


ECG_MEAN = np.array(
    [0.618, 0.974, 0.080, 1.172, 1.415, 1.419,
     1.186931, 0.954, 0.356, -0.796, 0.131, 0.665]
).reshape((-1, 1))
ECG_STD = np.array(
    [24.862, 33.086, 39.441, 62.491, 59.789, 64.328,
     58.257, 50.321, 25.534, 26.332, 19.010, 26.810]
).reshape((-1, 1))

HRV_MEAN = np.load('hrv_mean.npy').reshape((1, -1))
HRV_STD = np.load('hrv_std.npy').reshape((1, -1)) + 1e-8

AGE_MEAN = 48.435359
AGE_STD = 19.560310


class ECGDataset(Dataset):

    def __init__(self, dataset, signal_dir, hrv_dir, is_train):
        super(ECGDataset, self).__init__()

        self.dataset = dataset
        self.hrv_dir = hrv_dir
        self.is_train = is_train
        self.signal_dir = signal_dir
        return

    def __len__(self):
        return len(self.dataset)

    def __augment(self, ecg):
        ecg_tmp = np.copy(ecg)
        channels, length = ecg.shape

        if np.random.randn() > 0.5:
            scale = np.random.normal(loc=1.0, scale=0.1, size=(channels, 1))
            scale = np.matmul(scale, np.ones((1, length)))
            ecg_tmp = ecg_tmp * scale

        if np.random.randn() > 0.5:
            ecg_tmp = ecg_tmp[:, ::-1]

        if np.random.randn() > 0.5:
            ecg_tmp = -ecg_tmp

        if np.random.randn() > 0.5:
            for c in range(channels):
                offset = np.random.choice(range(-20, 20))
                ecg_tmp[c, :] += offset
        return ecg_tmp

    def __getitem__(self, index):
        sample = self.dataset[index]
        ID, age, sex, label = sample[0].split('.')[0], sample[1], sample[2], sample[3:]
        hrv_path = os.path.join(self.hrv_dir, ID + '.npy')
        ecg_path = os.path.join(self.signal_dir, ID + '.npy')

        age = (age - AGE_MEAN) / AGE_STD
        age = np.nan_to_num(age)

        if sex == 'FEMALE':
            sex = [1, 0, 0]
        elif sex == 'MALE':
            sex = [0, 1, 0]
        else:
            sex = [0, 0, 1]

        hrv = np.load(hrv_path)
        hrv = (hrv - HRV_MEAN) / HRV_STD
        hrv = np.nan_to_num(hrv)
        hrv = np.append(hrv, [age] + sex)

        ecg = np.load(ecg_path)
        if self.is_train:
            ecg = self.__augment(ecg)
        ecg = (ecg - ECG_MEAN) / ECG_STD
        return torch.FloatTensor(ecg), \
            torch.FloatTensor(hrv), \
            torch.FloatTensor(label)


class ECGLoader():

    def __init__(self, dataset, signal_dir, hrv_dir, batch_size, is_train):
        self.dataset = dataset
        self.hrv_dir = hrv_dir
        self.is_train = is_train
        self.signal_dir = signal_dir
        self.batch_size = batch_size
        return

    def build(self):
        ecg_dataset = ECGDataset(self.dataset, self.signal_dir,
                                 self.hrv_dir, self.is_train)
        dataloader = DataLoader(ecg_dataset,
                                batch_size=self.batch_size,
                                shuffle=True, num_workers=6,
                                pin_memory=False)
        return dataloader
