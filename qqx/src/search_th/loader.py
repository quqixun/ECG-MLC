import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader


# 100Hz
ECG_MEAN = np.array(
    [0.617703, 0.973512, 0.079987, 1.172066,
     1.415036, 1.419375, 1.186931, 0.953722,
     0.355809, -0.795608, 0.130947, 0.664661]
).reshape((-1, 1))

ECG_STD = np.array(
    [24.861912, 33.085598, 39.441234, 62.491415,
     59.788809, 64.328094, 58.257034, 50.321352,
     25.534215, 26.332237, 19.010292, 26.810405]
).reshape((-1, 1))


class ECGDataset(Dataset):

    def __init__(self, dataset, signal_dir, is_train):
        super(ECGDataset, self).__init__()

        self.dataset = dataset
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
            for c in range(channels):
                offset = np.random.choice(range(-20, 20))
                ecg_tmp[c, :] += offset
        return ecg_tmp

    def __getitem__(self, index):
        sample = self.dataset[index]
        ID, label = sample[0].split('.')[0], sample[3:]
        ecg_path = os.path.join(self.signal_dir, ID + '.npy')

        ecg = np.load(ecg_path)
        if self.is_train:
            ecg = self.__augment(ecg)
        ecg = (ecg - ECG_MEAN) / ECG_STD
        return torch.FloatTensor(ecg), torch.FloatTensor(label)


class ECGLoader():

    def __init__(self, dataset, signal_dir, batch_size, is_train):
        self.dataset = dataset
        self.is_train = is_train
        self.signal_dir = signal_dir
        self.batch_size = batch_size
        return

    def build(self):
        ecg_dataset = ECGDataset(self.dataset, self.signal_dir, self.is_train)
        dataloader = DataLoader(ecg_dataset,
                                batch_size=self.batch_size,
                                shuffle=True, num_workers=6,
                                pin_memory=False)
        return dataloader
