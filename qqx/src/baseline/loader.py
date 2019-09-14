import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader


# 500HZ
# ECG_MEAN = np.array(
#     [0.618960, 0.975367, 0.080493, 1.174466,
#      1.418035, 1.422402, 1.189376, 0.955811]
# ).reshape((-1, 1))

# ECG_STD = np.array(
#     [25.066012, 33.287083, 39.536831, 62.618305,
#      60.006970, 64.556750, 58.439417, 50.985192]
# ).reshape((-1, 1))

# 100Hz
ECG_MEAN = np.array(
    [0.617703, 0.973512, 0.079987, 1.172066,
     1.415036, 1.419375, 1.186931, 0.953722]
).reshape((-1, 1))

ECG_STD = np.array(
    [24.861912, 33.085598, 39.441234, 62.491415,
     59.788809, 64.328094, 58.257034, 50.321352]
).reshape((-1, 1))


class ECGDataset(Dataset):

    def __init__(self, dataset, signal_dir):
        super(ECGDataset, self).__init__()

        self.dataset = dataset
        self.signal_dir = signal_dir
        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        ID, label = sample[0].split('.')[0], sample[3:]
        ecg_path = os.path.join(self.signal_dir, ID + '.npy')

        ecg = (np.load(ecg_path) - ECG_MEAN) / ECG_STD
        return torch.FloatTensor(ecg), torch.FloatTensor(label)


class ECGLoader():

    def __init__(self, dataset, signal_dir, batch_size):
        self.dataset = dataset
        self.signal_dir = signal_dir
        self.batch_size = batch_size
        return

    def build(self):
        ecg_dataset = ECGDataset(self.dataset, self.signal_dir)
        dataloader = DataLoader(ecg_dataset,
                                batch_size=self.batch_size,
                                shuffle=True, num_workers=8,
                                pin_memory=False)
        return dataloader
