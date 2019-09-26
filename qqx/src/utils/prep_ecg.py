import os
import warnings
import numpy as np

from tqdm import *
from scipy.signal import resample_poly


warnings.filterwarnings('ignore')


def convert2npy(input_dir, output_dir):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    samples = os.listdir(input_dir)
    for sample in tqdm(samples, ncols=75):
        sample_path = os.path.join(input_dir, sample)
        with open(sample_path, 'r') as f:
            content = f.readlines()

        content = [list(map(int, c.strip().split())) for c in content[1:]]
        ecg = np.array(content, dtype=np.int16).transpose()
        ecg = resample_poly(ecg, 200, 500, axis=1)

        I, II = ecg[0], ecg[1]
        III = np.expand_dims(II - I, axis=0)
        aVR = np.expand_dims(-(I + II) / 2, axis=0)
        aVL = np.expand_dims(I - II / 2, axis=0)
        aVF = np.expand_dims(II - I / 2, axis=0)
        ecg = np.concatenate([ecg, III, aVR, aVL, aVF], axis=0)

        output_file = sample.split('.')[0] + '.npy'
        output_path = os.path.join(output_dir, output_file)
        np.save(output_path, ecg)
    return


def compute_mean_std(input_dir):

    samples = os.listdir(input_dir)
    for channel in range(12):
        data = np.array([], dtype=np.int16)
        for sample in tqdm(samples, ncols=75):
            sample_path = os.path.join(input_dir, sample)

            ecg = np.load(sample_path)
            data = np.append(data, ecg[channel, :])

        print(channel, 'mean:', np.mean(data), 'std:', np.std(data))

    return


if __name__ == '__main__':

    # convert2npy(
    #     input_dir='../../data/train_txt',
    #     output_dir='../../data/train_200Hz'
    # )

    # convert2npy(
    #     input_dir='../../data/testA_txt',
    #     output_dir='../../data/testA_200Hz'
    # )

    compute_mean_std('../../data/train_200Hz')

    # 500Hz
    # channels_mean = [0.618960, 0.975367, 0.080493, 1.174466,
    #                  1.418035, 1.422402, 1.189376, 0.955811]
    # channels_std = [25.066012, 33.287083, 39.536831, 62.618305,
    #                 60.006970, 64.556750, 58.439417, 50.985192]

    # 100Hz
    # channels_mean = [0.617703, 0.973512, 0.079987, 1.172066,
    #                  1.415036, 1.419375, 1.186931, 0.953722，
    #                  0.355809, -0.795608, 0.130947, 0.664661]
    # channels_std = [24.861912, 33.085598, 39.441234, 62.491415,
    #                 59.788809, 64.328094, 58.257034, 50.321352,
    #                 25.534215, 26.332237, 19.010292, 26.810405]

    # 200Hz
    # channels_mean = [0.618506, 0.974696, 0.080317]
    # channels_std = [25.004078, 33.233443, 39.522393]
