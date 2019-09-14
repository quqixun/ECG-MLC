import os
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    data_dir = '../../data/train_npy'
    samples = os.listdir(data_dir)

    for sample in samples:
        sample_path = os.path.join(data_dir, sample)
        ecg = np.load(sample_path)
        ecg = (ecg - ECG_MEAN) / ECG_STD

        plt.figure()
        for i in range(ecg.shape[0]):
            plt.subplot(4, 2, i + 1)
            plt.plot(np.arange(ecg.shape[1]), ecg[i, :])
        plt.tight_layout()
        plt.show()
