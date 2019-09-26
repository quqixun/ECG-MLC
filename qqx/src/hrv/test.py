import os
import torch
import numpy as np

from tqdm import *
from pathlib import Path
from resnet import ResNeXt
from torch.autograd import Variable
from scipy.signal import resample_poly


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


class ECGPredictor(object):

    def __init__(self, model_name, model_path, threshold, labels):
        self.model = ResNeXt(
            kss=[13, 11, 9, 7, 5], layers=[3, 3, 3, 3, 3],
            groups=8, hrv_feats=2212
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu')
        )
        self.model.eval()

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model.cuda()

        self.threshold = threshold
        self.labels = labels
        return

    def __load_ecg(self, sample_path, downsample):
        with open(sample_path, 'r') as f:
            ecg = f.readlines()
        ecg = [list(map(int, c.strip().split())) for c in ecg[1:]]
        ecg = np.array(ecg, dtype=np.int16).transpose()
        ecg = resample_poly(ecg, downsample, 500, axis=1)

        I, II = ecg[0], ecg[1]
        III = np.expand_dims(II - I, axis=0)
        aVR = np.expand_dims(-(I + II) / 2, axis=0)
        aVL = np.expand_dims(I - II / 2, axis=0)
        aVF = np.expand_dims(II - I / 2, axis=0)
        ecg = np.concatenate([ecg, III, aVR, aVL, aVF], axis=0)

        ecg = (ecg - ECG_MEAN) / ECG_STD
        ecg = np.expand_dims(ecg, 0)
        return ecg

    def __load_hrv(self, hrv_path, age, sex):
        age = (int(age) - AGE_MEAN) / AGE_STD if age != '' else 0

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
        hrv = np.expand_dims(hrv, 0)
        return hrv

    def __to_variable(self, ecg):
        ecg_tensor = torch.Tensor(ecg)
        if self.cuda:
            ecg_tensor = ecg_tensor.cuda()
        ecg_var = Variable(ecg_tensor)
        return ecg_var

    def run(self, sample_path, hrv_path, age, sex, downsample):
        ecg = self.__load_ecg(sample_path, downsample)
        ecg_var = self.__to_variable(ecg)

        hrv = self.__load_hrv(hrv_path, age, sex)
        hrv_var = self.__to_variable(hrv)

        pred = self.model(ecg_var, hrv_var)
        pred = torch.sigmoid(pred)
        pred_max = torch.argmax(pred, dim=1)[0].item()
        pred = pred.data.cpu().numpy().flatten()
        pred = (pred > self.threshold) * 1
        result = [t for p, t in zip(pred, self.labels) if p]

        if not result:
            result = [self.labels[pred_max]]
        return result


def ensemble(multiple_results):
    submit_results = []
    n_results = len(multiple_results)
    for results in zip(*multiple_results):
        result_pre = results[0][0]
        result_dict = {}
        for result in results:
            for pred in result[1:]:
                if pred in result_dict.keys():
                    result_dict[pred] += 1
                else:
                    result_dict[pred] = 1
        result = []
        for k, v in result_dict.items():
            if v >= n_results / 2:
                result.append(k)
        result = [result_pre] + result
        result_str = '\t'.join(result)
        submit_results.append(result_str)
    return submit_results


def main(args):

    print('\n' + '=' * 70)
    print('Predict test set:')
    print('Submit ID:', args.submit_id)
    print('Input dir:', args.signal_dir)
    print('HRV dir:', args.hrv_dir)
    print('Model path:', args.model_path)
    print('Output_dir:', args.output_dir)
    print('-' * 70, '\n')

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    labels_file = '../../data/hf_round1_arrythmia.txt'
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    labels = [a.strip() for a in labels]

    with open(args.submit_txt, 'r', encoding='utf-8') as f:
        submit_samples = f.readlines()

    model_paths, single_model = [], True
    if os.path.isdir(args.model_path):
        model_paths = Path(args.model_path).glob('**/*.pth')
        single_model = False
    else:
        model_paths = [args.model_path]

    multiple_results = []
    for i, model_path in enumerate(model_paths):
        print('Model:', model_path)
        model_dir = '/'.join(str(model_path).split('/')[:-1])
        threshold = np.load(os.path.join(model_dir, 'threshold.npy'))
        predictor = ECGPredictor(args.model_name, model_path, threshold, labels)
        model_results = []
        for sample in tqdm(submit_samples, ncols=70):
            sample_strip = sample.strip('\n')
            sample_split = sample_strip.split('\t')
            sample_id = sample_split[0].split('.')[0]

            age, sex = sample_split[1], sample_split[2]
            sample_path = os.path.join(args.signal_dir, sample_id + '.txt')
            hrv_path = os.path.join(args.hrv_dir, sample_id + '.npy')

            result = predictor.run(sample_path, hrv_path, age, sex,
                                   args.downsample)
            result = [sample_strip] + result
            model_results.append(result)
        multiple_results.append(model_results)
        print()

    submit_results = []
    if single_model:
        for result in multiple_results[0]:
            result_str = '\t'.join(result)
            submit_results.append(result_str)
    else:
        submit_results = ensemble(multiple_results)

    output_path = os.path.join(args.output_dir, args.submit_id + '.txt')
    with open(output_path, 'w') as f:
        for line in submit_results:
            f.write('{}\n'.format(line))
    print('Submission:', output_path)
    print('=' * 70)
    return


if __name__ == '__main__':
    import warnings
    import argparse

    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description='HF ECG Compitition - Predict Test Samples'
    )

    parser.add_argument('--submit-id', '-id', type=str,
                        action='store', dest='submit_id',
                        help='ID of submission')
    parser.add_argument('--model-name', '-n', type=str,
                        action='store', dest='model_name',
                        help='Model type used to predict')
    parser.add_argument('--signal-dir', '-s', type=str,
                        action='store', dest='signal_dir',
                        help='Directory of training data')
    parser.add_argument('--hrv-dir', '-v', type=str,
                        action='store', dest='hrv_dir',
                        help='Directory of HRV features')
    parser.add_argument('--model-path', '-m', type=str,
                        action='store', dest='model_path',
                        help='Path of model used to predict')
    parser.add_argument('--submit-txt', '-t', type=str,
                        action='store', dest='submit_txt',
                        help='Txt file of test sample for submitting')
    parser.add_argument('--output-dir', '-o', type=str,
                        action='store', dest='output_dir',
                        help='Directory to save results')
    parser.add_argument('--downsample', '-d', type=int,
                        action='store', dest='down_sample',
                        help='Down sampling frequency')
    parser.add_argument('--gpu', '-g', type=str,
                        action='store', dest='gpu',
                        help='Devoce NO. of GPU')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
