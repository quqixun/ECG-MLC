import os
import torch
import numpy as np

from tqdm import *
from pathlib import Path
from resnet import ResNeXt
from tcn import TCN, MSTCN
from torch.autograd import Variable
from scipy.signal import resample_poly


# 100Hz
ECG_MEAN = np.array(
    [0.617703, 0.973512, 0.079987, 1.172066,
     1.415036, 1.419375, 1.186931, 0.953722]
).reshape((-1, 1))

ECG_STD = np.array(
    [24.861912, 33.085598, 39.441234, 62.491415,
     59.788809, 64.328094, 58.257034, 50.321352]
).reshape((-1, 1))


class ECGPredictor(object):

    def __init__(self, model_name, model_path, labels):
        if model_name == 'resnet':
            self.model = ResNeXt()
        elif model_name == 'tcn':
            self.model = TCN(8, 55, [32, 32, 64, 64, 128, 128], 7)
        elif model_name == 'mstcn':
            self.model = MSTCN(8, 55, [32, 32, 64, 64, 128, 128], 7)
        else:
            raise ValueError('Invalid model_name')

        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu')
        )
        self.model.eval()

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model.cuda()

        self.labels = labels
        return

    def __load(self, sample_path):
        with open(sample_path, 'r') as f:
            ecg = f.readlines()
        ecg = [list(map(int, c.strip().split())) for c in ecg[1:]]
        ecg = np.array(ecg, dtype=np.int16).transpose()
        ecg = resample_poly(ecg, 20, 100, axis=1)
        ecg = (ecg - ECG_MEAN) / ECG_STD
        ecg = np.expand_dims(ecg, 0)
        return ecg

    def __to_variable(self, ecg):
        ecg_tensor = torch.Tensor(ecg)
        if self.cuda:
            ecg_tensor = ecg_tensor.cuda()
        ecg_var = Variable(ecg_tensor)
        return ecg_var

    def run(self, sample_path):
        ecg = self.__load(sample_path)
        ecg_var = self.__to_variable(ecg)

        pred = self.model(ecg_var)
        pred_max = torch.argmax(pred, dim=1)[0].item()
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        pred = pred.data.cpu().numpy().flatten()
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

    print('\n' + '=' * 75)
    print('Predict test set:')
    print('Submit ID:', args.submit_id)
    print('Input dir:', args.signal_dir)
    print('Model path:', args.model_path)
    print('output_dir:', args.output_dir)
    print('-' * 75, '\n')

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
        predictor = ECGPredictor(args.model_name, model_path, labels)
        model_results = []
        for sample in tqdm(submit_samples, ncols=75):
            sample_strip = sample.strip('\n')
            sample_file = sample_strip.split('\t')[0]
            sample_path = os.path.join(args.signal_dir, sample_file)
            result = predictor.run(sample_path)
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
    print('=' * 75)
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
    parser.add_argument('--model-path', '-m', type=str,
                        action='store', dest='model_path',
                        help='Path of model used to predict')
    parser.add_argument('--submit-txt', '-t', type=str,
                        action='store', dest='submit_txt',
                        help='Txt file of test sample for submitting')
    parser.add_argument('--output-dir', '-o', type=str,
                        action='store', dest='output_dir',
                        help='Directory to save results')
    parser.add_argument('--gpu', '-g', type=str,
                        action='store', dest='gpu',
                        help='Devoce NO. of GPU')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
