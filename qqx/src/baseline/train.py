import os
import json
import torch
import random
import numpy as np
import pandas as pd
import torch.optim as optim

from tcn import TCN, MSTCN
from resnet import ResNeXt
from utils import ComboLoss
from loader import ECGLoader
from adabound import AdaBound
from sklearn.metrics import f1_score


class ECGTrainer(object):

    def __init__(self, **params):
        torch.set_num_threads(2)
        self.n_epochs = params['n_epochs']
        self.batch_size = params['batch_size']
        self.cuda = torch.cuda.is_available()

        self.__build_model(**params)
        self.__build_criterion()
        self.__build_optimizer(**params)
        self.__build_scheduler()
        return

    def __build_model(self, **params):
        if params['model'] == 'resnet':
            self.model = ResNeXt()
        else:
            model_params = {'num_channels': params['num_channels'],
                            'kernel_size': params['kernel_size'],
                            'dropout': params['dropout']}
            if params['model'] == 'tcn':
                self.model = TCN(8, 55, **model_params)
            else:  # params['model'] == 'mstcn'
                self.model = MSTCN(8, 55, **model_params)

        if self.cuda:
            self.model.cuda()
        return

    def __build_criterion(self):
        self.criterion = ComboLoss(
            losses=['bce', 'mlsml', 'focal'],
            weights=[1, 1, 1]
        )
        return

    def __build_optimizer(self, **params):
        opt_params = {'lr': params['lr'],
                      'params': self.model.parameters(),
                      'weight_decay': params['weight_decay']}
        self.optimizer = AdaBound(amsbound=True, **opt_params)
        return

    def __build_scheduler(self):
        self.scheduler = None
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.3, patience=10,
            verbose=True, min_lr=1e-5)
        return

    def run(self, signal_dir, trainset, validset, model_dir):
        print('=' * 70 + '\n' + 'TRAINING MODEL\n' + '-' * 70 + '\n')
        model_path = os.path.join(model_dir, 'model.pth')

        dataloader = {
            'train': ECGLoader(trainset, signal_dir, self.batch_size).build(),
            'valid': ECGLoader(validset, signal_dir, self.batch_size).build()
        }

        best_metric = None
        for epoch in range(self.n_epochs):
            e_message = '[EPOCH {:0=3d}/{:0=3d}]'.format(
                epoch + 1, self.n_epochs)

            for phase in ['train', 'valid']:
                ep_message = e_message + '[' + phase.upper() + ']'
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                losses, preds, labels = [], np.array([]), np.array([])
                batch_num = len(dataloader[phase])
                for ith_batch, data in enumerate(dataloader[phase]):
                    ecg, label = [d.cuda() for d in data] if self.cuda else data

                    pred = self.model(ecg)
                    loss = self.criterion(pred, label)
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    pred[pred > 0.5] = 1
                    pred[pred <= 0.5] = 0
                    pred = pred.data.cpu().numpy().flatten()
                    label = label.data.cpu().numpy().flatten()
                    f1 = f1_score(label, pred, average='macro')

                    losses.append(loss.item())
                    preds = np.append(preds, pred)
                    labels = np.append(labels, label)

                    sr_message = '[STEP {:0=3d}/{:0=3d}]-[Loss: {:.6f} F1: {:.6f}]'
                    sr_message = ep_message + sr_message
                    print(sr_message.format(ith_batch + 1, batch_num, loss, f1), end='\r')

                avg_loss = np.mean(losses)
                avg_f1 = f1_score(labels, preds, average='macro')
                er_message = '-----[Loss: {:.6f} F1: {:.6f}]'
                er_message = '\n\033[94m' + ep_message + er_message + '\033[0m'
                print(er_message.format(avg_loss, avg_f1))

                if phase == 'valid':
                    if self.scheduler is not None:
                        self.scheduler.step(avg_loss)
                    if best_metric is None or best_metric > avg_loss:
                        best_metric = avg_loss
                        best_loss_metrics = [epoch + 1, avg_loss, avg_f1]
                        torch.save(self.model.state_dict(), model_path)
                        print('[Best validation loss, model: {}]'.format(model_path))
                    print()

        res_message = 'VALIDATION Performance: BEST LOSS' + '\n' \
            + '[EPOCH:{} LOSS:{:.6f} F1:{:.6f}]\n'.format(
                best_loss_metrics[0], best_loss_metrics[1], best_loss_metrics[2]) \
            + '=' * 70 + '\n'
        print(res_message)
        return


class ECGTrain(object):

    def __init__(self, params_file, param_set, cv=5):
        with open(params_file, 'r') as f:
            param_json = json.load(f)

        self.cv = cv
        self.param_set = param_set
        self.params = param_json[param_set]
        return

    def __split(self, i):
        idx1 = self.num_cv * (i - 1)
        idx2 = self.num_cv * i if i != self.cv else self.num
        trainset = self.dataset[0:idx1] + self.dataset[idx2:]
        validset = self.dataset[idx1:idx2]
        return trainset, validset

    def __model_dir(self, models_dir, i):
        model_dir = os.path.join(models_dir, self.param_set, str(i))
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        return model_dir

    def __train(self, signal_dir, trainset, validset, model_dir):
        trainer = ECGTrainer(**self.params)
        trainer.run(signal_dir, trainset, validset, model_dir)
        return

    def run(self, signal_dir, labels_csv, models_dir):
        self.dataset = pd.read_csv(labels_csv).values.tolist()
        random.seed(325)
        random.shuffle(self.dataset)

        self.num = len(self.dataset)
        self.num_cv = self.num // self.cv

        for i in range(1, self.cv + 1):
            trainset, validset = self.__split(i)
            model_dir = self.__model_dir(models_dir, i)
            self.__train(signal_dir, trainset, validset, model_dir)
        return


def main(args):
    pipeline = ECGTrain(params_file=args.param_json,
                        param_set=args.param_set,
                        cv=args.cv)
    pipeline.run(signal_dir=args.signal_dir,
                 labels_csv=args.labels_csv,
                 models_dir=args.models_dir)
    return


if __name__ == '__main__':
    import warnings
    import argparse

    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description='HF ECG Compitition - Training Pipeline'
    )

    parser.add_argument('--signal-dir', '-s', type=str,
                        action='store', dest='signal_dir',
                        help='Directory of input signals')
    parser.add_argument('--labels-csv', '-l', type=str,
                        action='store', dest='labels_csv',
                        help='Label file of input signals')
    parser.add_argument('--models-dir', '-m', type=str,
                        action='store', dest='models_dir',
                        help='Directory of output models')
    parser.add_argument('--param-file', '-f', type=str,
                        action='store', dest='param_json',
                        help='Json file of parameters')
    parser.add_argument('--param-set', '-p', type=str,
                        action='store', dest='param_set',
                        help='Set of parameters')
    parser.add_argument('--cv', '-c', type=int,
                        action='store', dest='cv',
                        help='Number of cross validation')
    parser.add_argument('--gpu', '-g', type=str,
                        action='store', dest='gpu',
                        help='Device NO. of GPU')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
