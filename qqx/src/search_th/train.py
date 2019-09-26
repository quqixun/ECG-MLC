import os
import copy
import json
import torch
import random
import pickle
import numpy as np
import pandas as pd
import torch.optim as optim

from resnet import ResNeXt
from loader import ECGLoader
from adabound import AdaBound
from sklearn.metrics import f1_score
from utils import ComboLoss, best_f1_score


class ECGTrainer(object):

    def __init__(self, **params):
        torch.set_num_threads(6)
        self.n_epochs = params['n_epochs']
        self.batch_size = params['batch_size']
        self.cuda = torch.cuda.is_available()

        self.__build_model(**params)
        self.__build_criterion()
        self.__build_optimizer(**params)
        self.__build_scheduler()
        return

    def __build_model(self, **params):
        self.model = ResNeXt(in_channels=12, groups=8)
        if self.cuda:
            self.model.cuda()
        return

    def __build_criterion(self):
        self.criterion = ComboLoss(
            losses=['focal', 'f1', 'mlsml'], weights=[1, 1, 1]
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
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, 'min', factor=0.333, patience=5,
        #     verbose=True, min_lr=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[31, 81], gamma=0.333)
        return

    def run(self, signal_dir, trainset, validset, model_dir):
        print('=' * 70 + '\n' + 'TRAINING MODEL\n' + '-' * 70 + '\n')
        model_path = os.path.join(model_dir, 'model.pth')
        thresh_path = os.path.join(model_dir, 'threshold.npy')

        dataloader = {
            'train': ECGLoader(trainset, signal_dir, self.batch_size, True).build(),
            'valid': ECGLoader(validset, signal_dir, self.batch_size, False).build()
        }

        best_metric, best_preds = None, None
        for epoch in range(self.n_epochs):
            e_message = '[EPOCH {:0=3d}/{:0=3d}]'.format(epoch + 1, self.n_epochs)

            for phase in ['train', 'valid']:
                ep_message = e_message + '[' + phase.upper() + ']'
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                losses, preds, labels = [], [], []
                batch_num = len(dataloader[phase])
                for ith_batch, data in enumerate(dataloader[phase]):
                    ecg, label = [d.cuda() for d in data] if self.cuda else data

                    pred = self.model(ecg)
                    loss = self.criterion(pred, label)
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    pred = torch.sigmoid(pred)
                    pred = pred.data.cpu().numpy()
                    label = label.data.cpu().numpy()

                    bin_pred = np.copy(pred)
                    bin_pred[bin_pred > 0.5] = 1
                    bin_pred[bin_pred <= 0.5] = 0
                    f1 = f1_score(label.flatten(), bin_pred.flatten(), average='macro')

                    losses.append(loss.item())
                    preds.append(pred)
                    labels.append(label)

                    sr_message = '[STEP {:0=3d}/{:0=3d}]-[Loss: {:.6f} F1: {:.6f}]'
                    sr_message = ep_message + sr_message
                    print(sr_message.format(ith_batch + 1, batch_num, loss, f1), end='\r')

                preds = np.concatenate(preds, axis=0)
                labels = np.concatenate(labels, axis=0)
                bin_preds = np.copy(preds)
                bin_preds[bin_preds > 0.5] = 1
                bin_preds[bin_preds <= 0.5] = 0

                avg_loss = np.mean(losses)
                avg_f1 = f1_score(labels.flatten(), bin_preds.flatten(), average='macro')
                er_message = '-----[Loss: {:.6f} F1: {:.6f}]'
                er_message = '\n\033[94m' + ep_message + er_message + '\033[0m'
                print(er_message.format(avg_loss, avg_f1))

                if phase == 'valid':
                    self.scheduler.step()
                    # if self.scheduler is not None:
                    #     self.scheduler.step(avg_loss)
                    if best_metric is None or best_metric < avg_f1:
                        best_metric = avg_f1
                        best_preds = [labels, preds]
                        best_loss_metrics = [epoch + 1, avg_loss, avg_f1]
                        torch.save(self.model.state_dict(), model_path)
                        print('[Best validation metric, model: {}]'.format(model_path))
                    print()

        best_f1, best_th = best_f1_score(*best_preds)
        np.save(thresh_path, np.array(best_th))
        print('[Searched Best F1: {:.6f}]'.format(best_f1))
        res_message = 'VALIDATION PERFORMANCE: BEST F1' + '\n' \
            + '[EPOCH:{} LOSS:{:.6f} F1:{:.6f} BEST F1:{:.6f}]\n'.format(
                best_loss_metrics[0], best_loss_metrics[1],
                best_loss_metrics[2], best_f1) \
            + '[BEST THRESHOLD:\n{}]\n'.format(best_th) \
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

    def __split(self, dataset, classes):

        labels = np.array([d[3:] for d in dataset])
        classes_samples = np.sum(labels, axis=0)
        sorted_classes_index = np.argsort(classes_samples)
        # classes_weights = np.round(1000 * len(labels) / classes_samples) / 1000

        classes_dict, visited_rows = {}, []
        for class_index in sorted_classes_index:
            rows = list(np.where(labels[:, class_index] == 1)[0])
            rows_tmp = copy.deepcopy(rows)

            for row in rows:
                if row in visited_rows:
                    rows_tmp.remove(row)
                else:
                    visited_rows.append(row)

            if len(rows_tmp) > 0:
                splits = [rows_tmp[i::self.cv] for i in range(self.cv)]
                classes_dict[classes[class_index]] = splits

        return classes_dict

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
        self.dataset_df = pd.read_csv(labels_csv)
        self.dataset = self.dataset_df.values.tolist()
        self.classes = self.dataset_df.columns.tolist()

        random.seed(325)
        random.shuffle(self.dataset)
        classes_dict = self.__split(self.dataset, self.classes)

        splits = []
        for i in range(self.cv):
            trainidx, valididx = [], []
            for _, (_, value) in enumerate(classes_dict.items()):
                for j, v in enumerate(value):
                    if j == i:
                        valididx += value[j]
                    else:
                        trainidx += value[j]

            trainset = [self.dataset[k] for k in trainidx]
            validset = [self.dataset[k] for k in valididx]
            splits.append([trainset, validset])

            model_dir = self.__model_dir(models_dir, i + 1)
            self.__train(signal_dir, trainset, validset, model_dir)

        pickle.dump(splits, open('./train_valid_splits.pkl', 'wb'))
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
