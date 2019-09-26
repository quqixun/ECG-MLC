import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


CLASSES_WEIGHTS = np.log10([
    1.425, 7.046, 7.149, 8.125, 5.77,
    5.572, 11.419, 11.993, 12.614, 15.623,
    16.399, 21.201, 22.849, 19.808, 22.720,
    24.826, 21.737, 26.089, 55.801, 61.495,
    85.482, 105.266, 121.1360, 192.848, 238.673,
    227.415, 309.051, 321.413, 395.180, 325.757,
    438.291, 454.830, 573.952, 709.000, 669.611,
    651.514, 892.815, 730.485, 803.533, 831.241,
    964.240, 892.815, 803.533, 1147.905, 1506.625,
    1004.417, 1339.222, 1268.737, 1339.222, 2410.600,
    1205.300, 1418.000, 1418.000, 1205.300, 1339.222
])


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, weight=None, balance=0.25):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.balance = balance
        return

    def forward(self, inputs, targets):
        logpt = -F.binary_cross_entropy_with_logits(
            input=inputs, target=targets, weight=self.weight)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance * focal_loss
        return balanced_focal_loss


class FBetaLoss(nn.Module):

    def __init__(self, beta=1):
        super(FBetaLoss, self).__init__()

        self.eps = 1e-8
        self.beta = beta
        self.beta2 = beta ** 2
        return

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        tp = (inputs * targets).sum(dim=1)
        precision = tp.div(inputs.sum(dim=1).add(self.eps))
        recall = tp.div(targets.sum(dim=1).add(self.eps))

        fbeta = torch.mean(
            (precision * recall)
            .div(precision.mul(self.beta2) + recall + self.eps)
            .mul(1 + self.beta2)
        )
        return 1 - fbeta


class ComboLoss(nn.Module):

    def __init__(self, losses, weights):
        super(ComboLoss, self).__init__()

        assert len(losses) == len(weights),\
            'losses and weights should have same length'

        classes_weights = torch.tensor(CLASSES_WEIGHTS).float()
        if torch.cuda.is_available():
            classes_weights = classes_weights.cuda()

        bce_loss = nn.BCEWithLogitsLoss(pos_weight=classes_weights)
        mlsml_loss = nn.MultiLabelSoftMarginLoss(weight=classes_weights)
        focal_loss = FocalLoss(gamma=2, weight=classes_weights)
        f1_loss = FBetaLoss(beta=1)

        self.losses_weights = []
        for loss, weight in zip(losses, weights):
            if loss == 'bce':
                item = [bce_loss, weight]
            elif loss == 'mlsml':
                item = [mlsml_loss, weight]
            elif loss == 'focal':
                item = [focal_loss, weight]
            elif loss == 'f1':
                item = [f1_loss, weight]
            else:
                continue
            self.losses_weights.append(item)
        return

    def forward(self, inputs, targets):
        total_loss = 0
        for loss, weight in self.losses_weights:
            total_loss += weight * loss(inputs, targets)
        return total_loss.clamp(min=1e-5)
