import torch
import torch.nn as nn


class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        eps = 1e-8
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** self.gamma * torch.log(pt)).mean()


class ComboLoss(nn.Module):

    def __init__(self, losses, weights):
        super(ComboLoss, self).__init__()

        assert len(losses) == len(weights),\
            'losses and weights should have same length'

        self.losses_weights = []
        for loss, weight in zip(losses, weights):
            if loss == 'bce':
                self.losses_weights.append([nn.BCEWithLogitsLoss(), weight])
            elif loss == 'mlsml':
                self.losses_weights.append([nn.MultiLabelSoftMarginLoss(), weight])
            elif loss == 'focal':
                self.losses_weights.append([FocalLoss2d(gamma=1), weight])
            else:
                continue
        return

    def forward(self, outputs, targets):
        total_loss = 0
        for loss, weight in self.losses_weights:
            total_loss += weight * loss(outputs, targets)
        return total_loss.clamp(min=1e-5)
