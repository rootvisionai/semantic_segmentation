"""
reference: https://github.com/yassouali/pytorch-segmentation/blob/master/utils/losses.py
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch_metric_learning import miners, losses
from einops import rearrange


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    # cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=torch.tensor(weight), reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):
        # target = make_one_hot(target, classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        return loss.mean()


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', weight=None):
        super(CrossEntropyDiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50

        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)

    def preprocess(self, pred, mask):
        # pred: bs, c, h, w -> bs*h*w, c
        pred = torch.nn.functional.interpolate(pred, size=(16, 16), mode="nearest")
        mask = torch.nn.functional.interpolate(mask, size=(16, 16), mode="nearest")
        pred = rearrange(pred, 'b c h w -> (b h w) c')
        mask = rearrange(mask, 'b c h w -> (b h w) c')
        mask = mask.argmax(dim=1)
        return pred, mask

    def forward(self, pred, mask):
        pred, mask = self.preprocess(pred, mask)
        hard_pairs = self.miner(pred, mask)
        loss = self.loss_func(pred, mask, hard_pairs)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin, )

    def preprocess(self, pred, mask):
        # pred: bs, c, h, w -> bs*h*w, c
        pred = torch.nn.functional.interpolate(pred, size=(16, 16), mode="nearest")
        mask = torch.nn.functional.interpolate(mask, size=(16, 16), mode="nearest")
        pred = rearrange(pred, 'b c h w -> (b h w) c')
        mask = rearrange(mask, 'b c h w -> (b h w) c')
        mask = mask.argmax(dim=1)
        return pred, mask

    def forward(self, pred, mask):
        pred, mask = self.preprocess(pred, mask)
        loss = self.loss_func(pred, mask)
        return loss