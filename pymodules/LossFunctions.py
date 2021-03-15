import torch
import numpy as np


def dice_score(gt, pred, smooth):
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    intersection = (pred_flat * gt_flat).sum()
    unionset = pred_flat.sum() + gt_flat.sum()
    loss = ((2 * intersection + smooth)) / ((unionset + smooth))
    return 1 - loss


def tversky(y_true, y_pred, smooth=0.0001, alpha=0.8):
    y_true_pos = y_true.flatten()
    y_pred_pos = y_pred.flatten()
    true_pos = (y_true_pos * y_pred_pos).sum()
    false_neg = (y_true_pos * (1 - y_pred_pos)).sum()
    false_pos = ((1 - y_true_pos) * y_pred_pos).sum()
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.9):
    tv = tversky(y_true, y_pred)
    return torch.pow((1 - tv), gamma)


# 0 = Normal dice
# 1 = tversky
# 2 = focal tversky
def dice(gt, pred, smooth=1., mode=1):
    sample_scores = []
    divider = 0
    for sample in range(pred.shape[0]):
        sample_pred = pred[sample]
        sample_gt = gt[sample]
        class_scores = []
        weights = [1, 1, 1, 1]
        for clazz in range(sample_pred.shape[0]):
            class_pred = sample_pred[clazz]
            class_gt = sample_gt[clazz]
            # calc class-wise dice
            score = 1.
            if mode == 0:
                score = dice_score(class_gt, class_pred, smooth)
            if mode == 1:
                score = tversky_loss(class_gt, class_pred)
            if mode == 2:
                score = focal_tversky_loss(class_gt, class_pred)
            class_scores.append(weights[clazz] * score)
        sample_scores.append((sum(class_scores) / sum(weights)))
    score = sum(sample_scores) / len(sample_scores)
    if torch.is_nonzero(score):
        return score
    return torch.tensor(1.)


def class_cross_entropy(pred, gt):
    eps = 1e-7
    pred_log = -torch.log(torch.clip(pred, eps, 1 - eps))
    pred_log_flat = pred_log.flatten()
    gt_flat = gt.flatten()
    return (pred_log_flat * gt_flat).sum()


def weighted_cross_entropy(pred, gt):
    sample_scores = []
    eps = 1e-10
    for sample in range(pred.shape[0]):
        sample_pred = pred[sample]
        sample_gt = gt[sample]
        gt_frequencies = sample_gt.view(sample_gt.size(0), -1).sum(1)
        gt_frequencies = gt_frequencies / (sample_pred.shape[1] * sample_pred.shape[2])
        gt_frequencies = 1 - gt_frequencies
        class_scores = []
        for clazz in range(sample_pred.shape[0]):
            class_pred = sample_pred[clazz]
            class_gt = sample_gt[clazz]
            # calc class-wise dice
            class_scores.append(gt_frequencies[clazz] * class_cross_entropy(class_pred, class_gt))
        sample_scores.append(sum(class_scores) / len(class_scores))
    return sum(sample_scores) / len(sample_scores)
