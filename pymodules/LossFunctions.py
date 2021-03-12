import torch
import numpy as np


def dice_score(pred, gt, smooth):
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    intersection = (pred_flat * gt_flat).sum()
    unionset = pred_flat.sum() + gt_flat.sum()
    loss = (2 * (intersection + smooth) / (unionset + smooth))
    return - loss


def weighted_dice(pred, gt):
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
            class_scores.append(gt_frequencies[clazz] * dice_score(class_pred, class_gt, eps))
        sample_scores.append(sum(class_scores) / len(class_scores))
    return sum(sample_scores) / len(sample_scores)


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
        print(gt_frequencies)
        for clazz in range(sample_pred.shape[0]):
            class_pred = sample_pred[clazz]
            class_gt = sample_gt[clazz]
            # calc class-wise dice
            class_scores.append(gt_frequencies[clazz] * class_cross_entropy(class_pred, class_gt))
        sample_scores.append(sum(class_scores) / len(class_scores))
    return sum(sample_scores) / len(sample_scores)
