import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score


class HybridDirectionalLoss(nn.Module):
    def __init__(self, direction_weight=0.5, delta=10.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction='mean')
        self.dir_weight = direction_weight

    def forward(self, pred, target, prev_value):
        loss_val = self.huber(pred, target)
        true_delta = target - prev_value
        pred_delta = pred - prev_value
        sign_agreement = torch.tanh(true_delta * 10) * torch.tanh(pred_delta * 10)
        dir_loss = torch.mean(1 - sign_agreement)
        return (1 - self.dir_weight) * loss_val + self.dir_weight * dir_loss


def get_metrics(targets, preds, last_knowns):
    # targets, preds shape: (Num_Samples, Horizon)
    # last_knowns shape: (Num_Samples, 1)

    if last_knowns.ndim == 1:
        last_knowns = last_knowns.reshape(-1, 1)

    # 1. R2 & RMSE (整體數值誤差)
    r2 = r2_score(targets.flatten(), preds.flatten())
    rmse = np.sqrt(mean_squared_error(targets.flatten(), preds.flatten()))

    # 2. 方向準確度 (細分 Day 1, Day 2, Day 3)
    true_delta = targets - last_knowns
    pred_delta = preds - last_knowns
    dir_correct = (np.sign(true_delta) == np.sign(pred_delta))  # Boolean Matrix (N, 3)

    acc_avg = np.mean(dir_correct)  # 總平均
    acc_steps = np.mean(dir_correct, axis=0)  # [Day1_Acc, Day2_Acc, Day3_Acc]

    # 3. 高波動準確度
    magnitude = np.abs(true_delta)
    thresh = np.percentile(magnitude, 80)
    high_vol_mask = magnitude > thresh

    if np.sum(high_vol_mask) > 0:
        acc_high = np.mean(dir_correct[high_vol_mask])
    else:
        acc_high = 0

    # 多回傳一個 acc_steps
    return r2, rmse, acc_avg, acc_steps, acc_high
