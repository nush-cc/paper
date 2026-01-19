import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score


class HybridDirectionalLoss(nn.Module):
    def __init__(self, delta=10.0, dir_weight=0.1, threshold=0.01):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction='mean')
        self.dir_weight = dir_weight
        self.threshold = threshold  # 設定一個極小值，忽略雜訊

    def forward(self, pred, target, prev_value):
        # 1. 數值誤差 (Huber)
        loss_val = self.huber(pred, target)

        # 2. 計算變化量
        true_delta = target - prev_value
        pred_delta = pred - prev_value

        # 3. 製作遮罩 (Mask): 只有當真實變化夠大時，才在乎方向對不對
        # abs(true_delta) > threshold 的部分才算分
        mask = torch.abs(true_delta) > self.threshold
        
        if mask.sum() > 0:
            # 使用 SoftMarginLoss (標準的方向分類 Loss)
            # target 必須是 1 或 -1 (使用 sign)
            # pred_delta 保持數值，讓模型知道"信心程度"
            true_sign = torch.sign(true_delta[mask])
            pred_val = pred_delta[mask]
            
            # SoftMarginLoss 公式: log(1 + exp(-y * x))
            dir_loss = nn.functional.soft_margin_loss(pred_val, true_sign)
        else:
            dir_loss = torch.tensor(0.0, device=pred.device)

        # 4. 總合
        return (1 - self.dir_weight) * loss_val + self.dir_weight * dir_loss

def get_metrics(targets, preds, last_knowns=None):
    """
    Args:
        targets: 真實值 [N, Horizon]
        preds:   預測值 [N, Horizon]
        last_knowns: 最後一筆已知值 [N, 1] (可選)

    Returns:
        r2, rmse, acc_avg, acc_steps, acc_high
    """

    # 1. R2 & RMSE (整體數值誤差 - Flatten 處理)
    # 這裡維持你的做法，看整體的擬合程度
    r2 = r2_score(targets.flatten(), preds.flatten())
    rmse = np.sqrt(mean_squared_error(targets.flatten(), preds.flatten()))

    # 預設回傳值 (防止 last_knowns 為 None 時崩潰)
    acc_avg = 0.0
    acc_steps = np.zeros(targets.shape[1]) if targets.ndim > 1 else np.array([0.0])
    acc_high = 0.0

    # 2. 方向準確度計算 (包含防呆)
    if last_knowns is not None:
        # 確保維度是 [N, 1]，方便廣播運算
        if last_knowns.ndim == 1:
            last_knowns = last_knowns.reshape(-1, 1)

        # 計算變化量 (相對於最後一個已知點)
        # targets, preds: [N, Horizon]
        # last_knowns:  [N, 1]
        true_delta = targets - last_knowns
        pred_delta = preds - last_knowns

        # Boolean Matrix [N, Horizon]
        # 判斷符號是否相同 (同漲或同跌)
        # 使用 sign 函數比較穩健，0 的情況通常視為 False 或忽略，這裡視為不匹配
        dir_correct = (np.sign(true_delta) == np.sign(pred_delta))

        # A. 總平均準確度
        acc_avg = np.mean(dir_correct)

        # B. 每一步的準確度 (Day 1, Day 2, Day 3...)
        acc_steps = np.mean(dir_correct, axis=0)

        # 3. 高波動準確度 (High Volatility Accuracy)
        # 使用絕對變化量
        magnitude = np.abs(true_delta)

        # 使用 80 分位數作為門檻 (保留你的邏輯)
        # 注意：這裡是指 "所有樣本、所有時間步" 的 80 分位數
        thresh = np.percentile(magnitude, 80)
        high_vol_mask = magnitude > thresh

        if np.sum(high_vol_mask) > 0:
            acc_high = np.mean(dir_correct[high_vol_mask])
        else:
            acc_high = 0.0

    return r2, rmse, acc_avg, acc_steps, acc_high
