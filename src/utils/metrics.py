import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score


class HybridDirectionalLoss(nn.Module):
    def __init__(self, delta=1.0, dir_weight=0.5, threshold=0.01):
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

# class HybridDirectionalLoss(nn.Module):
#     def __init__(self, delta=1.0, dir_weight=0.5, threshold=0.01, tanh_scale=5.0):
#         """
#         Args:
#             delta: Huber Loss 的閾值
#             dir_weight: 方向性 Loss 的權重
#             threshold: 忽略微小變化的門檻
#             tanh_scale: Tanh 的放大倍率 (溫度係數)。
#                         因為 Tanh 在 0 附近很平緩，如果 pred_delta 很小 (如 0.05)，
#                         tanh(0.05) ~= 0.05，還不夠接近 1 或 -1。
#                         乘上 scale (如 5) 變成 tanh(0.25)，能強迫模型表態。
#         """
#         super().__init__()
#         self.huber = nn.HuberLoss(delta=delta, reduction='mean')
#         self.dir_weight = dir_weight
#         self.threshold = threshold
#         self.tanh_scale = tanh_scale  # <--- 新增這個參數

#     def forward(self, pred, target, prev_value):
#         # 1. 數值誤差 (Huber) - 負責 "準不準"
#         loss_val = self.huber(pred, target)

#         # 2. 計算變化量
#         true_delta = target - prev_value
#         pred_delta = pred - prev_value

#         # 3. 製作遮罩 (Mask)
#         mask = torch.abs(true_delta) > self.threshold
        
#         if mask.sum() > 0:
#             # 取出需要計算方向的樣本
#             true_sign = torch.sign(true_delta[mask]) # +1 或 -1
#             pred_delta_masked = pred_delta[mask]

#             # --- [修改重點] 使用 Tanh 模擬 Soft Sign ---
            
#             # A. 使用 Tanh 壓縮預測值到 [-1, 1]
#             # 乘上 scale 是為了讓靠近 0 的微小變化也能被放大成明確的方向
#             pred_sign_approx = torch.tanh(pred_delta_masked * self.tanh_scale)

#             # B. 計算方向一致性
#             # 如果同號: 1 * 1 = 1 或 -1 * -1 = 1 -> Loss = 1 - 1 = 0
#             # 如果異號: 1 * -1 = -1 -> Loss = 1 - (-1) = 2
#             dir_loss = torch.mean(1 - (pred_sign_approx * true_sign))
            
#         else:
#             dir_loss = torch.tensor(0.0, device=pred.device)

#         # 4. 總合
#         return (1 - self.dir_weight) * loss_val + self.dir_weight * dir_loss

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


def get_sharpe_ratio(targets, preds, last_knowns, annual_factor=252):
    """
    計算基於方向預測的模擬交易夏普值 (日資料預設 252)
    """
    if last_knowns is None:
        return 0.0
    
    if last_knowns.ndim == 1:
        last_knowns = last_knowns.reshape(-1, 1)

    # 1. 取得預測方向與實際報酬率
    # 我們以 Horizon 的第一步 (t+1) 作為交易基準，這是最實務的做法
    true_delta_pct = (targets[:, 0] - last_knowns[:, 0]) / last_knowns[:, 0]
    pred_direction = np.sign(preds[:, 0] - last_knowns[:, 0])

    # 2. 計算策略日報酬
    # 方向正確得正報酬，方向錯誤得負報酬
    strategy_returns = pred_direction * true_delta_pct

    # 3. 計算 Sharpe Ratio
    avg_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)

    if std_ret < 1e-9: # 防止除以 0
        return 0.0

    sharpe = (avg_ret / std_ret) * np.sqrt(annual_factor)
    return sharpe