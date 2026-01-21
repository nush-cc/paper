import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score


# 保留你的 Loss Function
class HybridDirectionalLoss(nn.Module):
    def __init__(self, delta=1.0, dir_weight=0.5, threshold=0.01):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction='mean')
        self.dir_weight = dir_weight
        self.threshold = threshold

    def forward(self, pred, target, prev_value):
        loss_val = self.huber(pred, target)
        true_delta = target - prev_value
        pred_delta = pred - prev_value
        mask = torch.abs(true_delta) > self.threshold
        
        if mask.sum() > 0:
            true_sign = torch.sign(true_delta[mask])
            pred_val = pred_delta[mask]
            dir_loss = nn.functional.soft_margin_loss(pred_val, true_sign)
        else:
            dir_loss = torch.tensor(0.0, device=pred.device)

        return (1 - self.dir_weight) * loss_val + self.dir_weight * dir_loss

# 保留原本的統計指標計算
def get_metrics(targets, preds, last_knowns=None):
    """
    Returns: r2, rmse, acc_avg, acc_steps, acc_high
    """
    r2 = r2_score(targets.flatten(), preds.flatten())
    rmse = np.sqrt(mean_squared_error(targets.flatten(), preds.flatten()))

    acc_avg = 0.0
    acc_steps = np.zeros(targets.shape[1]) if targets.ndim > 1 else np.array([0.0])
    acc_high = 0.0

    if last_knowns is not None:
        if last_knowns.ndim == 1:
            last_knowns = last_knowns.reshape(-1, 1)

        true_delta = targets - last_knowns
        pred_delta = preds - last_knowns
        dir_correct = (np.sign(true_delta) == np.sign(pred_delta))

        acc_avg = np.mean(dir_correct)
        acc_steps = np.mean(dir_correct, axis=0)

        magnitude = np.abs(true_delta)
        thresh = np.percentile(magnitude, 80)
        high_vol_mask = magnitude > thresh

        if np.sum(high_vol_mask) > 0:
            acc_high = np.mean(dir_correct[high_vol_mask])
        else:
            acc_high = 0.0

    return r2, rmse, acc_avg, acc_steps, acc_high

# 新增：金融回測指標 (計算真正的夏普值)
def calculate_financial_metrics(strategy_returns, benchmark_returns, annual_factor=252):
    """
    計算金融回測指標 (Sharpe, MDD, CumRet)
    """
    # 1. 累積報酬
    strat_cum = np.prod(1 + strategy_returns) - 1
    bench_cum = np.prod(1 + benchmark_returns) - 1
    
    # 2. 夏普值 (Sharpe Ratio)
    strat_mean = np.mean(strategy_returns)
    strat_std = np.std(strategy_returns)
    if strat_std < 1e-9:
        strat_sharpe = 0.0
    else:
        strat_sharpe = (strat_mean / strat_std) * np.sqrt(annual_factor)
        
    bench_mean = np.mean(benchmark_returns)
    bench_std = np.std(benchmark_returns)
    if bench_std < 1e-9:
        bench_sharpe = 0.0
    else:
        bench_sharpe = (bench_mean / bench_std) * np.sqrt(annual_factor)

    # 3. 最大回撤 (Max Drawdown)
    def get_max_drawdown(returns):
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return np.min(drawdown)

    strat_mdd = get_max_drawdown(strategy_returns)
    
    return {
        "Strategy_Sharpe": strat_sharpe,
        "Benchmark_Sharpe": bench_sharpe,
        "Strategy_CumRet": strat_cum,
        "Benchmark_CumRet": bench_cum,
        "Max_Drawdown": strat_mdd
    }