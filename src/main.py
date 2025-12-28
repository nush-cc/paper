"""
E2E-ALR
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent

dataset_path = BASE_DIR / "dataset" / "USD_TWD.csv"
save_path = BASE_DIR / "results" / "main"

if not save_path.exists():
    save_path.mkdir(parents=True, exist_ok=True)

# ==================== Configuration ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
HORIZON = 3
LOOKBACK = 30

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print(f"[Setup] Device: {DEVICE} | Seed: {SEED} | Horizon: {HORIZON}")

# ==================== 1. Components & Model ====================
class LearnableMovingAvg(nn.Module):
    def __init__(self, kernel_size, input_channels):
        super(LearnableMovingAvg, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=input_channels,
            bias=False
        )
        nn.init.constant_(self.conv.weight, 1.0 / kernel_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size, input_channels):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = LearnableMovingAvg(kernel_size, input_channels)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend

class CNNExpert(nn.Module):
    def __init__(self, seq_len, pred_len, input_channels, hidden_dim=64):
        super().__init__()

        #Params needed for tuning
        k_size = 5  # <--- 在這裡修改 Kernel Size (3, 5, 7...)
        dropout_p = 0.2 # <--- 在這裡修改 Dropout (0.1 ~ 0.5)

        # 自動計算 Padding (保證輸入輸出長度一致：30 -> 30)
        # 公式：(kernel_size - 1) // 2
        # 例如：k=3 -> p=1; k=5 -> p=2; k=7 -> p=3
        padding = (k_size - 1) // 2

        self.net = nn.Sequential(
            # 第一層
            nn.Conv1d(input_channels, hidden_dim, kernel_size=k_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p),

            # 第二層 (為了簡化，我們讓第二層也跟著變，或者你可以固定第二層為 k=3)
            # 這裡示範兩層都改大，讓感受野最大化
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=k_size, padding=padding),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p)
        )

        # 只要 Padding 設定正確，長度 seq_len (30) 就不會變
        self.flatten_dim = seq_len * (hidden_dim * 2)
        self.fc = nn.Linear(self.flatten_dim, pred_len)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Channels] -> [Batch, Channels, Seq_Len]
        x = x.permute(0, 2, 1)
        feat = self.net(x)
        # Flatten
        feat = feat.reshape(feat.size(0), -1)
        out = self.fc(feat)
        return out

class EnhancedDLinear(nn.Module):
    def __init__(self, seq_len=30, pred_len=1, input_channels=2):
        super().__init__()

        # 1. Decomposition
        self.decomp = SeriesDecomp(kernel_size=15, input_channels=input_channels)

        # 2. Linear Backbone (DLinear)
        self.linear_trend = nn.Linear(seq_len * input_channels, pred_len)
        self.linear_seasonal = nn.Linear(seq_len * input_channels, pred_len)

        # 3. CNN Booster
        self.cnn_trend = CNNExpert(seq_len, pred_len, input_channels, hidden_dim=32)
        self.cnn_seasonal = CNNExpert(seq_len, pred_len, input_channels, hidden_dim=32)

        # 4. Static Weights (Learnable Scalars)
        # 給予一點點初始偏置，讓它容易學到負值 (阻尼)
        self.trend_gate = nn.Parameter(torch.tensor(-0.01))
        self.seasonal_gate = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        seasonal_part, trend_part = self.decomp(x)
        B, S, C = x.shape

        trend_flat = trend_part.reshape(B, -1)
        seasonal_flat = seasonal_part.reshape(B, -1)

        # Linear Parts
        trend_out_linear = self.linear_trend(trend_flat)
        seasonal_out_linear = self.linear_seasonal(seasonal_flat)

        # CNN Parts
        trend_out_cnn = self.cnn_trend(trend_part)
        seasonal_out_cnn = self.cnn_seasonal(seasonal_part)

        # Fusion
        trend_final = trend_out_linear + (torch.tanh(self.trend_gate) * trend_out_cnn)
        seasonal_final = seasonal_out_linear + (torch.tanh(self.seasonal_gate) * seasonal_out_cnn)

        output = x[:, -1, 0:1] + trend_final + seasonal_final

        # Return weights for logging
        return output, None, torch.stack([self.trend_gate, self.seasonal_gate])

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

class VolatilityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return {"raw_input": self.X[idx], "target": self.y[idx]}

# ==================== 2. Data Preparation ====================
def prepare_data(df, lookback=30, horizon=3, mode='raw'):
    df = df.copy()
    vol_window = 7

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = df["log_return"].rolling(vol_window).std() * np.sqrt(252) * 100
    df = df.dropna().reset_index(drop=True)

    feature_cols = ["Volatility", "log_return"]
    features = df[feature_cols].values

    split_idx = int(len(features) * 0.8)
    train_feat, test_feat = features[:split_idx], features[split_idx:]

    scalers = {}
    train_feat_scaled = np.zeros_like(train_feat)
    test_feat_scaled = np.zeros_like(test_feat)

    for i in range(features.shape[1]):
        s = StandardScaler()
        train_feat_scaled[:, i] = s.fit_transform(train_feat[:, i].reshape(-1, 1)).flatten()
        test_feat_scaled[:, i] = s.transform(test_feat[:, i].reshape(-1, 1)).flatten()
        if i == 0: scalers["target"] = s

    def create_sequences(data, lookback, horizon):
        X, y = [], []
        for i in range(len(data) - lookback - horizon + 1):
            X.append(data[i : i + lookback])
            # y.append(data[i + lookback + horizon - 1, 0])
            y.append(data[i + lookback : i + lookback + horizon, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_feat_scaled, lookback, horizon)
    X_test, y_test = create_sequences(test_feat_scaled, lookback, horizon)

    train_loader = DataLoader(VolatilityDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(VolatilityDataset(X_test, y_test), batch_size=32, shuffle=False)

    return train_loader, test_loader, scalers, X_train, y_train, X_test, y_test

# ==================== 3. Training Function ====================
def train_v11(train_loader, test_loader, num_epochs=120, lr=0.001):
    model = EnhancedDLinear(seq_len=30, pred_len=HORIZON, input_channels=2).to(DEVICE)
    criterion = HybridDirectionalLoss(direction_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print("\n[Training] V11 (Enhanced DLinear)...")
    for epoch in range(num_epochs):
        model.train()
        losses = []
        for batch in train_loader:
            x, y = batch['raw_input'].to(DEVICE), batch['target'].to(DEVICE)
            optimizer.zero_grad()
            prev_val = x[:, -1, 0:1]
            out, _, _ = model(x)
            loss = criterion(out, y, prev_val)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()

        if (epoch+1) % 20 == 0:
            tr_w = torch.tanh(model.trend_gate).item()
            se_w = torch.tanh(model.seasonal_gate).item()
            print(f"  Epoch {epoch+1} | Loss: {np.mean(losses):.4f} | Trend W: {tr_w:.3f} | Seas W: {se_w:.3f}")

    return model

# def get_metrics(targets, preds, last_knowns):
#     r2 = r2_score(targets, preds)
#     rmse = np.sqrt(mean_squared_error(targets, preds))
#
#     true_delta = targets - last_knowns
#     pred_delta = preds - last_knowns
#     dir_correct = (np.sign(true_delta) == np.sign(pred_delta))
#     acc = np.mean(dir_correct)
#
#     magnitude = np.abs(true_delta)
#     thresh = np.percentile(magnitude, 80)
#     high_vol_mask = magnitude > thresh
#     acc_high = np.mean(dir_correct[high_vol_mask]) if np.sum(high_vol_mask)>0 else 0
#     return r2, rmse, acc, acc_high

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
    dir_correct = (np.sign(true_delta) == np.sign(pred_delta)) # Boolean Matrix (N, 3)

    acc_avg = np.mean(dir_correct)       # 總平均
    acc_steps = np.mean(dir_correct, axis=0) # [Day1_Acc, Day2_Acc, Day3_Acc]

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

# ==================== Main Execution ====================
if __name__ == "__main__":
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        exit()

    df = pd.read_csv(dataset_path)

    # 1. Prepare Data
    train_loader, test_loader, scalers_raw, _, _, _, _ = prepare_data(df, lookback=LOOKBACK, horizon=HORIZON)

    # 2. Train Model
    model = train_v11(train_loader, test_loader)

    # 3. Evaluation Loop (Fixed for Sequence)
    model.eval()

    preds_dlinear = []
    preds_hybrid = []
    targets_all = []
    last_knowns_all = []

    trend_w = torch.tanh(model.trend_gate).item()
    seas_w = torch.tanh(model.seasonal_gate).item()

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch['raw_input'].to(DEVICE), batch['target'].to(DEVICE)
            B = x.shape[0]

            # Re-run Forward Pass Manually
            seasonal_part, trend_part = model.decomp(x)

            # Linear -> [B, 3]
            trend_out_linear = model.linear_trend(trend_part.reshape(B, -1))
            seasonal_out_linear = model.linear_seasonal(seasonal_part.reshape(B, -1))

            # CNN -> [B, 3]
            trend_out_cnn = model.cnn_trend(trend_part)
            seasonal_out_cnn = model.cnn_seasonal(seasonal_part)

            last_val = x[:, -1, 0:1] # [B, 1]

            # Pred A: Pure DLinear
            pred_base = last_val + trend_out_linear + seasonal_out_linear

            # Pred B: Hybrid
            correction = (trend_w * trend_out_cnn) + (seas_w * seasonal_out_cnn)
            pred_full = pred_base + correction

            # 儲存結果 (不 Flatten，保留 Bx3 結構)
            preds_dlinear.append(pred_base.cpu().numpy())
            preds_hybrid.append(pred_full.cpu().numpy())
            targets_all.append(y.cpu().numpy())
            last_knowns_all.append(last_val.cpu().numpy())

    # 4. Inverse Transform (修正版)
    scaler = scalers_raw['target']

    def inv_seq(data_list):
        # 1. Stack: List of (B, 3) -> (N, 3)
        data_arr = np.vstack(data_list)
        N, H = data_arr.shape

        # 2. Flatten for Scaler: (N*3, 1)
        data_flat = data_arr.reshape(-1, 1)
        data_inv = scaler.inverse_transform(data_flat)

        # 3. Reshape back: (N, 3)
        return data_inv.reshape(N, H)

    def inv_anchor(data_list):
        data_arr = np.vstack(data_list) # (N, 1)
        return scaler.inverse_transform(data_arr)

    y_true = inv_seq(targets_all)
    y_last = inv_anchor(last_knowns_all)
    y_dlinear = inv_seq(preds_dlinear)
    y_hybrid = inv_seq(preds_hybrid)

    # 5. Metrics & Detailed Report
    # 注意：這裡接收的變數變多了 (acc_steps)
    r2_base, rmse_base, acc_base, step_base, h_acc_base = get_metrics(y_true, y_dlinear, y_last)
    r2_full, rmse_full, acc_full, step_full, h_acc_full = get_metrics(y_true, y_hybrid, y_last)

    print("\n" + "="*95)
    print(f"📊 FINAL SEQUENCE FORECAST (Horizon={HORIZON}): Detailed Breakdown")
    print(f"   (Weights -> Trend: {trend_w:.4f} | Seasonal: {seas_w:.4f})")
    print("="*95)
    print(f"{'Metric':<20} | {'Pure DLinear':<15} | {'Enhanced':<15} | {'Improvement':<15}")
    print("-" * 95)

    # R2 & RMSE
    diff_r2 = r2_full - r2_base
    print(f"{'R2 Score (Overall)':<20} | {r2_base:<15.4f} | {r2_full:<15.4f} | {diff_r2:+.4f} {'✅' if diff_r2>0 else '❌'}")

    diff_rmse = rmse_full - rmse_base
    print(f"{'RMSE (Overall)':<20} | {rmse_base:<15.4f} | {rmse_full:<15.4f} | {diff_rmse:+.4f} {'✅' if diff_rmse<0 else '❌'}")

    print("-" * 95)
    print("🎯 Direction Accuracy (Win Rate)")

    # 總平均
    diff_acc = acc_full - acc_base
    print(f"{'  Average (All Days)':<20} | {acc_base:<15.4f} | {acc_full:<15.4f} | {diff_acc:+.4f} {'✅' if diff_acc>0 else '❌'}")

    # Day-wise Breakdown (這就是你要的序列效果！)
    for i in range(HORIZON):
        d_base = step_base[i]
        d_full = step_full[i]
        diff = d_full - d_base
        day_label = f"  Day {i+1} (T+{i+1})"
        print(f"{day_label:<20} | {d_base:<15.4f} | {d_full:<15.4f} | {diff:+.4f} {'✅' if diff>0 else '❌'}")

    print("-" * 95)

    # High Volatility
    diff_h_acc = h_acc_full - h_acc_base
    print(f"{'High Volatility Acc':<20} | {h_acc_base:<15.4f} | {h_acc_full:<15.4f} | {diff_h_acc:+.4f} {'✅' if diff_h_acc>0 else '❌'}")
    print("="*95)