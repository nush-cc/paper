"""
V11 E2E Challenge: Deep Learning (Raw) vs XGBoost (Raw vs Full)
===============================================================
Hypothesis:
1. V11 (Raw) > XGBoost (Raw) -> Proves CNN architecture value (Feature Extraction).
2. V11 (Raw) ~ XGBoost (Full) -> The "Holy Grail". Proves E2E learning capability.

Structure:
- Model: V11 Hybrid CNN-MoE + Learnable Decomp + Wide
- Input: ONLY Volatility & Log_Return (2 channels)
- Comparison: XGBoost (Raw) and XGBoost (MACD/RSI/etc.)
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
import xgboost as xgb

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent

dataset_path = BASE_DIR / "dataset" / "GBP_TWD.csv"
save_path = BASE_DIR / "results" / "horizon_3_e2e_challenge"

if not save_path.exists():
    save_path.mkdir(parents=True, exist_ok=True)

# ==================== Configuration ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 5827
HORIZON = 3
LOOKBACK = 30

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print(f"[Setup] Device: {DEVICE} | Seed: {SEED} | Horizon: {HORIZON}")

# ==================== 1. V11 Architecture (Raw Input Version) ====================
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
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.flatten_dim = seq_len * (hidden_dim * 2)
        self.fc = nn.Linear(self.flatten_dim, pred_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        feat = self.net(x)
        feat = feat.reshape(feat.size(0), -1)
        out = self.fc(feat)
        return out

class WideLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x_last):
        return self.linear(x_last)

class HybridCNNMoE(nn.Module):
    def __init__(self, seq_len=30, pred_len=1, input_channels=2): # Input=2 (Raw Only)
        super().__init__()

        self.decomp = SeriesDecomp(kernel_size=15, input_channels=input_channels)

        self.expert_trend_1 = CNNExpert(seq_len, pred_len, input_channels)
        self.expert_trend_2 = CNNExpert(seq_len, pred_len, input_channels)
        self.expert_cyclic_1 = CNNExpert(seq_len, pred_len, input_channels)
        self.expert_cyclic_2 = CNNExpert(seq_len, pred_len, input_channels)

        self.base_linear = nn.Linear(seq_len * input_channels, pred_len)
        self.wide = WideLayer(input_channels)

        self.gating_trend = nn.Sequential(nn.Linear(input_channels * 3, 64), nn.LeakyReLU(0.1), nn.Linear(64, 2))
        self.gating_cyclic = nn.Sequential(nn.Linear(input_channels * 3, 64), nn.LeakyReLU(0.1), nn.Linear(64, 2))
        self.meta_gate = nn.Sequential(nn.Linear(input_channels * 3, 32), nn.LeakyReLU(0.1), nn.Linear(32, 1))
        nn.init.constant_(self.meta_gate[-1].bias, -0.5)

    def forward(self, x):
        seasonal_part, trend_part = self.decomp(x)
        B, S, C = x.shape

        base_delta = self.base_linear(x.reshape(B, -1))
        wide_delta = self.wide(x[:, -1, :])

        t1 = self.expert_trend_1(trend_part)
        t2 = self.expert_trend_2(trend_part)
        c1 = self.expert_cyclic_1(seasonal_part)
        c2 = self.expert_cyclic_2(seasonal_part)

        ctx_mean = x.mean(dim=1)
        ctx_std = x.std(dim=1)
        ctx_last = x[:, -1, :]
        context = torch.cat([ctx_mean, ctx_std, ctx_last], dim=1)

        w_trend = torch.softmax(self.gating_trend(context), dim=1)
        w_cyclic = torch.softmax(self.gating_cyclic(context), dim=1)

        trend_final = (w_trend[:, 0:1] * t1) + (w_trend[:, 1:2] * t2)
        cyclic_final = (w_cyclic[:, 0:1] * c1) + (w_cyclic[:, 1:2] * c2)

        moe_delta = trend_final + cyclic_final
        alpha = torch.sigmoid(self.meta_gate(context))

        total_delta = base_delta + wide_delta + (alpha * moe_delta)
        output = x[:, -1, 0:1] + total_delta
        aux_output = x[:, -1, 0:1] + base_delta + wide_delta

        return output, aux_output

class HybridDirectionalLoss(nn.Module):
    def __init__(self, direction_weight=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.dir_weight = direction_weight

    def forward(self, pred, target, prev_value):
        loss_val = self.mse(pred, target)
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
# ==================== 2. Data Preparation (Fixed Alignment) ====================
def prepare_data(df, lookback=30, horizon=3, mode='raw'):
    """
    Unified Data Prep: Always calculate FULL features first to ensure alignment,
    then select columns based on mode.
    """
    df = df.copy()
    vol_window = 7

    # 1. Always calculate ALL features first (to ensure same dropna behavior)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = df["log_return"].rolling(vol_window).std() * np.sqrt(252) * 100

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2

    sma = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Width'] = ((sma + std*2) - (sma - std*2)) / sma

    # 2. Drop NA (This aligns the rows for both modes)
    df = df.dropna().reset_index(drop=True)

    # 3. Select Columns based on Mode
    if mode == 'full':
        feature_cols = ["Volatility", "RSI", "log_return", "MACD", "BB_Width"]
    else: # mode == 'raw'
        feature_cols = ["Volatility", "log_return"]

    features = df[feature_cols].values

    # 4. Split & Scale
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
            y.append(data[i + lookback + horizon - 1, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_feat_scaled, lookback, horizon)
    X_test, y_test = create_sequences(test_feat_scaled, lookback, horizon)

    # 5. Return
    if mode == 'raw':
        # Create Loaders only for Raw mode (Deep Learning)
        train_loader = DataLoader(VolatilityDataset(X_train, y_train), batch_size=32, shuffle=True)
        test_loader = DataLoader(VolatilityDataset(X_test, y_test), batch_size=32, shuffle=False)
        return train_loader, test_loader, scalers, X_train, y_train, X_test, y_test
    else:
        # For XGBoost Full, we just need the numpy arrays
        return None, None, scalers, X_train, y_train, X_test, y_test

# ==================== 3. Training & Evaluation Functions ====================
def train_v11(train_loader, test_loader, num_epochs=120, lr=0.001):
    model = HybridCNNMoE(seq_len=30, pred_len=1, input_channels=2).to(DEVICE)
    criterion = HybridDirectionalLoss(direction_weight=0.5)
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print("\n[Training] V11 (Raw Data)...")
    for epoch in range(num_epochs):
        model.train()
        losses = []
        aux_weight = max(0.1, 1.0 - (epoch / (num_epochs * 0.8)))

        for batch in train_loader:
            x, y = batch['raw_input'].to(DEVICE), batch['target'].to(DEVICE).unsqueeze(-1)
            optimizer.zero_grad()
            prev_val = x[:, -1, 0:1]
            out, aux_out = model(x)
            loss = criterion(out, y, prev_val) + (aux_weight * mse(aux_out, y))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        if (epoch+1) % 40 == 0:
            print(f"  Epoch {epoch+1} | Loss: {np.mean(losses):.4f}")

    return model

def run_xgboost(X_train, y_train, X_test, y_test, scaler):
    # Flatten for XGBoost
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=SEED)
    model.fit(X_train_flat, y_train)
    preds_scaled = model.predict(X_test_flat)
    return scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

def get_metrics(targets, preds, last_knowns):
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    true_delta = targets - last_knowns
    pred_delta = preds - last_knowns
    dir_correct = (np.sign(true_delta) == np.sign(pred_delta))
    acc = np.mean(dir_correct)
    magnitude = np.abs(true_delta)
    thresh = np.percentile(magnitude, 80)
    high_vol_mask = magnitude > thresh
    acc_high = np.mean(dir_correct[high_vol_mask]) if np.sum(high_vol_mask)>0 else 0
    return r2, acc_high, acc

# ==================== Main Execution ====================
if __name__ == "__main__":
    df = pd.read_csv(dataset_path)

    # -------------------------------------------------------
    # 1. Prepare Data for V11 & XGBoost (Raw)
    # -------------------------------------------------------
    train_loader, test_loader, scalers_raw, X_train_raw, y_train_raw, X_test_raw, y_test_raw = \
        prepare_data(df, lookback=LOOKBACK, horizon=HORIZON, mode='raw')

    # -------------------------------------------------------
    # 2. Train V11 (Raw)
    # -------------------------------------------------------
    model = train_v11(train_loader, test_loader)

    # Evaluate V11 with Nuclear Fix
    model.eval()
    preds_v11, targets_v11, last_knowns_v11 = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch['raw_input'].to(DEVICE), batch['target'].to(DEVICE)
            out, _ = model(x)
            p, t, l = out.detach().cpu().numpy(), y.cpu().numpy(), x[:, -1, 0].cpu().numpy()

            bs = x.shape[0]
            if p.ndim > 2: p = p[:, :, 0]
            preds_v11.append(p.flatten()[:bs].reshape(bs, 1))
            targets_v11.append(t.flatten()[:bs].reshape(bs, 1))
            last_knowns_v11.append(l.flatten()[:bs].reshape(bs, 1))

    preds_v11 = scalers_raw['target'].inverse_transform(np.concatenate(preds_v11, axis=0)).flatten()
    targets_v11 = scalers_raw['target'].inverse_transform(np.concatenate(targets_v11, axis=0)).flatten()
    last_knowns_v11 = scalers_raw['target'].inverse_transform(np.concatenate(last_knowns_v11, axis=0)).flatten()

    r2_v11, h_acc_v11, acc_v11 = get_metrics(targets_v11, preds_v11, last_knowns_v11)

    # -------------------------------------------------------
    # 3. XGBoost Baseline (Raw Data)
    # -------------------------------------------------------
    print("\n[Baseline] Running XGBoost (Raw Data)...")
    preds_xgb_raw = run_xgboost(X_train_raw, y_train_raw, X_test_raw, y_test_raw, scalers_raw['target'])
    r2_xgb_raw, h_acc_xgb_raw, acc_xgb_raw = get_metrics(targets_v11, preds_xgb_raw, last_knowns_v11)

    # -------------------------------------------------------
    # 4. XGBoost Baseline (Full Features: MACD, RSI...)
    # -------------------------------------------------------
    print("[Baseline] Running XGBoost (Full Features: MACD, RSI...)...")
    _, _, scalers_full, X_train_full, y_train_full, X_test_full, y_test_full = \
        prepare_data(df, lookback=LOOKBACK, horizon=HORIZON, mode='full')

    preds_xgb_full = run_xgboost(X_train_full, y_train_full, X_test_full, y_test_full, scalers_full['target'])
    r2_xgb_full, h_acc_xgb_full, acc_xgb_full = get_metrics(targets_v11, preds_xgb_full, last_knowns_v11)

    # -------------------------------------------------------
    # 5. Final Report
    # -------------------------------------------------------
    print("\n" + "="*75)
    print(f"📊 FINAL SHOWDOWN: V11 (Deep Learning) vs XGBoost (Machine Learning)")
    print("="*75)
    print(f"{'Model':<20} | {'Inputs':<15} | {'R2 Score':<10} | {'High Vol Acc':<12} | {'Dir Acc':<10}")
    print("-" * 75)
    print(f"{'XGBoost (Weak)':<20} | {'Raw Only':<15} | {r2_xgb_raw:<10.4f} | {h_acc_xgb_raw:<12.4f} | {acc_xgb_raw:<10.4f}")
    print(f"{'XGBoost (Strong)':<20} | {'Full Feats':<15} | {r2_xgb_full:<10.4f} | {h_acc_xgb_full:<12.4f} | {acc_xgb_full:<10.4f}")
    print("-" * 75)
    print(f"{'V11 CNN-MoE':<20} | {'Raw Only':<15} | {r2_v11:<10.4f} | {h_acc_v11:<12.4f} | {acc_v11:<10.4f}")
    print("="*75)

    if r2_v11 > r2_xgb_raw:
        print("✅ CONCLUSION: V11 beats XGBoost (Raw). Deep Learning successfully extracts features!")
    else:
        print("⚠️ CONCLUSION: V11 needs more tuning to beat XGBoost (Raw).")

    if r2_v11 > r2_xgb_full:
        print("🏆 HOLY GRAIL: V11 (Raw) beats XGBoost (Full). E2E Learning is Superior!")
    elif r2_v11 > (r2_xgb_full - 0.02):
        print("🥈 EXCELLENT: V11 (Raw) is competitive with XGBoost (Full). Feature Engineering is automated.")