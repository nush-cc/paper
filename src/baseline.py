"""
Baseline Suite for Multi-Step Forecasting (Horizon=3)
=====================================================
Features: Includes V2 enhanced features (MACD, BB_Width) for fair comparison.
Models:
1. Naive (Persistence): Pred(t+H) = Actual(t)
2. XGBoost (ML SOTA)
3. Vanilla LSTM (DL SOTA)

Run: python baseline_horizon.py
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
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
import random

warnings.filterwarnings("ignore")
os.environ["avg_verbose"] = "0"

BASE_DIR = Path(__file__).resolve().parent.parent
dataset_path = BASE_DIR / "dataset" / "GBP_TWD.csv"
save_path = BASE_DIR / "results" / "baseline_horizon_3"

if not save_path.exists():
    save_path.mkdir(parents=True, exist_ok=True)

# 設定 Horizon = 3 (與你的 V2 保持一致)
HORIZON = 3
SEED = 5827
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(SEED)

# ==================== 1. Data Preparation (Fair Fight) ====================
class VolatilityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return {"raw_input": self.X[idx], "target": self.y[idx]}

def prepare_data_fair(df, vol_window=7, lookback=30, horizon=3):
    print(f"[Data] Preparing Horizon={horizon} dataset with Enhanced Features...")
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = df["log_return"].rolling(vol_window).std() * np.sqrt(252) * 100

    # 1. RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    # 2. MACD (Added for fairness)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2

    # 3. BB Width (Added for fairness)
    sma = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Width'] = ((sma + std*2) - (sma - std*2)) / sma

    df = df.dropna().reset_index(drop=True)

    # 5 Features
    features = df[["Volatility", "RSI", "log_return", "MACD", "BB_Width"]].values

    split_idx = int(len(features) * 0.8)
    train_feat, test_feat = features[:split_idx], features[split_idx:]

    train_feat_scaled = np.zeros_like(train_feat)
    test_feat_scaled = np.zeros_like(test_feat)
    target_scaler = None

    for i in range(features.shape[1]):
        s = StandardScaler()
        train_feat_scaled[:, i] = s.fit_transform(train_feat[:, i].reshape(-1, 1)).flatten()
        test_feat_scaled[:, i] = s.transform(test_feat[:, i].reshape(-1, 1)).flatten()
        if i == 0: target_scaler = s

    def create_sequences(data, lookback, horizon):
        X, y = [], []
        # X: [t-L : t]
        # y: [t+H-1] (Future value)
        for i in range(len(data) - lookback - horizon + 1):
            X.append(data[i : i + lookback])
            y.append(data[i + lookback + horizon - 1, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_feat_scaled, lookback, horizon)
    X_test, y_test = create_sequences(test_feat_scaled, lookback, horizon)

    train_loader = DataLoader(VolatilityDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(VolatilityDataset(X_test, y_test), batch_size=32, shuffle=False)

    return train_loader, test_loader, target_scaler, X_train, y_train, X_test, y_test

# ==================== 2. Evaluation Logic ====================
def evaluate_horizon_metrics(name, targets, preds, last_known_vals):
    """
    targets: 真實值 (at T+H)
    preds: 預測值 (for T+H)
    last_known_vals: T 時刻的值 (用來算方向)
    """
    # 1. R2 & RMSE
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)

    # 2. Directional Accuracy
    # 真實變動: Target(T+H) - Known(T)
    true_delta = targets - last_known_vals
    pred_delta = preds - last_known_vals

    # 防止除以0或delta為0的情況
    # 這裡如果不相等，且方向一致才算對
    # 對 Naive 來說，pred_delta 永遠是 0，所以 sign 是 0
    # 除非 true_delta 也是 0，否則 Naive 在這項會拿 0 分 (這很合理，因為它沒預測變動)

    dir_correct = (np.sign(true_delta) == np.sign(pred_delta))

    # 特別處理 Naive: 如果 Naive 預測不變，而市場真的沒變，算對；否則算錯
    # 但通常市場都會變，所以 Naive 會死很慘

    acc_overall = np.mean(dir_correct)

    # 3. High Volatility Accuracy
    magnitude = np.abs(true_delta)
    threshold = np.percentile(magnitude, 80)
    high_vol_mask = magnitude > threshold

    if np.sum(high_vol_mask) > 0:
        acc_high_vol = np.mean(dir_correct[high_vol_mask])
    else:
        acc_high_vol = 0.0

    print(f"   > {name:<15} | RMSE: {rmse:.4f} | R2: {r2:.4f} | Dir: {acc_overall*100:.2f}% | Hi-Vol: {acc_high_vol*100:.2f}%")
    return {
        "Model": name,
        "RMSE": rmse,
        "R2": r2,
        "Dir Acc": acc_overall,
        "High Vol Acc": acc_high_vol
    }

# ==================== 3. Models ====================

# --- Naive ---
def run_naive(y_test, X_test, scaler, horizon):
    # Naive Logic: Pred[T+H] = Known[T]
    # y_test is Actual[T+H]
    # X_test[:, -1, 0] is Known[T] (Scaled)

    targets = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    last_known = scaler.inverse_transform(X_test[:, -1, 0].reshape(-1, 1)).flatten()

    preds = last_known # Naive prediction

    return evaluate_horizon_metrics(f"Naive (Lag-{horizon})", targets, preds, last_known)

# --- XGBoost ---
def run_xgboost(X_train, y_train, X_test, y_test, scaler):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=SEED)
    model.fit(X_train_flat, y_train)
    preds_scaled = model.predict(X_test_flat)

    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    targets = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    last_known = scaler.inverse_transform(X_test[:, -1, 0].reshape(-1, 1)).flatten()

    return evaluate_horizon_metrics("XGBoost", targets, preds, last_known)

# --- LSTM ---
class VanillaLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

def run_lstm(train_loader, test_loader, scaler):
    model = VanillaLSTM(input_size=5).to(DEVICE) # Input size 5 for enhanced features
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        model.train()
        for batch in train_loader:
            x, y = batch['raw_input'].to(DEVICE), batch['target'].to(DEVICE).unsqueeze(-1)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, targets, last_knowns = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch['raw_input'].to(DEVICE), batch['target'].to(DEVICE)
            out = model(x)
            preds.append(out.cpu().numpy())
            targets.append(y.cpu().numpy())
            last_knowns.append(x[:, -1, 0].cpu().numpy()) # Capture last known vol

    preds = scaler.inverse_transform(np.concatenate(preds).flatten().reshape(-1,1)).flatten()
    targets = scaler.inverse_transform(np.concatenate(targets).flatten().reshape(-1,1)).flatten()
    last_known = scaler.inverse_transform(np.concatenate(last_knowns).flatten().reshape(-1,1)).flatten()

    return evaluate_horizon_metrics("Vanilla LSTM", targets, preds, last_known)

# ==================== Main ====================
if __name__ == "__main__":

    # Load and Prepare Data
    df = pd.read_csv(dataset_path)
    train_loader, test_loader, scaler, X_train, y_train, X_test, y_test = prepare_data_fair(df, horizon=HORIZON)

    results = []
    print(f"\n[Running Baselines for Horizon={HORIZON}]...")

    # 1. Naive
    results.append(run_naive(y_test, X_test, scaler, HORIZON))

    # 2. XGBoost
    results.append(run_xgboost(X_train, y_train, X_test, y_test, scaler))

    # 3. LSTM
    results.append(run_lstm(train_loader, test_loader, scaler))

    # =======================================================
    # ★★★ 填入你的 V2 模型數據 (從 new_main_v2.py 跑出來的) ★★★
    # =======================================================
    results.append({
        "Model": "E2E-FAR-MoE (V2)",
        "RMSE": 2.9285,       # <-- 填入你的 V2 RMSE
        "R2": 0.5618,         # <-- 填入你的 V2 R2
        "Dir Acc": 0.6690,    # <-- 填入你的 V2 Dir Acc
        "High Vol Acc": 0.8007 # <-- 填入你的 V2 High Vol Acc
    })

    # Print Table
    df_res = pd.DataFrame(results)
    print("\n" + "="*80)
    print(f"🏆 HORIZON={HORIZON} FINAL SHOWDOWN")
    print("="*80)
    print(df_res.to_string(index=False, float_format="%.4f"))

    # Plot
    metrics = ['R2', 'High Vol Acc']
    ax = df_res.set_index('Model')[metrics].plot(kind='bar', figsize=(10, 6), rot=45,
                                                 color=['#A0A0A0', '#845EC2'], edgecolor='black')
    plt.title(f"Horizon={HORIZON}: R2 vs High Volatility Accuracy")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(save_path / "final_benchmark_horizon3.png")
    print("\n✓ Comparison plot saved.")