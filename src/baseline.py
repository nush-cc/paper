"""
The ALL-STAR Arena: E2E-ALR vs. The World (10 Models)
====================================================================
Objective: Benchmark E2E-ALR against Stats, ML, and DL models.
Sorting:   Ranked by R2 Score (Best to Worst).

Competitors:
[Stats] Naive, SMA, Linear Regression
[ML]    SVR, Random Forest, XGBoost, LightGBM (Full Features)
[DL]    LSTM, Transformer
[Ours]  E2E-ALR Framework
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 🔴 請在此填入您 E2E-ALR Framework (V11) 的最佳成績
# ==============================================================================
MY_MODEL_NAME = "E2E-ALR (Ours)"
MY_R2         = 0.6594  # 您提供的數據
MY_RMSE       = 2.0784
MY_ACC        = 0.7262
MY_H_ACC      = 0.8973
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
dataset_path = BASE_DIR / "dataset" / "GBP_TWD.csv"
LOOKBACK = 30
HORIZON = 3

# ==================== DL Models Definitions ====================
class LSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, output_dim=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=32, nhead=2, num_layers=1, output_dim=1):
        super(SimpleTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])

# ==================== Helper Functions ====================
def train_dl_model(model, X_train, y_train, epochs=60, lr=0.001):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by.unsqueeze(-1))
            loss.backward()
            optimizer.step()
    return model

def predict_dl(model, X_test):
    model.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(X_test).to(DEVICE)
        preds = model(inputs)
    return preds.cpu().numpy().flatten()

def prepare_data_comparison(df):
    df = df.copy()
    vol_window = 7
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = df["log_return"].rolling(vol_window).std() * np.sqrt(252) * 100

    # 技術指標 (給 ML 模型作弊用)
    df["RSI"] = df["log_return"].rolling(14).mean().fillna(0)
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    sma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_Width'] = ((sma + std*2) - (sma - std*2)) / sma

    df = df.dropna().reset_index(drop=True)

    # 定義特徵集
    raw_cols = ["Volatility", "log_return"]
    full_cols = ["Volatility", "log_return", "RSI", "MACD", "BB_Width"]

    split_idx = int(len(df) * 0.8)

    # Scaling
    scaler_raw = StandardScaler()
    data_raw = scaler_raw.fit_transform(df[raw_cols].values)

    scaler_full = StandardScaler()
    data_full = scaler_full.fit_transform(df[full_cols].values)

    target_scaler = StandardScaler()
    target_scaler.fit(df["Volatility"].values[:split_idx].reshape(-1, 1))

    # Helper Sequence
    def create_seq(data, label_data):
        X, y = [], []
        for i in range(len(data) - LOOKBACK - HORIZON + 1):
            X.append(data[i:i+LOOKBACK])
            y.append(label_data[i+LOOKBACK+HORIZON-1])
        return np.array(X), np.array(y)

    y_labels = data_raw[:, 0] # Volatility is always col 0

    # Raw Data (For DL, Linear, Stats)
    X_raw, y_all = create_seq(data_raw, y_labels)
    X_tr_r, X_te_r = X_raw[:split_idx], X_raw[split_idx:]
    y_tr, y_te = y_all[:split_idx], y_all[split_idx:]

    # Full Data (For ML)
    X_full, _ = create_seq(data_full, y_labels)
    X_tr_f, X_te_f = X_full[:split_idx], X_full[split_idx:]

    return (X_tr_r, X_te_r), (X_tr_f, X_te_f), y_tr, y_te, target_scaler

def get_metrics(targets, preds, last_knowns):
    r2 = r2_score(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))

    # Direction Accuracy
    # Note: For Naive, preds == last_knowns, so sign(0) is 0.
    # Accuracy will be 0. This is mathematically correct (Persistence predicts no change).
    dir_correct = (np.sign(targets - last_knowns) == np.sign(preds - last_knowns))
    acc = np.mean(dir_correct)

    # High Vol Acc
    mag = np.abs(targets - last_knowns)
    thresh = np.percentile(mag, 80)
    high_vol_mask = mag > thresh
    h_acc = np.mean(dir_correct[high_vol_mask]) if np.sum(high_vol_mask)>0 else 0

    return r2, rmse, acc, h_acc

# ==================== Main ====================
if __name__ == "__main__":
    print("🚀 Initiating ALL-STAR Benchmark Arena...")
    df = pd.read_csv(dataset_path)
    (X_tr_r, X_te_r), (X_tr_f, X_te_f), y_tr, y_te, scaler = prepare_data_comparison(df)

    # Flatten Inputs for ML models
    X_tr_r_flat = X_tr_r.reshape(X_tr_r.shape[0], -1)
    X_te_r_flat = X_te_r.reshape(X_te_r.shape[0], -1)
    X_tr_f_flat = X_tr_f.reshape(X_tr_f.shape[0], -1)
    X_te_f_flat = X_te_f.reshape(X_te_f.shape[0], -1)

    def inv(d): return scaler.inverse_transform(d.reshape(-1, 1)).flatten()
    y_true = inv(y_te)
    last_known = inv(X_te_r[:, -1, 0])

    results = []

    # --- 1. Statistical Baselines ---
    print("Running Stats Models (Naive, SMA, Linear)...")
    # Naive
    metrics = get_metrics(y_true, last_known, last_known)
    results.append(["Naive (Persistence)", "Stats", *metrics])

    # SMA
    sma = np.mean(X_te_r[:, :, 0], axis=1)
    metrics = get_metrics(y_true, inv(sma), last_known)
    results.append(["SMA (Moving Avg)", "Stats", *metrics])

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_tr_r_flat, y_tr)
    y_pred = inv(lr.predict(X_te_r_flat))
    metrics = get_metrics(y_true, y_pred, last_known)
    results.append(["Linear Regression", "Stats", *metrics])

    # --- 2. Machine Learning (Full Features) ---
    print("Running ML Models (XGB, LGBM, RF, SVR)...")

    # SVR
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr.fit(X_tr_f_flat, y_tr)
    y_pred = inv(svr.predict(X_te_f_flat))
    metrics = get_metrics(y_true, y_pred, last_known)
    results.append(["SVR (RBF)", "ML", *metrics])

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_tr_f_flat, y_tr)
    y_pred = inv(rf.predict(X_te_f_flat))
    metrics = get_metrics(y_true, y_pred, last_known)
    results.append(["Random Forest", "ML", *metrics])

    # XGBoost
    xgb_m = xgb.XGBRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    xgb_m.fit(X_tr_f_flat, y_tr)
    y_pred = inv(xgb_m.predict(X_te_f_flat))
    metrics = get_metrics(y_true, y_pred, last_known)
    results.append(["XGBoost", "ML", *metrics])

    # # LightGBM
    # lgb_m = lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)
    # lgb_m.fit(X_tr_f_flat, y_tr)
    # y_pred = inv(lgb_m.predict(X_te_f_flat))
    # metrics = get_metrics(y_true, y_pred, last_known)
    # results.append(["LightGBM", "ML", *metrics])

    # --- 3. Deep Learning ---
    print("Running DL Models (LSTM, Transformer)...")

    # LSTM
    lstm = LSTMModel(input_dim=2)
    lstm = train_dl_model(lstm, X_tr_r, y_tr)
    y_pred = inv(predict_dl(lstm, X_te_r))
    metrics = get_metrics(y_true, y_pred, last_known)
    results.append(["LSTM", "DL", *metrics])

    # Transformer
    tf = SimpleTransformer(input_dim=2)
    tf = train_dl_model(tf, X_tr_r, y_tr)
    y_pred = inv(predict_dl(tf, X_te_r))
    metrics = get_metrics(y_true, y_pred, last_known)
    results.append(["Transformer", "DL", *metrics])

    # --- 4. OUR CHAMPION ---
    results.append([MY_MODEL_NAME, "Ours", MY_R2, MY_RMSE, MY_ACC, MY_H_ACC])

    # --- Reporting & Sorting ---
    cols = ["Model", "Type", "R2 Score", "RMSE", "Dir Acc", "High Vol Acc"]
    # 關鍵：這裡按照 R2 Score 由大到小排序
    res_df = pd.DataFrame(results, columns=cols).sort_values(by="R2 Score", ascending=False)

    print("\n" + "="*95)
    print("🏆 GRAND UNIFIED BENCHMARK: E2E-ALR vs. EVERYONE 🏆")
    print("="*95)
    print(res_df.to_string(index=False, float_format="%.4f"))
    print("="*95)

    # Plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    # 畫圖時也使用排序過的 DataFrame
    sns.barplot(data=res_df, y="Model", x="R2 Score", hue="Type", dodge=False, palette="Spectral")
    plt.title("R2 Score Comparison (Ranked Best to Worst)")
    plt.xlim(0, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()