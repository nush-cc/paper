"""
The ALL-STAR Arena: E2E-ALR vs. The World (Sequence Version - Detailed Breakdown)
=================================================================================
Objective: Benchmark E2E-ALR against Stats, ML, and DL models on 3-day sequences.
Sorting:   Ranked by R2 Score.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 🔴 請在此填入您 E2E-ALR Framework (V11) 的詳細成績
# (參考您剛剛跑出來的 "Direction Accuracy (Win Rate)" 表格)
# ==============================================================================
MY_MODEL_NAME = "E2E-ALR (Ours)"

# 這裡填入您剛剛跑出的數值
MY_R2         = 0.7868
MY_RMSE       = 1.6442
MY_AVG_ACC    = 0.7242  # Average (All Days)
MY_DAY1_ACC   = 0.7186  # Day 1
MY_DAY2_ACC   = 0.7209  # Day 2
MY_DAY3_ACC   = 0.7331  # Day 3
MY_H_ACC      = 0.8758  # High Volatility Acc
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
dataset_path = BASE_DIR / "dataset" / "USD_TWD.csv"
LOOKBACK = 30
HORIZON = 3

# ==================== DL Models Definitions ====================
class LSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, output_dim=HORIZON):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=32, nhead=2, num_layers=1, output_dim=HORIZON):
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
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    return model

def predict_dl(model, X_test):
    model.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(X_test).to(DEVICE)
        preds = model(inputs)
    return preds.cpu().numpy()

def prepare_data_comparison(df):
    df = df.copy()
    vol_window = 7
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = df["log_return"].rolling(vol_window).std() * np.sqrt(252) * 100

    # 技術指標
    df["RSI"] = df["log_return"].rolling(14).mean().fillna(0)
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    sma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_Width'] = ((sma + std*2) - (sma - std*2)) / sma

    df = df.dropna().reset_index(drop=True)

    raw_cols = ["Volatility", "log_return"]
    full_cols = ["Volatility", "log_return", "RSI", "MACD", "BB_Width"]

    split_idx = int(len(df) * 0.8)

    scaler_raw = StandardScaler()
    data_raw = scaler_raw.fit_transform(df[raw_cols].values)

    scaler_full = StandardScaler()
    data_full = scaler_full.fit_transform(df[full_cols].values)

    target_scaler = StandardScaler()
    target_scaler.fit(df["Volatility"].values[:split_idx].reshape(-1, 1))

    def create_seq(data, label_data):
        X, y = [], []
        for i in range(len(data) - LOOKBACK - HORIZON + 1):
            X.append(data[i:i+LOOKBACK])
            y.append(label_data[i+LOOKBACK : i+LOOKBACK+HORIZON])
        return np.array(X), np.array(y)

    y_labels = data_raw[:, 0]

    X_raw, y_all = create_seq(data_raw, y_labels)
    X_tr_r, X_te_r = X_raw[:split_idx], X_raw[split_idx:]
    y_tr, y_te = y_all[:split_idx], y_all[split_idx:]

    X_full, _ = create_seq(data_full, y_labels)
    X_tr_f, X_te_f = X_full[:split_idx], X_full[split_idx:]

    return (X_tr_r, X_te_r), (X_tr_f, X_te_f), y_tr, y_te, target_scaler

def get_metrics_detailed(targets, preds, last_knowns):
    # targets, preds: (N, 3)
    # last_knowns: (N, 1)

    if last_knowns.ndim == 1:
        last_knowns = last_knowns.reshape(-1, 1)

    # R2 & RMSE
    r2 = r2_score(targets.flatten(), preds.flatten())
    rmse = np.sqrt(mean_squared_error(targets.flatten(), preds.flatten()))

    # Direction Accuracy Matrix
    true_delta = targets - last_knowns
    pred_delta = preds - last_knowns
    dir_correct = (np.sign(true_delta) == np.sign(pred_delta)) # (N, 3) boolean

    # 1. Average Acc
    acc_avg = np.mean(dir_correct)

    # 2. Step-wise Acc (Day 1, Day 2, Day 3)
    acc_steps = np.mean(dir_correct, axis=0) # [acc_d1, acc_d2, acc_d3]

    # 3. High Vol Acc
    magnitude = np.abs(true_delta)
    thresh = np.percentile(magnitude, 80)
    high_vol_mask = magnitude > thresh
    h_acc = np.mean(dir_correct[high_vol_mask]) if np.sum(high_vol_mask)>0 else 0

    return r2, rmse, acc_avg, acc_steps, h_acc

# ==================== Main ====================
if __name__ == "__main__":
    print(f"🚀 Initiating ALL-STAR Benchmark Arena (Horizon={HORIZON})...")
    df = pd.read_csv(dataset_path)
    (X_tr_r, X_te_r), (X_tr_f, X_te_f), y_tr, y_te, scaler = prepare_data_comparison(df)

    X_tr_r_flat = X_tr_r.reshape(X_tr_r.shape[0], -1)
    X_te_r_flat = X_te_r.reshape(X_te_r.shape[0], -1)
    X_tr_f_flat = X_tr_f.reshape(X_tr_f.shape[0], -1)
    X_te_f_flat = X_te_f.reshape(X_te_f.shape[0], -1)

    def inv_seq(d):
        N, H = d.shape
        flat = d.reshape(-1, 1)
        inv_flat = scaler.inverse_transform(flat)
        return inv_flat.reshape(N, H)

    def inv_anchor(d):
        return scaler.inverse_transform(d.reshape(-1, 1))

    y_true = inv_seq(y_te)
    last_known = inv_anchor(X_te_r[:, -1, 0])

    results = []

    # --- 1. Statistical Baselines ---
    print("Running Stats Models...")

    # Naive
    y_naive = np.repeat(last_known, HORIZON, axis=1)
    metrics = get_metrics_detailed(y_true, y_naive, last_known)
    # unpack: r2, rmse, avg, steps, high
    results.append(["Naive", "Stats", metrics[0], metrics[1], metrics[2], metrics[3][0], metrics[3][1], metrics[3][2], metrics[4]])

    # SMA
    sma = np.mean(X_te_r[:, :, 0], axis=1).reshape(-1, 1)
    sma_seq = np.repeat(scaler.inverse_transform(sma), HORIZON, axis=1)
    metrics = get_metrics_detailed(y_true, sma_seq, last_known)
    results.append(["SMA", "Stats", metrics[0], metrics[1], metrics[2], metrics[3][0], metrics[3][1], metrics[3][2], metrics[4]])

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_tr_r_flat, y_tr)
    y_pred = inv_seq(lr.predict(X_te_r_flat))
    metrics = get_metrics_detailed(y_true, y_pred, last_known)
    results.append(["Linear Reg", "Stats", metrics[0], metrics[1], metrics[2], metrics[3][0], metrics[3][1], metrics[3][2], metrics[4]])

    # --- 2. Machine Learning ---
    print("Running ML Models...")

    # SVR
    svr = MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1))
    svr.fit(X_tr_f_flat, y_tr)
    y_pred = inv_seq(svr.predict(X_te_f_flat))
    metrics = get_metrics_detailed(y_true, y_pred, last_known)
    results.append(["SVR (RBF)", "ML", metrics[0], metrics[1], metrics[2], metrics[3][0], metrics[3][1], metrics[3][2], metrics[4]])

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_tr_f_flat, y_tr)
    y_pred = inv_seq(rf.predict(X_te_f_flat))
    metrics = get_metrics_detailed(y_true, y_pred, last_known)
    results.append(["Random Forest", "ML", metrics[0], metrics[1], metrics[2], metrics[3][0], metrics[3][1], metrics[3][2], metrics[4]])

    # XGBoost
    xgb_m = xgb.XGBRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    xgb_m.fit(X_tr_f_flat, y_tr)
    y_pred = inv_seq(xgb_m.predict(X_te_f_flat))
    metrics = get_metrics_detailed(y_true, y_pred, last_known)
    results.append(["XGBoost", "ML", metrics[0], metrics[1], metrics[2], metrics[3][0], metrics[3][1], metrics[3][2], metrics[4]])

    # --- 3. Deep Learning ---
    print("Running DL Models...")

    # LSTM
    lstm = LSTMModel(input_dim=2, output_dim=HORIZON)
    lstm = train_dl_model(lstm, X_tr_r, y_tr)
    y_pred = inv_seq(predict_dl(lstm, X_te_r))
    metrics = get_metrics_detailed(y_true, y_pred, last_known)
    results.append(["LSTM", "DL", metrics[0], metrics[1], metrics[2], metrics[3][0], metrics[3][1], metrics[3][2], metrics[4]])

    # Transformer
    tf = SimpleTransformer(input_dim=2, output_dim=HORIZON)
    tf = train_dl_model(tf, X_tr_r, y_tr)
    y_pred = inv_seq(predict_dl(tf, X_te_r))
    metrics = get_metrics_detailed(y_true, y_pred, last_known)
    results.append(["Transformer", "DL", metrics[0], metrics[1], metrics[2], metrics[3][0], metrics[3][1], metrics[3][2], metrics[4]])

    # --- 4. OUR CHAMPION ---
    # Manually append the detailed results
    results.append([MY_MODEL_NAME, "Ours", MY_R2, MY_RMSE, MY_AVG_ACC, MY_DAY1_ACC, MY_DAY2_ACC, MY_DAY3_ACC, MY_H_ACC])

    # --- Reporting ---
    cols = ["Model", "Type", "R2", "RMSE", "Avg Acc", "Day1", "Day2", "Day3", "High Vol"]
    res_df = pd.DataFrame(results, columns=cols).sort_values(by="R2", ascending=False)

    print("\n" + "="*120)
    print("🏆 GRAND UNIFIED BENCHMARK: Sequence Forecast Breakdown (T+1 to T+3) 🏆")
    print("="*120)
    # 使用 pd.to_string 確保所有欄位都印出來，不省略
    print(res_df.to_string(index=False, float_format="%.4f"))
    print("="*120)