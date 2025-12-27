"""
Baseline Comparison Benchmark
===============================================================
Purpose: Compare your V11 model against standard industry baselines.
Models: Naive, SMA, Linear Regression, XGBoost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 請在此填入您 V11 (Enhanced DLinear) 的成績
#    (從剛才的 Output 複製過來)
# ==========================================
MY_V11_NAME = "V11 (Hybrid CNN)"
MY_V11_R2   = 0.6965  # <--- 填入您的 R2
MY_V11_RMSE = 2.4346  # <--- 填入您的 RMSE
MY_V11_ACC  = 0.7131  # <--- 填入您的 Direction Acc
MY_V11_H_ACC= 0.8707  # <--- 填入您的 High Vol Acc

# ==================== Configuration ====================
BASE_DIR = Path(__file__).resolve().parent.parent
dataset_path = BASE_DIR / "dataset" / "GBP_TWD.csv"
SEED = 42
HORIZON = 3
LOOKBACK = 30
np.random.seed(SEED)

# ==================== Data Prep (Same as V11) ====================
def prepare_data(df, lookback=30, horizon=3):
    df = df.copy()
    vol_window = 7

    # Feature Engineering
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = df["log_return"].rolling(vol_window).std() * np.sqrt(252) * 100
    df = df.dropna().reset_index(drop=True)

    # Use same features as V11 (Raw Mode)
    feature_cols = ["Volatility", "log_return"]
    features = df[feature_cols].values

    # Split
    split_idx = int(len(features) * 0.8)
    train_feat, test_feat = features[:split_idx], features[split_idx:]

    # Scale
    scalers = {}
    train_feat_scaled = np.zeros_like(train_feat)
    test_feat_scaled = np.zeros_like(test_feat)

    for i in range(features.shape[1]):
        s = StandardScaler()
        train_feat_scaled[:, i] = s.fit_transform(train_feat[:, i].reshape(-1, 1)).flatten()
        test_feat_scaled[:, i] = s.transform(test_feat[:, i].reshape(-1, 1)).flatten()
        if i == 0: scalers["target"] = s

    # Create Sequences
    def create_sequences(data, lookback, horizon):
        X, y = [], []
        for i in range(len(data) - lookback - horizon + 1):
            X.append(data[i : i + lookback])
            y.append(data[i + lookback + horizon - 1, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_feat_scaled, lookback, horizon)
    X_test, y_test = create_sequences(test_feat_scaled, lookback, horizon)

    return X_train, y_train, X_test, y_test, scalers["target"]

# ==================== Metrics Function ====================
def get_metrics_values(targets, preds, last_knowns):
    r2 = r2_score(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))

    true_delta = targets - last_knowns
    pred_delta = preds - last_knowns
    dir_correct = (np.sign(true_delta) == np.sign(pred_delta))
    acc = np.mean(dir_correct)

    magnitude = np.abs(true_delta)
    thresh = np.percentile(magnitude, 80)
    high_vol_mask = magnitude > thresh
    acc_high = np.mean(dir_correct[high_vol_mask]) if np.sum(high_vol_mask)>0 else 0
    return r2, rmse, acc, acc_high

# ==================== Main Execution ====================
if __name__ == "__main__":
    print(f"Loading data from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    X_train, y_train, X_test, y_test, scaler = prepare_data(df, LOOKBACK, HORIZON)

    # Flatten inputs for non-sequence models (Linear, XGBoost)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Inverse Transform Helpers
    def inv(data): return scaler.inverse_transform(data.reshape(-1, 1)).flatten()
    y_true = inv(y_test)

    # Last Known Value (Volatility at t=0 relative to prediction)
    # X_test shape: (samples, lookback, features). Feature 0 is Volatility.
    # We want the last volatility in the sequence.
    last_known_scaled = X_test[:, -1, 0]
    last_known_real = inv(last_known_scaled)

    results = []

    # ---------------------------------------------------------
    # Model 1: Naive (Persistence)
    # Prediction = Last observed value
    # ---------------------------------------------------------
    print("Running Naive Baseline...")
    y_pred_naive = last_known_real
    metrics = get_metrics_values(y_true, y_pred_naive, last_known_real)
    results.append(["Naive (Persistence)", *metrics])

    # ---------------------------------------------------------
    # Model 2: Simple Moving Average (SMA)
    # Prediction = Mean of the input lookback window
    # ---------------------------------------------------------
    print("Running SMA Baseline...")
    # Calculate mean of volatility feature (index 0) across the lookback axis (axis 1)
    sma_scaled = np.mean(X_test[:, :, 0], axis=1)
    y_pred_sma = inv(sma_scaled)
    metrics = get_metrics_values(y_true, y_pred_sma, last_known_real)
    results.append(["SMA (Moving Avg)", *metrics])

    # ---------------------------------------------------------
    # Model 3: Linear Regression
    # ---------------------------------------------------------
    print("Running Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_flat, y_train)
    lr_pred_scaled = lr.predict(X_test_flat)
    y_pred_lr = inv(lr_pred_scaled)
    metrics = get_metrics_values(y_true, y_pred_lr, last_known_real)
    results.append(["Linear Regression", *metrics])

    # ---------------------------------------------------------
    # Model 4: XGBoost (The Strong Baseline)
    # ---------------------------------------------------------
    print("Running XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=SEED, n_jobs=-1)
    xgb_model.fit(X_train_flat, y_train)
    xgb_pred_scaled = xgb_model.predict(X_test_flat)
    y_pred_xgb = inv(xgb_pred_scaled)
    metrics = get_metrics_values(y_true, y_pred_xgb, last_known_real)
    results.append(["XGBoost", *metrics])

    # ---------------------------------------------------------
    # Add Your V11 Data
    # ---------------------------------------------------------
    results.append(["MY_MODEL", 0.6965, 2.4346, 0.7131, 0.8707])

    # ==================== Visualization & Table ====================
    cols = ["Model", "R2 Score", "RMSE", "Dir Acc", "High Vol Acc"]
    res_df = pd.DataFrame(results, columns=cols).sort_values(by="R2 Score", ascending=False)

    print("\n" + "="*85)
    print("🏆 FINAL LEADERBOARD 🏆")
    print("="*85)
    print(res_df.to_string(index=False, float_format="%.4f"))
    print("="*85)
    #
    # # Plot
    # plt.figure(figsize=(14, 6))
    #
    # # Subplot 1: R2 Score
    # plt.subplot(1, 2, 1)
    # sns.barplot(data=res_df, x="Model", y="R2 Score", palette="viridis")
    # plt.title("R2 Score Comparison (Higher is Better)")
    # plt.ylim(0, 1.0)
    # plt.xticks(rotation=45)
    #
    # # Subplot 2: High Volatility Accuracy
    # plt.subplot(1, 2, 2)
    # sns.barplot(data=res_df, x="Model", y="High Vol Acc", palette="magma")
    # plt.title("High Volatility Accuracy (Crucial for Risk)")
    # plt.ylim(0, 1.0)
    # plt.xticks(rotation=45)
    #
    # plt.tight_layout()
    # plt.show()
    # print("Comparison plot generated.")