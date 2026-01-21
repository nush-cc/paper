import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class VolatilityDataset(Dataset):
    def __init__(self, X, y, returns):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.r = torch.FloatTensor(returns) # 新增：真實報酬率

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "raw_input": self.X[idx],
            "target": self.y[idx],
            "return": self.r[idx] # 新增：回傳對應的未來報酬
        }


def prepare_data(df, lookback=30, horizon=3, mode="raw"):
    df = df.copy()
    vol_window = 7

    # === 1. 計算 Log Return (這是回測算損益的基礎) ===
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return"] = df["log_return"].fillna(0.0)

    # === 2. 計算波動率 (這是模型的預測目標) ===
    df["Volatility"] = df["log_return"].rolling(vol_window).std() * np.sqrt(252) * 100
    df = df.dropna().reset_index(drop=True)

    feature_cols = ["Volatility", "log_return"]
    features = df[feature_cols].values

    # === 防止 Data Leakage (只用訓練集 fit) ===
    n_samples = len(features)
    split_1 = int(n_samples * 0.70)
    split_2 = int(n_samples * 0.80)

    scaler = StandardScaler()
    scaler.fit(features[:split_1])
    features_scaled = scaler.transform(features)

    # 為了回測，我們需要保留 "原始的 Log Return" (不經過 Scaler)
    raw_returns = df["log_return"].values

    train_feat = features_scaled[:split_1]
    val_feat = features_scaled[split_1 - lookback : split_2]
    test_feat = features_scaled[split_2 - lookback :]

    # 對應的原始報酬率切分
    train_ret = raw_returns[:split_1]
    val_ret = raw_returns[split_1 - lookback : split_2]
    test_ret = raw_returns[split_2 - lookback :]

    def create_sequences(data, raw_ret, lookback, horizon):
        X, y, y_ret = [], [], []
        
        if len(data) < lookback + horizon:
            return np.array([]), np.array([]), np.array([])

        for i in range(len(data) - lookback - horizon + 1):
            X.append(data[i : i + lookback])
            # y 取波動率 (第 0 欄)
            y.append(data[i + lookback : i + lookback + horizon, 0])
            # y_ret 取原始報酬率 (未來 horizon 天)
            y_ret.append(raw_ret[i + lookback : i + lookback + horizon])
            
        return np.array(X), np.array(y), np.array(y_ret)

    X_train, y_train, r_train = create_sequences(train_feat, train_ret, lookback, horizon)
    X_val, y_val, r_val = create_sequences(val_feat, val_ret, lookback, horizon)
    X_test, y_test, r_test = create_sequences(test_feat, test_ret, lookback, horizon)

    # Loader
    train_loader = DataLoader(
        VolatilityDataset(X_train, y_train, r_train), batch_size=32, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        VolatilityDataset(X_val, y_val, r_val), batch_size=32, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        VolatilityDataset(X_test, y_test, r_test), batch_size=32, shuffle=False, drop_last=False
    )
    
    print("[Data Split]")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val  : {len(X_val)} samples")
    print(f"  Test : {len(X_test)} samples")

    return train_loader, val_loader, test_loader, scaler