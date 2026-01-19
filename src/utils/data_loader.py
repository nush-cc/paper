import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class VolatilityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"raw_input": self.X[idx], "target": self.y[idx]}


def prepare_data(df, lookback=30, horizon=3, mode="raw"):
    df = df.copy()
    vol_window = 7

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = df["log_return"].rolling(vol_window).std() * np.sqrt(252) * 100
    df = df.dropna().reset_index(drop=True)

    feature_cols = ["Volatility", "log_return"]
    features = df[feature_cols].values

    # === [修正 1] 防止 Data Leakage (只用訓練集 fit) ===
    n_samples = len(features)
    split_1 = int(n_samples * 0.70)  # 70% 處
    split_2 = int(n_samples * 0.80)  # 80% 處

    scaler = StandardScaler()
    # 只拿 0%~70% 的資料來計算 Mean 和 Std
    scaler.fit(features[:split_1])
    
    # 將轉換套用到全體數據
    features_scaled = scaler.transform(features)

    # === [修正 2] 確保使用 scaled 的數據進行切分 ===
    # 原本代碼誤用了 unscaled 的 'features'
    train_feat = features_scaled[:split_1]
    
    # 這裡的寫法已經正確處理了 Overlap (Lookback)
    val_feat = features_scaled[split_1 - lookback : split_2]
    test_feat = features_scaled[split_2 - lookback :]

    def create_sequences(data, lookback, horizon):
        X, y = [], []
        if len(data) < lookback + horizon:
            return np.array([]), np.array([])

        for i in range(len(data) - lookback - horizon + 1):
            X.append(data[i : i + lookback])
            # 取出未來 horizon 的 Volatility (第 0 欄)
            y.append(data[i + lookback : i + lookback + horizon, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_feat, lookback, horizon)
    X_val, y_val = create_sequences(val_feat, lookback, horizon)
    X_test, y_test = create_sequences(test_feat, lookback, horizon)

    # Loader
    train_loader = DataLoader(
        VolatilityDataset(X_train, y_train), batch_size=32, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        VolatilityDataset(X_val, y_val), batch_size=32, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        VolatilityDataset(X_test, y_test), batch_size=32, shuffle=False, drop_last=False
    )
    
    print("[Data Split]")
    print(f"  Train: {len(X_train)} samples (0% - 70%)")
    print(f"  Val  : {len(X_val)} samples (70% - 80%)")
    print(f"  Test : {len(X_test)} samples (80% - 100%)")

    # === [修正 3] 回傳 scaler 以供後續反轉換 ===
    return train_loader, val_loader, test_loader, scaler