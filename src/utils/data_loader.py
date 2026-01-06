import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class VolatilityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx): return {"raw_input": self.X[idx], "target": self.y[idx]}


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

    def create_sequences(data, lookback, horizon):
        X, y = [], []
        for i in range(len(data) - lookback - horizon + 1):
            X.append(data[i: i + lookback])
            # 取出未來 horizon 的 Volatility (第 0 欄)
            y.append(data[i + lookback: i + lookback + horizon, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_feat, lookback, horizon)
    X_test, y_test = create_sequences(test_feat, lookback, horizon)

    # 5. Loader (建議 drop_last=True)
    train_loader = DataLoader(VolatilityDataset(X_train, y_train), batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(VolatilityDataset(X_test, y_test), batch_size=32, shuffle=False, drop_last=False)

    # Scalers 回傳 None，因為已經不需要了
    return train_loader, test_loader, None, X_train, y_train, X_test, y_test

    return train_loader, test_loader, scalers, X_train, y_train, X_test, y_test
