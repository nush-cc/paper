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
            X.append(data[i: i + lookback])
            # y.append(data[i + lookback + horizon - 1, 0])
            y.append(data[i + lookback: i + lookback + horizon, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_feat_scaled, lookback, horizon)
    X_test, y_test = create_sequences(test_feat_scaled, lookback, horizon)

    train_loader = DataLoader(VolatilityDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(VolatilityDataset(X_test, y_test), batch_size=32, shuffle=False)

    return train_loader, test_loader, scalers, X_train, y_train, X_test, y_test
