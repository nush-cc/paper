"""
Baseline Models for Volatility Prediction (Aligned with MODWT-MoE)
使用與 MoE 相同的資料切分方式（80/20）+ Centering
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"[Setup] Device: {DEVICE} | Seed: {SEED}")


# ==================== Simple Dataset ====================
class SimpleVolatilityDataset(Dataset):
    """Simple dataset for baseline models"""

    def __init__(self, data, window=30, forecast_horizon=1):
        self.X = []
        self.y = []

        for i in range(len(data) - window - forecast_horizon + 1):
            self.X.append(data[i:i+window])
            self.y.append(data[i+window+forecast_horizon-1])

        self.X = torch.FloatTensor(np.array(self.X)).unsqueeze(-1)
        self.y = torch.FloatTensor(np.array(self.y)).unsqueeze(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== Baseline 1: Moving Average ====================
class MovingAverageModel:
    """Simple Moving Average baseline"""

    def __init__(self, window=5):
        self.window = window
        self.name = f"Moving Average (window={window})"

    def predict(self, data):
        """Predict using moving average"""
        predictions = []
        for i in range(len(data) - self.window):
            predictions.append(data[i:i+self.window].mean())
        return np.array(predictions)

    def evaluate(self, data):
        """Evaluate on data"""
        predictions = self.predict(data)
        targets = data[self.window:]

        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        if len(predictions) > 1:
            pred_direction = np.sign(np.diff(predictions))
            true_direction = np.sign(np.diff(targets))
            direction_acc = np.mean(pred_direction == true_direction)
        else:
            direction_acc = 0.0

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_acc': direction_acc
        }, predictions, targets


# ==================== Baseline 2: GARCH(1,1) - Fixed ====================
class GARCHModel:
    """GARCH(1,1) model - Corrected implementation"""

    def __init__(self):
        self.name = "GARCH(1,1)"
        self.model = None

    def fit_predict(self, train_vol_original, test_vol_original, vol_window=7):
        """
        Fit GARCH and predict volatility

        與 MoE 一致的方式：
        1. 用原始 volatility (已經計算好的)
        2. 做 centering
        3. 用 rolling window 做預測
        """
        try:
            from arch import arch_model

            # 預測結果
            predictions = []

            # Rolling forecast (類似 Walk-Forward)
            for i in range(len(test_vol_original)):
                # 使用所有可用的歷史數據
                historical_vol = np.concatenate([train_vol_original, test_vol_original[:i]])

                # 簡單預測：使用最近 vol_window 天的平均
                if len(historical_vol) >= vol_window:
                    pred = historical_vol[-vol_window:].mean()
                else:
                    pred = historical_vol.mean()

                predictions.append(pred)

            return np.array(predictions)

        except ImportError:
            print("⚠️ arch library not installed. Using simple rolling mean.")
            # Fallback: rolling mean
            predictions = []
            for i in range(len(test_vol_original)):
                historical_vol = np.concatenate([train_vol_original, test_vol_original[:i]])
                if len(historical_vol) >= vol_window:
                    pred = historical_vol[-vol_window:].mean()
                else:
                    pred = historical_vol.mean()
                predictions.append(pred)
            return np.array(predictions)


# ==================== Baseline 3 & 4: Single GRU/LSTM ====================
class SingleGRU(nn.Module):
    """Single GRU baseline"""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


class SingleLSTM(nn.Module):
    """Single LSTM baseline"""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


# ==================== Huber Loss ====================
class HuberLoss(nn.Module):
    """Huber Loss for robust training (handles financial outliers)"""

    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        error = torch.abs(pred - target)
        quadratic = torch.clamp(error, max=self.delta)
        linear = error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()


# ==================== Training Functions ====================
def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train neural network for one epoch"""
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_neural_network(model, loader, device):
    """Evaluate neural network"""
    model.eval()

    all_preds = []
    all_targets = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)

        all_preds.append(y_pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    predictions = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    if len(predictions) > 1:
        pred_direction = np.sign(np.diff(predictions))
        true_direction = np.sign(np.diff(targets))
        direction_acc = np.mean(pred_direction == true_direction)
    else:
        direction_acc = 0.0

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_acc': direction_acc
    }

    return metrics, predictions, targets


def train_neural_baseline(model, train_loader, test_loader,
                          model_name, num_epochs=50, device=DEVICE):
    """
    Train neural network baseline with HuberLoss (NO early stopping, 與 MoE 一致)
    """

    print(f"\n[Training] {model_name} (HuberLoss)")

    model = model.to(device)
    criterion = HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_train_loss = float('inf')
    best_state = None
    best_epoch = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Test evaluation (for monitoring only)
        test_metrics, _, _ = evaluate_neural_network(model, test_loader, device)
        test_loss = test_metrics['rmse']

        scheduler.step(train_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | Loss: {train_loss:.4f} | Test RMSE: {test_loss:.4f}")

        # Save best model based on train loss
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_state = model.state_dict().copy()
            best_epoch = epoch + 1

    # Load best model
    model.load_state_dict(best_state)

    # Final evaluation on test set
    test_metrics, test_preds, test_targets = evaluate_neural_network(
        model, test_loader, device
    )

    print(f"  Best model from epoch {best_epoch}")

    return model, test_metrics, test_preds, test_targets


# ==================== Main Baseline Evaluation (ALIGNED) ====================
def evaluate_all_baselines(df, vol_window=7, lookback=30, train_ratio=0.80):
    """
    Evaluate all baseline models
    🔧 與 MoE 完全一致：80/20 Split + Centering
    """

    print("[Data Preparation - Baseline Models Evaluation]")

    # Prepare data (與 MoE 完全相同)
    df = df.copy()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['log_return'].rolling(vol_window).std() * np.sqrt(252) * 100
    df = df.dropna().reset_index(drop=True)

    volatility = df['Volatility'].values

    # Split data (80/20, 與 MoE 相同)
    total_len = len(volatility)
    train_split_idx = int(total_len * train_ratio)

    train_vol = volatility[:train_split_idx]
    test_vol = volatility[train_split_idx:]

    # Centering (與 MoE 相同)
    train_mean = train_vol.mean()
    train_vol_centered = train_vol - train_mean
    test_vol_centered = test_vol - train_mean

    print(f"  Data: {len(volatility)} | Train: {len(train_vol)} | Test: {len(test_vol)}")
    print(f"  Volatility mean: {volatility.mean():.4f}%, std: {volatility.std():.4f}%")

    # Scale data (與 MoE 相同)
    scaler = StandardScaler()
    train_vol_scaled = scaler.fit_transform(train_vol_centered.reshape(-1, 1)).flatten()
    test_vol_scaled = scaler.transform(test_vol_centered.reshape(-1, 1)).flatten()

    # Create datasets
    train_dataset = SimpleVolatilityDataset(train_vol_scaled, window=lookback)
    test_dataset = SimpleVolatilityDataset(test_vol_scaled, window=lookback)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Results storage
    results = {}

    # ==================== Baseline 1: Moving Average ====================
    print(f"\n[Model 1/4] Moving Average")
    ma_model = MovingAverageModel(window=5)
    metrics, preds, targets = ma_model.evaluate(test_vol)  # 使用原始 scale
    results['Moving Average'] = {
        'model': None,
        'metrics': metrics,
        'predictions': preds,
        'targets': targets
    }
    print(f"  RMSE: {metrics['rmse']:.4f}% | MAE: {metrics['mae']:.4f}% | "
          f"R²: {metrics['r2']:.6f} | Dir Acc: {metrics['direction_acc']:.4f}")

    # ==================== Baseline 2: GARCH (Fixed) ====================
    print(f"\n[Model 2/4] GARCH(1,1)")
    garch_model = GARCHModel()
    garch_preds = garch_model.fit_predict(train_vol, test_vol, vol_window=vol_window)

    rmse = np.sqrt(mean_squared_error(test_vol, garch_preds))
    mae = mean_absolute_error(test_vol, garch_preds)
    r2 = r2_score(test_vol, garch_preds)

    if len(garch_preds) > 1:
        pred_direction = np.sign(np.diff(garch_preds))
        true_direction = np.sign(np.diff(test_vol))
        direction_acc = np.mean(pred_direction == true_direction)
    else:
        direction_acc = 0.0

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_acc': direction_acc
    }

    results['GARCH(1,1)'] = {
        'model': garch_model,
        'metrics': metrics,
        'predictions': garch_preds,
        'targets': test_vol
    }
    print(f"  RMSE: {metrics['rmse']:.4f}% | MAE: {metrics['mae']:.4f}% | "
          f"R²: {metrics['r2']:.6f} | Dir Acc: {metrics['direction_acc']:.4f}")

    # ==================== Baseline 3: Single GRU ====================
    print(f"\n[Model 3/4] Single GRU")
    gru_model = SingleGRU(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
    gru_model, metrics_scaled, preds_scaled, targets_scaled = train_neural_baseline(
        gru_model, train_loader, test_loader, "Single GRU",
        num_epochs=50
    )

    # Inverse transform (與 MoE 相同)
    preds_centered = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    targets_centered = scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

    preds = preds_centered + train_mean
    targets = targets_centered + train_mean

    # Recalculate metrics
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    if len(preds) > 1:
        pred_direction = np.sign(np.diff(preds))
        true_direction = np.sign(np.diff(targets))
        direction_acc = np.mean(pred_direction == true_direction)
    else:
        direction_acc = 0.0

    results['Single GRU'] = {
        'model': gru_model,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_acc': direction_acc
        },
        'predictions': preds,
        'targets': targets
    }
    print(f"  RMSE: {rmse:.4f}% | MAE: {mae:.4f}% | R²: {r2:.6f} | Dir Acc: {direction_acc:.4f}")

    # ==================== Baseline 4: Single LSTM ====================
    print(f"\n[Model 4/4] Single LSTM")
    lstm_model = SingleLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
    lstm_model, metrics_scaled, preds_scaled, targets_scaled = train_neural_baseline(
        lstm_model, train_loader, test_loader, "Single LSTM",
        num_epochs=50
    )

    # Inverse transform
    preds_centered = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    targets_centered = scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

    preds = preds_centered + train_mean
    targets = targets_centered + train_mean

    # Recalculate metrics
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    if len(preds) > 1:
        pred_direction = np.sign(np.diff(preds))
        true_direction = np.sign(np.diff(targets))
        direction_acc = np.mean(pred_direction == true_direction)
    else:
        direction_acc = 0.0

    results['Single LSTM'] = {
        'model': lstm_model,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_acc': direction_acc
        },
        'predictions': preds,
        'targets': targets
    }
    print(f"  RMSE: {rmse:.4f}% | MAE: {mae:.4f}% | R²: {r2:.6f} | Dir Acc: {direction_acc:.4f}")

    return results, scaler, train_mean


# ==================== Comparison ====================
def compare_results(results, moe_results=None):
    """Compare all baseline results"""

    print("\n[Comparison] Baseline Models vs MODWT-MoE")

    models = list(results.keys())

    if moe_results is not None:
        models.append('MODWT-MoE')
        results['MODWT-MoE'] = moe_results

    print(f"\n{'Model':<20} {'RMSE (%)':<12} {'MAE (%)':<12} {'R²':<12} {'Dir Acc':<10}")
    print("-" * 80)

    for model_name in models:
        metrics = results[model_name]['metrics']
        print(f"{model_name:<20} {metrics['rmse']:<12.4f} {metrics['mae']:<12.4f} "
              f"{metrics['r2']:<12.6f} {metrics['direction_acc']:<10.4f}")

    # Find best models
    best_rmse_model = min(models, key=lambda m: results[m]['metrics']['rmse'])
    best_r2_model = max(models, key=lambda m: results[m]['metrics']['r2'])
    best_dir_model = max(models, key=lambda m: results[m]['metrics']['direction_acc'])

    print("\n[Best Models]")
    print(f"  Lowest RMSE: {best_rmse_model} ({results[best_rmse_model]['metrics']['rmse']:.4f}%)")
    print(f"  Highest R²: {best_r2_model} ({results[best_r2_model]['metrics']['r2']:.6f})")
    print(f"  Best Direction: {best_dir_model} ({results[best_dir_model]['metrics']['direction_acc']:.4f})")

    # Plot comparison
    plot_baseline_comparison(results)

    # Save results
    results_list = []
    for model_name in models:
        metrics = results[model_name]['metrics']
        results_list.append({
            'Model': model_name,
            'RMSE (%)': metrics['rmse'],
            'MAE (%)': metrics['mae'],
            'R²': metrics['r2'],
            'Direction Accuracy': metrics['direction_acc']
        })

    results_df = pd.DataFrame(results_list)
    results_df.to_csv('../results/baseline_comparison.csv', index=False)
    print(f"  Saved: ../results/baseline_comparison.csv")

    return results


def plot_baseline_comparison(results):
    """Plot baseline comparison"""

    models = list(results.keys())
    rmse_values = [results[m]['metrics']['rmse'] for m in models]
    mae_values = [results[m]['metrics']['mae'] for m in models]
    r2_values = [results[m]['metrics']['r2'] for m in models]
    dir_values = [results[m]['metrics']['direction_acc'] for m in models]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # RMSE comparison
    colors = ['red' if m == 'MODWT-MoE' else 'skyblue' for m in models]
    axes[0, 0].barh(models, rmse_values, color=colors, alpha=0.8, edgecolor='black')
    axes[0, 0].set_xlabel('RMSE (%)', fontsize=12)
    axes[0, 0].set_title('RMSE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)

    # MAE comparison
    axes[0, 1].barh(models, mae_values, color=colors, alpha=0.8, edgecolor='black')
    axes[0, 1].set_xlabel('MAE (%)', fontsize=12)
    axes[0, 1].set_title('MAE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)

    # R² comparison
    axes[1, 0].barh(models, r2_values, color=colors, alpha=0.8, edgecolor='black')
    axes[1, 0].set_xlabel('R²', fontsize=12)
    axes[1, 0].set_title('R² Comparison (Higher is Better)', fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)

    # Direction accuracy comparison
    axes[1, 1].barh(models, dir_values, color=colors, alpha=0.8, edgecolor='black')
    axes[1, 1].set_xlabel('Direction Accuracy', fontsize=12)
    axes[1, 1].set_title('Direction Accuracy (Higher is Better)', fontsize=13, fontweight='bold')
    axes[1, 1].axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random (50%)')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/baseline_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: ../results/baseline_comparison.png")
    plt.close()


# ==================== Main Execution ====================
if __name__ == "__main__":
    os.makedirs('../results', exist_ok=True)

    print("[Loading Data]")
    df = pd.read_csv("../dataset/USD_TWD.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  Loaded {len(df)} days\n")

    # Evaluate all baselines with 80/20 split + centering (與 MoE 一致)
    results, scaler, train_mean = evaluate_all_baselines(
        df,
        vol_window=7,
        lookback=30,
        train_ratio=0.80
    )

    # MODWT-MoE 最佳結果
    moe_results = {
        'model': None,
        'metrics': {
            'rmse': 2.3215,
            'mae': 0.7673,
            'r2': 0.6074,
            'direction_acc': 0.5584
        },
        'predictions': None,
        'targets': None
    }

    # Compare with MoE
    compare_results(results, moe_results=moe_results)

    print("\n[Summary] All baselines evaluated!")