"""
Baseline Models for Volatility Prediction
Includes: Persistence, ARIMA, GARCH, Single GRU, Single LSTM
For comparison with MODWT-MoE model
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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"🔧 Device: {DEVICE}")
print(f"🔧 Random Seed: {SEED}\n")


# ==================== Simple Dataset ====================
class SimpleVolatilityDataset(Dataset):
    """Simple dataset for baseline models"""

    def __init__(self, data, window=30, forecast_horizon=1):
        self.X = []
        self.y = []

        for i in range(len(data) - window - forecast_horizon + 1):
            self.X.append(data[i:i+window])
            self.y.append(data[i+window+forecast_horizon-1])

        self.X = torch.FloatTensor(np.array(self.X)).unsqueeze(-1)  # (N, window, 1)
        self.y = torch.FloatTensor(np.array(self.y)).unsqueeze(-1)  # (N, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== Baseline 1: Persistence (Naive) ====================
# class PersistenceModel:
#     """Naive baseline: tomorrow = today"""
#
#     def __init__(self):
#         self.name = "Persistence (Naive)"
#
#     def predict(self, data):
#         """Predict: next value = current value"""
#         return data[:-1]  # Shift by 1
#
#     def evaluate(self, data):
#         """Evaluate on data"""
#         predictions = self.predict(data)
#         targets = data[1:]
#
#         rmse = np.sqrt(mean_squared_error(targets, predictions))
#         mae = mean_absolute_error(targets, predictions)
#         r2 = r2_score(targets, predictions)
#
#         # Direction accuracy
#         pred_direction = (predictions[1:] - predictions[:-1]) > 0
#         true_direction = (targets[1:] - targets[:-1]) > 0
#         direction_acc = np.mean(pred_direction == true_direction)
#
#         return {
#             'rmse': rmse,
#             'mae': mae,
#             'r2': r2,
#             'direction_acc': direction_acc
#         }, predictions, targets


# ==================== Baseline 2: Simple Moving Average ====================
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

        # Direction accuracy
        if len(predictions) > 1:
            pred_direction = (predictions[1:] - predictions[:-1]) > 0
            true_direction = (targets[1:] - targets[:-1]) > 0
            direction_acc = np.mean(pred_direction == true_direction)
        else:
            direction_acc = 0.0

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_acc': direction_acc
        }, predictions, targets


# ==================== Baseline 3: GARCH(1,1) ====================
class GARCHModel:
    """GARCH(1,1) model using arch library"""

    def __init__(self):
        self.name = "GARCH(1,1)"
        self.model = None

    def fit_predict(self, train_returns, test_returns):
        """Fit GARCH and predict"""
        try:
            from arch import arch_model

            # Fit GARCH(1,1) on training data
            model = arch_model(train_returns * 100, vol='Garch', p=1, q=1)
            self.model = model.fit(disp='off')

            # Rolling forecast
            predictions = []
            for i in range(len(test_returns)):
                # Forecast 1-step ahead
                forecast = self.model.forecast(horizon=1, reindex=False)
                vol_forecast = np.sqrt(forecast.variance.values[-1, 0]) / 100  # Back to original scale
                predictions.append(vol_forecast)

            return np.array(predictions)

        except ImportError:
            print("⚠️ arch library not installed. Skipping GARCH model.")
            print("   Install with: pip install arch")
            return None
        except Exception as e:
            print(f"⚠️ GARCH model failed: {e}")
            return None


# ==================== Baseline 4: Single GRU ====================
class SingleGRU(nn.Module):
    """Single GRU baseline (your original baseline)"""

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


# ==================== Baseline 5: Single LSTM ====================
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

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    # Direction accuracy
    if len(all_preds) > 1:
        pred_direction = (all_preds[1:] - all_preds[:-1]) > 0
        true_direction = (all_targets[1:] - all_targets[:-1]) > 0
        direction_acc = np.mean(pred_direction == true_direction)
    else:
        direction_acc = 0.0

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_acc': direction_acc
    }, all_preds, all_targets


def train_neural_baseline(model, train_loader, test_loader, model_name,
                          num_epochs=50, device=DEVICE):
    """Train neural network baseline"""

    print(f"\n{'='*80}")
    print(f"🚀 Training {model_name}")
    print(f"{'='*80}")

    model = model.to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_rmse = float('inf')
    patience_counter = 0
    early_stop_patience = 15

    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics, preds, targets = evaluate_neural_network(model, test_loader, device)

        scheduler.step(metrics['rmse'])

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {train_loss:.6f} | "
                  f"RMSE: {metrics['rmse']:.6f} | "
                  f"R²: {metrics['r2']:.6f}")

        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"⚠️ Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_state)
    final_metrics, final_preds, final_targets = evaluate_neural_network(
        model, test_loader, device
    )

    print(f"\n✅ Best RMSE: {final_metrics['rmse']:.6f}")

    return model, final_metrics, final_preds, final_targets


# ==================== Main Baseline Evaluation ====================
def evaluate_all_baselines(df, vol_window=7, lookback=30, train_ratio=0.8):
    """Evaluate all baseline models"""

    print("="*80)
    print("📊 Baseline Models Evaluation")
    print("="*80)

    # Prepare data
    df = df.copy()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['log_return'].rolling(vol_window).std() * np.sqrt(252) * 100
    df = df.dropna()

    volatility = df['volatility'].values
    log_returns = df['log_return'].values[vol_window-1:]  # For GARCH

    print(f"\n✅ Data: {len(volatility)} samples")
    print(f"   Mean: {volatility.mean():.4f}%, Std: {volatility.std():.4f}%")

    # Split data
    split_idx = int(len(volatility) * train_ratio)
    train_vol = volatility[:split_idx]
    test_vol = volatility[split_idx:]
    train_returns = log_returns[:split_idx]
    test_returns = log_returns[split_idx:]

    # Scale data for neural networks
    scaler = StandardScaler()
    train_vol_scaled = scaler.fit_transform(train_vol.reshape(-1, 1)).flatten()
    test_vol_scaled = scaler.transform(test_vol.reshape(-1, 1)).flatten()

    # Create datasets for neural networks
    train_dataset = SimpleVolatilityDataset(train_vol_scaled, window=lookback)
    test_dataset = SimpleVolatilityDataset(test_vol_scaled, window=lookback)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Results storage
    results = {}

    # ==================== Baseline 1: Persistence ====================
    # print(f"\n{'='*80}")
    # print("1️⃣ Evaluating Persistence Model")
    # print(f"{'='*80}")
    #
    # persistence = PersistenceModel()
    # metrics, preds, targets = persistence.evaluate(test_vol)
    # results['Persistence'] = {
    #     'model': None,
    #     'metrics': metrics,
    #     'predictions': preds,
    #     'targets': targets
    # }
    # print(f"✅ RMSE: {metrics['rmse']:.4f}%, R²: {metrics['r2']:.6f}, "
    #       f"Direction: {metrics['direction_acc']:.4f}")

    # ==================== Baseline 2: Moving Average ====================
    print(f"\n{'='*80}")
    print("2️⃣ Evaluating Moving Average Model")
    print(f"{'='*80}")

    ma_model = MovingAverageModel(window=5)
    metrics, preds, targets = ma_model.evaluate(test_vol)
    results['Moving Average'] = {
        'model': None,
        'metrics': metrics,
        'predictions': preds,
        'targets': targets
    }
    print(f"✅ RMSE: {metrics['rmse']:.4f}%, R²: {metrics['r2']:.6f}, "
          f"Direction: {metrics['direction_acc']:.4f}")

    # ==================== Baseline 3: GARCH ====================
    print(f"\n{'='*80}")
    print("3️⃣ Evaluating GARCH(1,1) Model")
    print(f"{'='*80}")

    try:
        garch_model = GARCHModel()
        garch_preds = garch_model.fit_predict(train_returns, test_returns)

        if garch_preds is not None:
            # Convert volatility predictions to match test_vol
            garch_preds = garch_preds * np.sqrt(252) * 100  # Annualize

            # Align lengths
            min_len = min(len(garch_preds), len(test_vol))
            garch_preds = garch_preds[:min_len]
            garch_targets = test_vol[:min_len]

            rmse = np.sqrt(mean_squared_error(garch_targets, garch_preds))
            mae = mean_absolute_error(garch_targets, garch_preds)
            r2 = r2_score(garch_targets, garch_preds)

            if len(garch_preds) > 1:
                pred_direction = (garch_preds[1:] - garch_preds[:-1]) > 0
                true_direction = (garch_targets[1:] - garch_targets[:-1]) > 0
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
                'targets': garch_targets
            }
            print(f"✅ RMSE: {metrics['rmse']:.4f}%, R²: {metrics['r2']:.6f}, "
                  f"Direction: {metrics['direction_acc']:.4f}")
    except Exception as e:
        print(f"⚠️ GARCH evaluation failed: {e}")

    # ==================== Baseline 4: Single GRU ====================
    gru_model = SingleGRU(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
    gru_model, metrics, preds_scaled, targets_scaled = train_neural_baseline(
        gru_model, train_loader, test_loader, "Single GRU", num_epochs=50
    )

    # Inverse transform
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    targets = scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

    # Recalculate metrics on original scale
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    if len(preds) > 1:
        pred_direction = (preds[1:] - preds[:-1]) > 0
        true_direction = (targets[1:] - targets[:-1]) > 0
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

    # ==================== Baseline 5: Single LSTM ====================
    lstm_model = SingleLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
    lstm_model, metrics, preds_scaled, targets_scaled = train_neural_baseline(
        lstm_model, train_loader, test_loader, "Single LSTM", num_epochs=50
    )

    # Inverse transform
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    targets = scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

    # Recalculate metrics
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    if len(preds) > 1:
        pred_direction = (preds[1:] - preds[:-1]) > 0
        true_direction = (targets[1:] - targets[:-1]) > 0
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

    return results, scaler


# ==================== Comparison and Visualization ====================
def compare_results(results, moe_results=None):
    """Compare all baseline results"""

    print("\n" + "="*80)
    print("📊 BASELINE MODELS COMPARISON")
    print("="*80)

    # Create comparison table
    models = list(results.keys())

    if moe_results is not None:
        models.append('MODWT-MoE')
        results['MODWT-MoE'] = moe_results

    print(f"\n{'Model':<20} {'RMSE (%)':<12} {'MAE (%)':<12} {'R²':<10} {'Dir Acc':<10}")
    print("-"*80)

    for model_name in models:
        metrics = results[model_name]['metrics']
        print(f"{model_name:<20} "
              f"{metrics['rmse']:<12.4f} "
              f"{metrics['mae']:<12.4f} "
              f"{metrics['r2']:<10.6f} "
              f"{metrics['direction_acc']:<10.4f}")

    # Find best model
    best_rmse_model = min(models, key=lambda m: results[m]['metrics']['rmse'])
    best_r2_model = max(models, key=lambda m: results[m]['metrics']['r2'])
    best_dir_model = max(models, key=lambda m: results[m]['metrics']['direction_acc'])

    print("\n" + "="*80)
    print("🏆 Best Models:")
    print(f"   Lowest RMSE: {best_rmse_model} "
          f"({results[best_rmse_model]['metrics']['rmse']:.4f}%)")
    print(f"   Highest R²: {best_r2_model} "
          f"({results[best_r2_model]['metrics']['r2']:.6f})")
    print(f"   Best Direction: {best_dir_model} "
          f"({results[best_dir_model]['metrics']['direction_acc']:.4f})")

    # Plot comparison
    plot_baseline_comparison(results)

    return results


def plot_baseline_comparison(results):
    """Plot baseline comparison"""

    models = list(results.keys())
    rmse_values = [results[m]['metrics']['rmse'] for m in models]
    r2_values = [results[m]['metrics']['r2'] for m in models]
    dir_values = [results[m]['metrics']['direction_acc'] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # RMSE comparison
    colors = ['red' if m == 'MODWT-MoE' else 'skyblue' for m in models]
    axes[0].barh(models, rmse_values, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('RMSE (%)', fontsize=12)
    axes[0].set_title('RMSE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)

    # R² comparison
    axes[1].barh(models, r2_values, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('R²', fontsize=12)
    axes[1].set_title('R² Comparison (Higher is Better)', fontsize=13, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    # Direction accuracy comparison
    axes[2].barh(models, dir_values, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_xlabel('Direction Accuracy', fontsize=12)
    axes[2].set_title('Direction Accuracy (Higher is Better)', fontsize=13, fontweight='bold')
    axes[2].axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random (50%)')
    axes[2].legend()
    axes[2].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/baseline_comparison.png', dpi=300, bbox_inches='tight')
    print("\n📊 Saved: ../rssults/baseline_comparison.png")
    plt.show()


# ==================== Main Execution ====================
if __name__ == "__main__":
    # Load data
    print("📂 Loading data...")
    df = pd.read_csv("../dataset/USD_TWD.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"✅ Loaded {len(df)} days of data\n")

    # Evaluate all baselines
    results, scaler = evaluate_all_baselines(
        df,
        vol_window=7,
        lookback=30,
        train_ratio=0.8
    )

    # If you have MoE results, add them here
    moe_results = {
        'model': None,
        'metrics': {
            'rmse': 1.7614,
            'mae': 0.5439,
            'r2': 0.755151,
            'direction_acc': 0.7686
        },
        'predictions': None,
        'targets': None
    }

    print(f"✅ MoE RMSE: {moe_results['metrics']['rmse']:.4f}%")
    print(f"✅ MoE R²: {moe_results['metrics']['r2']:.6f}")
    print(f"✅ MoE Direction: {moe_results['metrics']['direction_acc']:.4f}")

    # Compare with MoE
    compare_results(results, moe_results=moe_results)

    print("\n✅ All baselines evaluated!")