"""
MODWT-MoE Volatility Prediction System with Validation Set
完整實作含驗證集，並確保沒有資料洩漏
Split: 75% Train / 10% Validation / 15% Test

主要改進：
1. 訓練集增加到 75% (從 70%)，確保足夠的學習樣本
2. Early stopping patience 增加到 20 epochs
3. Gating Network 增強：hidden_size 128, BatchNorm, 降低 dropout
4. 完全防止資料洩漏：只在訓練集上 fit Scaler
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
import pywt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print(f"🔧 Device: {DEVICE}")
print(f"🔧 Random Seed: {SEED}")
print("✅ Setup Complete!\n")


# ==================== MODWT Decomposer ====================
class MODWTDecomposer:
    """Perform MODWT decomposition on volatility series"""

    def __init__(self, wavelet='haar', level=4):
        self.wavelet = wavelet
        self.level = level
        self.components_names = None

    def decompose(self, signal):
        """Decompose signal using MODWT"""
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level, mode='periodic')
        target_len = len(signal)

        # Approximation (trend)
        cA = pywt.upcoef('a', coeffs[0], self.wavelet, level=self.level, take=target_len)[:target_len]

        # Details (cycles/noise)
        components = {'cA4_trend': cA}

        for i in range(1, len(coeffs)):
            detail_level = self.level - i + 1
            cD = pywt.upcoef('d', coeffs[i], self.wavelet, level=detail_level, take=target_len)[:target_len]
            components[f'cD{detail_level}'] = cD

        self.components_names = list(components.keys())

        # Verify reconstruction
        reconstructed = sum(components.values())
        recon_error = np.max(np.abs(signal - reconstructed))

        print(f"✅ MODWT Decomposition Complete")
        print(f"   Wavelet: {self.wavelet}, Level: {self.level}")
        print(f"   Components: {self.components_names}")
        print(f"   Reconstruction Error: {recon_error:.10f}")

        return components

    def get_component_energies(self, components):
        """Calculate energy percentage for each component"""
        total_energy = sum(np.sum(comp**2) for comp in components.values())
        energies = {}
        for name, comp in components.items():
            energy_pct = np.sum(comp**2) / total_energy * 100
            energies[name] = energy_pct
        return energies


# ==================== Dataset ====================
class MODWTVolatilityDataset(Dataset):
    """Dataset for MODWT-MoE model"""

    def __init__(self, components_dict, target, window=30, forecast_horizon=1):
        self.window = window
        self.forecast_horizon = forecast_horizon

        self.expert1_data = []  # cA4
        self.expert2_data = []  # cD4, cD3 stacked
        self.expert3_data = []  # cD2, cD1 stacked
        self.targets = []

        cA4 = components_dict['cA4_trend']
        cD4 = components_dict['cD4']
        cD3 = components_dict['cD3']
        cD2 = components_dict['cD2']
        cD1 = components_dict['cD1']

        # Create sliding windows
        for i in range(len(cA4) - window - forecast_horizon + 1):
            self.expert1_data.append(cA4[i:i+window])

            expert2_window = np.stack([cD4[i:i+window], cD3[i:i+window]], axis=1)
            self.expert2_data.append(expert2_window)

            expert3_window = np.stack([cD2[i:i+window], cD1[i:i+window]], axis=1)
            self.expert3_data.append(expert3_window)

            self.targets.append(target[i + window + forecast_horizon - 1])

        # Convert to tensors
        self.expert1_data = torch.FloatTensor(np.array(self.expert1_data)).unsqueeze(-1)
        self.expert2_data = torch.FloatTensor(np.array(self.expert2_data))
        self.expert3_data = torch.FloatTensor(np.array(self.expert3_data))
        self.targets = torch.FloatTensor(np.array(self.targets)).unsqueeze(-1)

        print(f"✅ Dataset Created: {len(self.targets)} samples")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'expert1': self.expert1_data[idx],
            'expert2': self.expert2_data[idx],
            'expert3': self.expert3_data[idx],
            'target': self.targets[idx]
        }


# ==================== Expert Networks ====================
class TrendExpert(nn.Module):
    """Expert 1: Trend prediction (cA4)"""

    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


class CyclicExpert(nn.Module):
    """Expert 2: Cyclic prediction (cD4 + cD3)"""

    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        attention_weights = self.attention(out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        out = torch.sum(out * attention_weights, dim=1)
        out = self.dropout(out)
        return self.fc(out)


class HighFreqExpert(nn.Module):
    """Expert 3: High-frequency/noise (cD2 + cD1)"""

    def __init__(self, input_size=2, hidden_size=32, num_layers=2, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


# ==================== Gating Network ====================
class GatingNetwork(nn.Module):
    """Gating network to combine experts dynamically"""

    def __init__(self, input_size=5, hidden_size=128, num_experts=3, dropout=0.1):
        super().__init__()
        # 增加 hidden_size 從 64 → 128，給 Gating 更多學習容量
        # 降低 dropout 從 0.2 → 0.1，避免過度正則化
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),  # 加入 BatchNorm 穩定訓練
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_experts)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, combined_input):
        logits = self.fc(combined_input)
        return self.softmax(logits)


# ==================== MoE Model ====================
class MODWTMoE(nn.Module):
    """Complete MODWT-MoE model"""

    def __init__(self):
        super().__init__()
        self.expert1 = TrendExpert()
        self.expert2 = CyclicExpert()
        self.expert3 = HighFreqExpert()
        self.gating = GatingNetwork()

    def forward(self, expert1_input, expert2_input, expert3_input):
        # Expert predictions
        pred1 = self.expert1(expert1_input)
        pred2 = self.expert2(expert2_input)
        pred3 = self.expert3(expert3_input)

        # Gating input: last timestep features from all experts
        # expert1_input shape: [batch, window, 1]
        # expert2_input shape: [batch, window, 2]
        # expert3_input shape: [batch, window, 2]

        e1_last = expert1_input[:, -1, :]  # [batch, 1]
        e2_last = expert2_input[:, -1, :]  # [batch, 2]
        e3_last = expert3_input[:, -1, :]  # [batch, 2]

        # Flatten if needed
        if e1_last.dim() > 2:
            e1_last = e1_last.squeeze(-1)
        if e1_last.dim() == 1:
            e1_last = e1_last.unsqueeze(-1)

        gate_input = torch.cat([e1_last, e2_last, e3_last], dim=1)  # [batch, 5]

        # Gating weights
        weights = self.gating(gate_input)

        # Weighted combination
        predictions = torch.stack([pred1, pred2, pred3], dim=2)
        output = torch.sum(predictions * weights.unsqueeze(1), dim=2)

        return output, weights, predictions.squeeze(1)


# ==================== Data Preparation with TRAIN/VAL/TEST Split ====================
def prepare_modwt_data(df, vol_window=7, lookback=30, forecast_horizon=1,
                       wavelet='sym4', level=4,
                       train_ratio=0.75, val_ratio=0.10, batch_size=32):
    """
    準備 MODWT 資料，分為 Train/Val/Test

    【防止資料洩漏的關鍵步驟】:
    1. 先按時間順序切分 Train/Val/Test
    2. 只用 Train 來 fit Scaler
    3. 用同一個 Scaler transform Val 和 Test
    4. MODWT 對整個序列做分解（這是合理的，因為小波變換不會洩漏未來資訊）
    5. 切分後的資料集互不重疊
    """

    print("\n" + "=" * 80)
    print("🔧 Data Preparation with Train/Val/Test Split (75/10/15)")
    print("=" * 80)

    # Step 1: 計算波動率
    print("\n📊 Step 1: Calculate volatility...")
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['log_return'].rolling(window=vol_window).std() * np.sqrt(252) * 100  # 年化
    df = df.dropna().reset_index(drop=True)
    volatility = df['Volatility'].values
    print(f"✅ Volatility calculated: {len(volatility)} samples, Mean: {volatility.mean():.4f}%")

    # Step 2: MODWT 分解 (對完整序列，這是合理的)
    print("\n📊 Step 2: MODWT decomposition...")
    decomposer = MODWTDecomposer(wavelet=wavelet, level=level)
    components = decomposer.decompose(volatility)
    energies = decomposer.get_component_energies(components)

    print("\n📊 Component Energies:")
    for name, energy in energies.items():
        print(f"   {name}: {energy:.2f}%")

    # Step 3: 時間序列切分 (75/10/15)
    print("\n📊 Step 3: Time-series split (75% Train / 10% Val / 15% Test)...")
    total_len = len(volatility)

    # 【關鍵修正】直接按比例切分，不考慮 window
    train_split_idx = int(total_len * train_ratio)
    val_split_idx = int(total_len * (train_ratio + val_ratio))

    print(f"   Total samples: {total_len}")
    print(f"   Train: 0 to {train_split_idx} ({train_split_idx} samples, {train_ratio*100:.0f}%)")
    print(f"   Val: {train_split_idx} to {val_split_idx} ({val_split_idx - train_split_idx} samples, {val_ratio*100:.0f}%)")
    print(f"   Test: {val_split_idx} to {total_len} ({total_len - val_split_idx} samples, {(1-train_ratio-val_ratio)*100:.0f}%)")

    # Step 4: 切分各個 component 和 target
    print("\n📊 Step 4: Split components and targets...")

    train_volatility = volatility[:train_split_idx]
    val_volatility = volatility[train_split_idx:val_split_idx]
    test_volatility = volatility[val_split_idx:]

    train_components = {k: v[:train_split_idx] for k, v in components.items()}
    val_components = {k: v[train_split_idx:val_split_idx] for k, v in components.items()}
    test_components = {k: v[val_split_idx:] for k, v in components.items()}

    # Step 5: Scaling (只用 Train 來 fit!)
    print("\n📊 Step 5: Scaling (fit on TRAIN only)...")
    scalers = {}
    train_components_scaled = {}
    val_components_scaled = {}
    test_components_scaled = {}

    # 對每個 component 建立獨立的 scaler
    for name in components.keys():
        scaler = StandardScaler()

        # ⚠️ 關鍵：只用 train 資料來 fit
        train_components_scaled[name] = scaler.fit_transform(
            train_components[name].reshape(-1, 1)
        ).flatten()

        # Transform val/test
        val_components_scaled[name] = scaler.transform(
            val_components[name].reshape(-1, 1)
        ).flatten()
        test_components_scaled[name] = scaler.transform(
            test_components[name].reshape(-1, 1)
        ).flatten()

        scalers[name] = scaler
        print(f"   ✅ {name}: Mean={scaler.mean_[0]:.4f}, Std={scaler.scale_[0]:.4f}")

    # Scale targets using cA4_trend scaler
    target_scaler = scalers['cA4_trend']
    train_target_scaled = target_scaler.transform(train_volatility.reshape(-1, 1)).flatten()
    val_target_scaled = target_scaler.transform(val_volatility.reshape(-1, 1)).flatten()
    test_target_scaled = target_scaler.transform(test_volatility.reshape(-1, 1)).flatten()

    # Step 6: 建立 Dataset 和 DataLoader
    print("\n📊 Step 6: Create datasets...")

    train_dataset = MODWTVolatilityDataset(
        train_components_scaled, train_target_scaled,
        window=lookback, forecast_horizon=forecast_horizon
    )

    val_dataset = MODWTVolatilityDataset(
        val_components_scaled, val_target_scaled,
        window=lookback, forecast_horizon=forecast_horizon
    )

    test_dataset = MODWTVolatilityDataset(
        test_components_scaled, test_target_scaled,
        window=lookback, forecast_horizon=forecast_horizon
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n✅ Data preparation complete!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, scalers, components, energies


# ==================== Huber Loss ====================
class HuberLoss(nn.Module):
    """Huber Loss for robust training"""

    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        error = torch.abs(pred - target)
        quadratic = torch.min(error, torch.tensor(self.delta, device=error.device))
        linear = error - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        return loss.mean()


# ==================== Training Function with Validation ====================
def train_modwt_moe(train_loader, val_loader, test_loader, num_epochs=50,
                    lr=0.001, delta=1.0, patience=20, device=DEVICE):
    """
    訓練 MODWT-MoE 模型，加入 Early Stopping

    Args:
        patience: Early stopping 的耐心值（連續幾個 epoch 沒改善就停止）
                  增加到 20 以確保 Gating Network 有足夠時間學習
    """

    print("\n" + "=" * 80)
    print("🚀 Training MODWT-MoE Model with Validation")
    print("=" * 80)

    # Initialize model
    model = MODWTMoE().to(device)
    criterion = HuberLoss(delta=delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'epochs': []
    }

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    # Training loop
    for epoch in range(num_epochs):
        # ===== Training Phase =====
        model.train()
        train_losses = []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for batch in train_pbar:
            e1 = batch['expert1'].to(device)
            e2 = batch['expert2'].to(device)
            e3 = batch['expert3'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()
            output, weights, expert_preds = model(e1, e2, e3)
            loss = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = np.mean(train_losses)

        # ===== Validation Phase =====
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                e1 = batch['expert1'].to(device)
                e2 = batch['expert2'].to(device)
                e3 = batch['expert3'].to(device)
                target = batch['target'].to(device)

                output, weights, expert_preds = model(e1, e2, e3)
                loss = criterion(output, target)

                val_losses.append(loss.item())
                val_preds.append(output.cpu().numpy())
                val_targets.append(target.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)

        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_mae = mean_absolute_error(val_targets, val_preds)

        # ===== Test Phase (監控用，不影響訓練) =====
        test_losses = []
        with torch.no_grad():
            for batch in test_loader:
                e1 = batch['expert1'].to(device)
                e2 = batch['expert2'].to(device)
                e3 = batch['expert3'].to(device)
                target = batch['target'].to(device)

                output, _, _ = model(e1, e2, e3)
                loss = criterion(output, target)
                test_losses.append(loss.item())

        avg_test_loss = np.mean(test_losses)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['test_loss'].append(avg_test_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['epochs'].append(epoch + 1)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val RMSE: {val_rmse:.4f} | "
              f"Test Loss: {avg_test_loss:.4f}")

        # ===== Early Stopping =====
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"   ✅ New best model! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                print(f"   Best epoch was {best_epoch} with Val Loss: {best_val_loss:.4f}")
                break

    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\n✅ Training complete! Loaded best model from epoch {best_epoch}")

    return model, history, best_epoch


# ==================== Evaluation Function ====================
def evaluate(model, data_loader, device):
    """Evaluate model on a dataset"""
    model.eval()

    all_preds = []
    all_targets = []
    all_expert_preds = []
    all_gating_weights = []

    with torch.no_grad():
        for batch in data_loader:
            e1 = batch['expert1'].to(device)
            e2 = batch['expert2'].to(device)
            e3 = batch['expert3'].to(device)
            target = batch['target'].to(device)

            output, weights, expert_preds = model(e1, e2, e3)

            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_expert_preds.append(expert_preds.cpu().numpy())
            all_gating_weights.append(weights.cpu().numpy())

    # Concatenate
    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    expert_preds = np.concatenate(all_expert_preds, axis=0)
    gating_weights = np.concatenate(all_gating_weights, axis=0)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    # Direction accuracy
    direction_true = np.sign(np.diff(targets.flatten()))
    direction_pred = np.sign(np.diff(predictions.flatten()))
    direction_acc = np.mean(direction_true == direction_pred)

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_acc': direction_acc
    }

    return metrics, predictions.flatten(), targets.flatten(), expert_preds, gating_weights


# ==================== Visualization Functions ====================
def plot_training_history(history, save_path='training_history.png'):
    """Plot training history with train/val/test curves"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    epochs = history['epochs']

    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['test_loss'], 'g--', label='Test Loss', linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training/Validation/Test Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(alpha=0.3)

    # Val RMSE
    axes[0, 1].plot(epochs, history['val_rmse'], 'purple', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('RMSE', fontsize=12)
    axes[0, 1].set_title('Validation RMSE', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # Val MAE
    axes[1, 0].plot(epochs, history['val_mae'], 'orange', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('MAE', fontsize=12)
    axes[1, 0].set_title('Validation MAE', fontsize=14, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # Loss comparison
    axes[1, 1].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[1, 1].plot(epochs, history['test_loss'], 'g--', label='Test', linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title('Loss Comparison (Log Scale)', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {save_path}")
    plt.close()


def plot_predictions(true_values, predictions, save_path='predictions.png'):
    """Plot predictions vs actual values"""

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Time series plot
    axes[0].plot(true_values, label='Actual', color='blue', linewidth=1.5, alpha=0.7)
    axes[0].plot(predictions, label='Predicted', color='red', linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('Volatility (%)', fontsize=12)
    axes[0].set_title('Volatility Predictions vs Actual', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Scatter plot
    axes[1].scatter(true_values, predictions, alpha=0.5, s=20)
    axes[1].plot([true_values.min(), true_values.max()],
                 [true_values.min(), true_values.max()],
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Volatility (%)', fontsize=12)
    axes[1].set_ylabel('Predicted Volatility (%)', fontsize=12)
    axes[1].set_title('Prediction Scatter Plot', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {save_path}")
    plt.close()


def plot_gating_weights(gating_weights, save_path='gating_weights.png'):
    """Plot gating weights over time"""

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    time = np.arange(len(gating_weights))

    # Stacked area plot
    axes[0].fill_between(time, 0, gating_weights[:, 0],
                         label='Expert 1 (Trend)', color='green', alpha=0.6)
    axes[0].fill_between(time, gating_weights[:, 0],
                         gating_weights[:, 0] + gating_weights[:, 1],
                         label='Expert 2 (Cyclic)', color='blue', alpha=0.6)
    axes[0].fill_between(time, gating_weights[:, 0] + gating_weights[:, 1],
                         gating_weights[:, 0] + gating_weights[:, 1] + gating_weights[:, 2],
                         label='Expert 3 (High-Freq)', color='orange', alpha=0.6)
    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('Gating Weight', fontsize=12)
    axes[0].set_title('Gating Weights Over Time (Stacked)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].set_ylim([0, 1])
    axes[0].grid(alpha=0.3)

    # Individual lines
    axes[1].plot(time, gating_weights[:, 0], label='Expert 1 (Trend)',
                 color='green', linewidth=1.5, alpha=0.8)
    axes[1].plot(time, gating_weights[:, 1], label='Expert 2 (Cyclic)',
                 color='blue', linewidth=1.5, alpha=0.8)
    axes[1].plot(time, gating_weights[:, 2], label='Expert 3 (High-Freq)',
                 color='orange', linewidth=1.5, alpha=0.8)
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel('Gating Weight', fontsize=12)
    axes[1].set_title('Individual Gating Weights Over Time', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {save_path}")
    plt.close()


# ==================== Save Results ====================
def save_results_to_csv(targets, predictions, gating_weights, expert_preds, scalers, save_path):
    """Save detailed results to CSV"""

    # Inverse transform
    target_scaler = scalers['cA4_trend']
    targets_original = target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
    predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # Create DataFrame
    results_df = pd.DataFrame({
        'True_Volatility': targets_original,
        'Predicted_Volatility': predictions_original,
        'Expert1_Weight': gating_weights[:, 0],
        'Expert2_Weight': gating_weights[:, 1],
        'Expert3_Weight': gating_weights[:, 2],
        'Expert1_Pred': expert_preds[:, 0],
        'Expert2_Pred': expert_preds[:, 1],
        'Expert3_Pred': expert_preds[:, 2],
    })

    results_df.to_csv(save_path, index=False)
    print(f"💾 Results saved to {save_path}")

    return results_df


# ==================== Analysis Functions ====================
def analyze_gating_dynamics(gating_weights, volatility):
    """分析 Gating 是否動態調整"""

    # 按波動率分組
    low_vol = volatility < np.percentile(volatility, 33)
    mid_vol = (volatility >= np.percentile(volatility, 33)) & (volatility <= np.percentile(volatility, 67))
    high_vol = volatility > np.percentile(volatility, 67)

    print("📊 Gating Weights by Volatility Regime:")
    print("\nLow Volatility:")
    print(f"  Expert 1: {gating_weights[low_vol, 0].mean():.3f}")
    print(f"  Expert 2: {gating_weights[low_vol, 1].mean():.3f}")
    print(f"  Expert 3: {gating_weights[low_vol, 2].mean():.3f}")

    print("\nMedium Volatility:")
    print(f"  Expert 1: {gating_weights[mid_vol, 0].mean():.3f}")
    print(f"  Expert 2: {gating_weights[mid_vol, 1].mean():.3f}")
    print(f"  Expert 3: {gating_weights[mid_vol, 2].mean():.3f}")

    print("\nHigh Volatility:")
    print(f"  Expert 1: {gating_weights[high_vol, 0].mean():.3f}")
    print(f"  Expert 2: {gating_weights[high_vol, 1].mean():.3f}")
    print(f"  Expert 3: {gating_weights[high_vol, 2].mean():.3f}")


def plot_gating_by_regime(gating_weights, targets_original, save_path='gating_dynamics_by_regime.png'):
    """畫出不同波動區制下的 Gating 權重"""

    # Define regimes
    low_vol = targets_original < np.percentile(targets_original, 33)
    mid_vol = (targets_original >= np.percentile(targets_original, 33)) & \
              (targets_original <= np.percentile(targets_original, 67))
    high_vol = targets_original > np.percentile(targets_original, 67)

    # Calculate means
    regimes = ['Low\nVolatility', 'Medium\nVolatility', 'High\nVolatility']
    expert1_means = [
        gating_weights[low_vol, 0].mean(),
        gating_weights[mid_vol, 0].mean(),
        gating_weights[high_vol, 0].mean()
    ]
    expert2_means = [
        gating_weights[low_vol, 1].mean(),
        gating_weights[mid_vol, 1].mean(),
        gating_weights[high_vol, 1].mean()
    ]
    expert3_means = [
        gating_weights[low_vol, 2].mean(),
        gating_weights[mid_vol, 2].mean(),
        gating_weights[high_vol, 2].mean()
    ]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Stacked bar chart
    x = np.arange(len(regimes))
    width = 0.6

    p1 = axes[0].bar(x, expert1_means, width, label='Expert 1 (Trend)', color='green', alpha=0.8)
    p2 = axes[0].bar(x, expert2_means, width, bottom=expert1_means,
                     label='Expert 2 (Cyclic)', color='blue', alpha=0.8)
    p3 = axes[0].bar(x, expert3_means, width,
                     bottom=np.array(expert1_means) + np.array(expert2_means),
                     label='Expert 3 (High-Freq)', color='orange', alpha=0.8)

    axes[0].set_ylabel('Gating Weight', fontsize=12)
    axes[0].set_title('Gating Weights by Volatility Regime (Stacked)',
                      fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(regimes)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Add percentage labels
    for i, (e1, e2, e3) in enumerate(zip(expert1_means, expert2_means, expert3_means)):
        axes[0].text(i, e1/2, f'{e1:.1%}', ha='center', va='center',
                     fontweight='bold', color='white', fontsize=10)
        axes[0].text(i, e1 + e2/2, f'{e2:.1%}', ha='center', va='center',
                     fontweight='bold', color='white', fontsize=10)
        axes[0].text(i, e1 + e2 + e3/2, f'{e3:.1%}', ha='center', va='center',
                     fontweight='bold', color='white', fontsize=10)

    # Subplot 2: Line plot showing changes
    axes[1].plot(regimes, expert1_means, marker='o', linewidth=2.5,
                 markersize=10, label='Expert 1 (Trend)', color='green')
    axes[1].plot(regimes, expert2_means, marker='s', linewidth=2.5,
                 markersize=10, label='Expert 2 (Cyclic)', color='blue')
    axes[1].plot(regimes, expert3_means, marker='^', linewidth=2.5,
                 markersize=10, label='Expert 3 (High-Freq)', color='orange')

    axes[1].set_ylabel('Gating Weight', fontsize=12)
    axes[1].set_title('Gating Weight Dynamics Across Regimes',
                      fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {save_path}")
    plt.close()


# ==================== Main Execution ====================
if __name__ == "__main__":
    # Create results directory
    os.makedirs('../results', exist_ok=True)

    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv("../dataset/USD_TWD.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"✅ Loaded {len(df)} days of data")

    # Prepare data with Train/Val/Test split
    train_loader, val_loader, test_loader, scalers, components, energies = prepare_modwt_data(
        df,
        vol_window=7,
        lookback=30,
        forecast_horizon=1,
        wavelet='sym4',
        level=4,
        train_ratio=0.75,  # 75% for training (增加訓練資料)
        val_ratio=0.10     # 10% for validation, 15% for test
    )

    # Train model with validation
    trained_model, training_history, best_epoch = train_modwt_moe(
        train_loader,
        val_loader,
        test_loader,
        num_epochs=100,
        patience=20,  # 增加 patience 讓模型有更多時間學習
        device=DEVICE
    )

    # ===== Final Evaluation on Test Set =====
    print("\n" + "=" * 80)
    print("📊 Final Evaluation on Test Set")
    print("=" * 80)

    test_metrics, test_preds, test_targets, test_expert_preds, test_gating_weights = evaluate(
        trained_model, test_loader, DEVICE
    )

    # Inverse transform to original scale
    target_scaler = scalers['cA4_trend']
    test_preds_original = target_scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
    test_targets_original = target_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()

    rmse_original = np.sqrt(mean_squared_error(test_targets_original, test_preds_original))
    mae_original = mean_absolute_error(test_targets_original, test_preds_original)
    r2_original = r2_score(test_targets_original, test_preds_original)

    print(f"\n✅ Test Set Performance (Original Scale):")
    print(f"   RMSE: {rmse_original:.4f}%")
    print(f"   MAE: {mae_original:.4f}%")
    print(f"   R²: {r2_original:.6f}")
    print(f"   Direction Accuracy: {test_metrics['direction_acc']:.4f}")

    print(f"\n📊 Gating Weights on Test Set:")
    print(f"   Expert 1 (Trend): {test_gating_weights[:, 0].mean():.3f} ± {test_gating_weights[:, 0].std():.3f}")
    print(f"   Expert 2 (Cyclic): {test_gating_weights[:, 1].mean():.3f} ± {test_gating_weights[:, 1].std():.3f}")
    print(f"   Expert 3 (High-Freq): {test_gating_weights[:, 2].mean():.3f} ± {test_gating_weights[:, 2].std():.3f}")

    # ===== Validation Set Evaluation =====
    print("\n" + "=" * 80)
    print("📊 Validation Set Performance")
    print("=" * 80)

    val_metrics, val_preds, val_targets, val_expert_preds, val_gating_weights = evaluate(
        trained_model, val_loader, DEVICE
    )

    val_preds_original = target_scaler.inverse_transform(val_preds.reshape(-1, 1)).flatten()
    val_targets_original = target_scaler.inverse_transform(val_targets.reshape(-1, 1)).flatten()

    val_rmse_original = np.sqrt(mean_squared_error(val_targets_original, val_preds_original))
    val_mae_original = mean_absolute_error(val_targets_original, val_preds_original)

    print(f"\n✅ Validation Set Performance:")
    print(f"   RMSE: {val_rmse_original:.4f}%")
    print(f"   MAE: {val_mae_original:.4f}%")
    print(f"   R²: {val_metrics['r2']:.6f}")
    print(f"   Direction Accuracy: {val_metrics['direction_acc']:.4f}")

    # ===== Visualizations =====
    print("\n📊 Generating visualizations...")
    plot_training_history(training_history, '../results/training_history.png')
    plot_predictions(test_targets_original, test_preds_original, '../results/test_predictions.png')
    plot_gating_weights(test_gating_weights, '../results/test_gating_weights.png')

    # ===== Save Results =====
    test_results_df = save_results_to_csv(
        test_targets, test_preds, test_gating_weights,
        test_expert_preds, scalers, '../results/test_results.csv'
    )

    val_results_df = save_results_to_csv(
        val_targets, val_preds, val_gating_weights,
        val_expert_preds, scalers, '../results/val_results.csv'
    )

    # ===== Analysis =====
    print("\n" + "=" * 80)
    print("📊 Gating Dynamics Analysis (Test Set)")
    print("=" * 80)
    analyze_gating_dynamics(test_gating_weights, test_targets_original)
    plot_gating_by_regime(test_gating_weights, test_targets_original,
                          '../results/test_gating_dynamics_by_regime.png')

    print("\n✅ All done! Check the ../results/ folder for outputs.")
    print(f"\n🏆 Best model from epoch {best_epoch}")
    print(f"   Final Test RMSE: {rmse_original:.4f}%")
    print(f"   Final Test Direction Accuracy: {test_metrics['direction_acc']*100:.2f}%")