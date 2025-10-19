"""
MODWT-MoE Volatility Prediction System
A complete implementation of Mixture of Experts with MODWT decomposition
for forex volatility prediction using Huber Loss.
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
    """Gating Network: decides which expert to trust"""

    def __init__(self, input_size=13, hidden_size=32, num_experts=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, features):
        return self.network(features)


def extract_gating_features(expert1_input, expert2_input, expert3_input, original_vol=None):
    """Extract statistical features for gating network"""

    features_list = []
    cA4 = expert1_input.squeeze(-1)

    # 1. Current volatility level
    current_level = cA4[:, -5:].mean(dim=1, keepdim=True)
    features_list.append(current_level)

    # 2. Change rate
    change_rate = (cA4[:, -1] - cA4[:, -5]) / 5
    features_list.append(change_rate.unsqueeze(1))

    # 3. Acceleration
    recent_change = (cA4[:, -1] - cA4[:, -3]) / 2
    past_change = (cA4[:, -3] - cA4[:, -5]) / 2
    acceleration = recent_change - past_change
    features_list.append(acceleration.unsqueeze(1))

    # 4. Recent std
    recent_std = cA4[:, -10:].std(dim=1, keepdim=True)
    features_list.append(recent_std)

    # 5-8. Component energies
    cD4_energy = (expert2_input[:, :, 0] ** 2).mean(dim=1, keepdim=True)
    cD3_energy = (expert2_input[:, :, 1] ** 2).mean(dim=1, keepdim=True)
    cD2_energy = (expert3_input[:, :, 0] ** 2).mean(dim=1, keepdim=True)
    cD1_energy = (expert3_input[:, :, 1] ** 2).mean(dim=1, keepdim=True)
    features_list.extend([cD4_energy, cD3_energy, cD2_energy, cD1_energy])

    # 9. Trend strength
    x = torch.arange(cA4.shape[1], device=cA4.device).float()
    x_mean = x.mean()
    y_mean = cA4.mean(dim=1, keepdim=True)
    numerator = ((x - x_mean) * (cA4 - y_mean)).sum(dim=1)
    denominator = ((x - x_mean) ** 2).sum()
    slope_value = numerator / denominator  # 先不取絕對值
    slope = torch.abs(slope_value).unsqueeze(1)  # 特徵用絕對值
    features_list.append(slope)

    # 10. Overall volatility
    overall_volatility = cA4.std(dim=1, keepdim=True)
    features_list.append(overall_volatility)

    # 11. 短期 vs 長期波動比
    short_vol = cA4[:, -5:].std(dim=1, keepdim=True)
    long_vol = cA4[:, -20:].std(dim=1, keepdim=True)
    vol_ratio = short_vol / (long_vol + 1e-8)
    features_list.append(vol_ratio)

    # 12. 趨勢強度 (R²)
    # (slope 已經算過了，用它計算 R²)
    y_pred = slope_value.unsqueeze(-1) * (torch.arange(cA4.shape[1], device=cA4.device).float() - x_mean).unsqueeze(0) + y_mean
    ss_res = ((cA4 - y_pred) ** 2).sum(dim=1, keepdim=True)
    ss_tot = ((cA4 - y_mean) ** 2).sum(dim=1, keepdim=True)
    trend_r2 = 1 - ss_res / (ss_tot + 1e-8)
    features_list.append(trend_r2)

    # 13. 波動方向
    vol_direction = (cA4[:, -1] > cA4[:, -5]).float().unsqueeze(1)
    features_list.append(vol_direction)

    features = torch.cat(features_list, dim=1)
    return features


# ==================== Complete MoE Model ====================
class MODWTMoE(nn.Module):
    """Complete MODWT-MoE Model"""

    def __init__(self):
        super().__init__()
        self.expert1 = TrendExpert(input_size=1, hidden_size=32, num_layers=2, dropout=0.2)
        self.expert2 = CyclicExpert(input_size=2, hidden_size=64, num_layers=2, dropout=0.3)
        self.expert3 = HighFreqExpert(input_size=2, hidden_size=32, num_layers=2, dropout=0.4)
        self.gating = GatingNetwork(input_size=13, hidden_size=32, num_experts=3)

    def forward(self, expert1_input, expert2_input, expert3_input):
        pred1 = self.expert1(expert1_input)
        pred2 = self.expert2(expert2_input)
        pred3 = self.expert3(expert3_input)

        expert_preds = torch.cat([pred1, pred2, pred3], dim=1)

        gating_features = extract_gating_features(expert1_input, expert2_input, expert3_input)
        gating_weights = self.gating(gating_features)

        final_pred = (expert_preds * gating_weights).sum(dim=1, keepdim=True)

        return final_pred, expert_preds, gating_weights


# ==================== Loss Function ====================
class CombinedLoss(nn.Module):
    """Combined Loss = Huber + Direction + Diversity"""

    def __init__(self, huber_delta=1.0, alpha=1.0, beta=0.2, gamma=0.05):
        super().__init__()
        self.huber_delta = huber_delta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.huber = nn.HuberLoss(delta=huber_delta)

    def forward(self, predictions, targets, expert_preds):
        # 1. Huber Loss
        huber_loss = self.huber(predictions, targets)

        # 2. Direction Loss
        if predictions.shape[0] > 1:
            pred_diff = predictions[1:] - predictions[:-1]
            target_diff = targets[1:] - targets[:-1]
            pred_direction = (pred_diff > 0).float()
            target_direction = (target_diff > 0).float()
            direction_loss = nn.functional.binary_cross_entropy(pred_direction, target_direction)
        else:
            direction_loss = torch.tensor(0.0, device=predictions.device)

        # 3. Diversity Loss (correlation-based)
        if expert_preds.shape[0] > 1:
            expert_preds_norm = (expert_preds - expert_preds.mean(dim=0)) / (expert_preds.std(dim=0) + 1e-8)
            corr_12 = (expert_preds_norm[:, 0] * expert_preds_norm[:, 1]).mean()
            corr_13 = (expert_preds_norm[:, 0] * expert_preds_norm[:, 2]).mean()
            corr_23 = (expert_preds_norm[:, 1] * expert_preds_norm[:, 2]).mean()
            diversity_loss = (torch.abs(corr_12) + torch.abs(corr_13) + torch.abs(corr_23)) / 3
        else:
            diversity_loss = torch.tensor(0.0, device=predictions.device)

        total_loss = self.alpha * huber_loss + self.beta * direction_loss + self.gamma * diversity_loss

        loss_dict = {
            'total': total_loss.item(),
            'huber': huber_loss.item(),
            'direction': direction_loss.item() if isinstance(direction_loss, torch.Tensor) else 0.0,
            'diversity': diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else 0.0
        }

        return total_loss, loss_dict


# ==================== Training Functions ====================
def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_components = {'huber': 0, 'direction': 0, 'diversity': 0}

    for batch in loader:
        expert1_input = batch['expert1'].to(device)
        expert2_input = batch['expert2'].to(device)
        expert3_input = batch['expert3'].to(device)
        targets = batch['target'].to(device)

        predictions, expert_preds, gating_weights = model(expert1_input, expert2_input, expert3_input)

        loss, loss_dict = criterion(predictions, targets, expert_preds)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        for key in loss_components.keys():
            loss_components[key] += loss_dict[key] * len(targets)

    n_samples = len(loader.dataset)
    avg_loss = total_loss / n_samples
    for key in loss_components.keys():
        loss_components[key] /= n_samples

    return avg_loss, loss_components


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()

    all_preds = []
    all_targets = []
    all_expert_preds = []
    all_gating_weights = []

    for batch in loader:
        expert1_input = batch['expert1'].to(device)
        expert2_input = batch['expert2'].to(device)
        expert3_input = batch['expert3'].to(device)
        targets = batch['target'].to(device)

        predictions, expert_preds, gating_weights = model(expert1_input, expert2_input, expert3_input)

        all_preds.append(predictions.cpu())
        all_targets.append(targets.cpu())
        all_expert_preds.append(expert_preds.cpu())
        all_gating_weights.append(gating_weights.cpu())

    all_preds = torch.cat(all_preds).numpy().flatten()
    all_targets = torch.cat(all_targets).numpy().flatten()
    all_expert_preds = torch.cat(all_expert_preds).numpy()
    all_gating_weights = torch.cat(all_gating_weights).numpy()

    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    if len(all_preds) > 1:
        pred_direction = (all_preds[1:] - all_preds[:-1]) > 0
        true_direction = (all_targets[1:] - all_targets[:-1]) > 0
        direction_acc = np.mean(pred_direction == true_direction)
    else:
        direction_acc = 0.0

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_acc': direction_acc
    }

    return metrics, all_preds, all_targets, all_expert_preds, all_gating_weights


# ==================== Data Preparation ====================
def prepare_modwt_data(df, vol_window=7, lookback=30, forecast_horizon=1,
                       wavelet='haar', level=4, train_ratio=0.8):
    """Complete data preparation pipeline"""

    print("=" * 80)
    print("📊 Preparing MODWT-MoE Data")
    print("=" * 80)

    # Calculate volatility
    df = df.copy()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['log_return'].rolling(vol_window).std() * np.sqrt(252) * 100
    df = df.dropna()

    volatility = df['volatility'].values
    print(f"\n✅ Volatility: {len(volatility)} samples, Mean: {volatility.mean():.4f}%")

    # MODWT Decomposition
    print(f"\n🔧 Performing MODWT decomposition...")
    decomposer = MODWTDecomposer(wavelet=wavelet, level=level)
    components = decomposer.decompose(volatility)
    energies = decomposer.get_component_energies(components)

    print(f"\n📊 Component Energies:")
    for name, energy in energies.items():
        print(f"   {name}: {energy:.2f}%")

    # Split data
    split_idx = int(len(volatility) * train_ratio)
    train_volatility = volatility[:split_idx]
    test_volatility = volatility[split_idx:]

    train_components = {k: v[:split_idx] for k, v in components.items()}
    test_components = {k: v[split_idx:] for k, v in components.items()}

    print(f"\n✅ Train: {len(train_volatility)}, Test: {len(test_volatility)}")

    # Scale data
    scalers = {}
    train_components_scaled = {}
    test_components_scaled = {}

    for comp_name in components.keys():
        scaler = StandardScaler()
        train_components_scaled[comp_name] = scaler.fit_transform(
            train_components[comp_name].reshape(-1, 1)
        ).flatten()
        test_components_scaled[comp_name] = scaler.transform(
            test_components[comp_name].reshape(-1, 1)
        ).flatten()
        scalers[comp_name] = scaler

    target_scaler = scalers['cA4_trend']
    train_target_scaled = target_scaler.transform(train_volatility.reshape(-1, 1)).flatten()
    test_target_scaled = target_scaler.transform(test_volatility.reshape(-1, 1)).flatten()

    # Create datasets
    train_dataset = MODWTVolatilityDataset(
        train_components_scaled, train_target_scaled, window=lookback, forecast_horizon=forecast_horizon
    )
    test_dataset = MODWTVolatilityDataset(
        test_components_scaled, test_target_scaled, window=lookback, forecast_horizon=forecast_horizon
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"\n✅ DataLoaders created: {len(train_loader)} train batches, {len(test_loader)} test batches")

    return train_loader, test_loader, scalers, components, energies


# ==================== Main Training Function ====================
def train_modwt_moe(train_loader, test_loader, num_epochs=50, device=DEVICE):
    """Main training function"""

    print("\n" + "=" * 80)
    print("🚀 Training MODWT-MoE Model")
    print("=" * 80)

    model = MODWTMoE().to(device)
    criterion = CombinedLoss(huber_delta=1.0, alpha=1.0, beta=0.2, gamma=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0026600047005483534, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = {
        'train_loss': [],
        'test_rmse': [],
        'test_r2': [],
        'test_direction_acc': [],
        'learning_rate': []
    }

    best_test_rmse = float('inf')
    patience_counter = 0
    early_stop_patience = 15

    print(f"\n📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🔧 Training on {device}, Epochs: {num_epochs}")

    for epoch in range(num_epochs):
        train_loss, loss_components = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics, preds, targets, expert_preds, gating_weights = evaluate(model, test_loader, device)

        scheduler.step(metrics['rmse'])

        history['train_loss'].append(train_loss)
        history['test_rmse'].append(metrics['rmse'])
        history['test_r2'].append(metrics['r2'])
        history['test_direction_acc'].append(metrics['direction_acc'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.6f} (Huber: {loss_components['huber']:.6f})")
            print(f"  Test RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.6f}, Dir Acc: {metrics['direction_acc']:.4f}")

        if metrics['rmse'] < best_test_rmse:
            best_test_rmse = metrics['rmse']
            patience_counter = 0
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_rmse': best_test_rmse,
                'metrics': metrics
            }
            torch.save(best_model_state, '../best_models/best_modwt_moe_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break

    print(f"\n✅ Training complete! Best RMSE: {best_test_rmse:.6f}")

    checkpoint = torch.load('../best_models/best_modwt_moe_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, history, checkpoint

# ==================== Visualization Functions ====================
def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # RMSE
    axes[0, 1].plot(history['test_rmse'], label='Test RMSE', color='orange')
    axes[0, 1].set_title('Test RMSE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # R²
    axes[1, 0].plot(history['test_r2'], label='Test R²', color='green')
    axes[1, 0].set_title('Test R²')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Direction Accuracy
    axes[1, 1].plot(history['test_direction_acc'], label='Direction Accuracy', color='purple')
    axes[1, 1].set_title('Direction Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training history saved to {save_path}")
    plt.close()


def plot_predictions(targets, predictions, save_path='predictions.png'):
    """Plot predictions vs targets"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Time series
    axes[0].plot(targets, label='True', alpha=0.7, linewidth=1)
    axes[0].plot(predictions, label='Predicted', alpha=0.7, linewidth=1)
    axes[0].set_title('Volatility Prediction: True vs Predicted', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Volatility (%)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Scatter plot
    axes[1].scatter(targets, predictions, alpha=0.5, s=10)
    axes[1].plot([targets.min(), targets.max()], [targets.min(), targets.max()],
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_title('Predicted vs True (Scatter)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('True Volatility (%)')
    axes[1].set_ylabel('Predicted Volatility (%)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Predictions plot saved to {save_path}")
    plt.close()


def plot_gating_weights(gating_weights, save_path='gating_weights.png'):
    """Plot gating weights distribution"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    expert_names = ['Expert 1 (Trend)', 'Expert 2 (Cyclic)', 'Expert 3 (High-Freq)']
    colors = ['green', 'blue', 'orange']

    for i, (name, color) in enumerate(zip(expert_names, colors)):
        axes[i].hist(gating_weights[:, i], bins=50, alpha=0.7, color=color, edgecolor='black')
        axes[i].axvline(gating_weights[:, i].mean(), color='red', linestyle='--',
                        linewidth=2, label=f'Mean: {gating_weights[:, i].mean():.3f}')
        axes[i].set_title(name, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Weight')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gating weights plot saved to {save_path}")
    plt.close()

def save_results_to_csv(targets, predictions, gating_weights, expert_preds,
                        scalers, save_path='results.csv'):
    """Save all results to CSV"""

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

# 分析 Gating 的動態性
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

def plot_gating_by_regime(gating_weights, targets_original):
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
    axes[1].set_ylim([0.15, 0.45])

    # Annotate changes
    axes[1].annotate(f'+2.7%', xy=(2, expert2_means[2]), xytext=(2.2, expert2_means[2] + 0.01),
                     arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                     fontsize=11, fontweight='bold', color='blue')
    axes[1].annotate(f'-2.0%', xy=(2, expert3_means[2]), xytext=(2.2, expert3_means[2] - 0.01),
                     arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                     fontsize=11, fontweight='bold', color='orange')

    plt.tight_layout()
    plt.savefig('../results/gating_dynamics_by_regime.png', dpi=300, bbox_inches='tight')
    print("📊 Saved: ../results/gating_dynamics_by_regime.png")
    plt.show()

# ==================== Main Execution ====================
if __name__ == "__main__":
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv("../dataset//USD_TWD.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"✅ Loaded {len(df)} days of data")

    # Prepare data
    train_loader, test_loader, scalers, components, energies = prepare_modwt_data(
        df,
        vol_window=7,
        lookback=30,
        forecast_horizon=1,
        wavelet='sym4',
        level=4,
        train_ratio=0.8
    )

    # Train model
    trained_model, training_history, best_checkpoint = train_modwt_moe(
        train_loader,
        test_loader,
        num_epochs=50,
        device=DEVICE
    )

    # Final evaluation
    print("\n" + "=" * 80)
    print("📊 Final Evaluation")
    print("=" * 80)

    final_metrics, final_preds, final_targets, final_expert_preds, final_gating_weights = evaluate(
        trained_model, test_loader, DEVICE
    )

    # Inverse transform to original scale
    target_scaler = scalers['cA4_trend']
    final_preds_original = target_scaler.inverse_transform(final_preds.reshape(-1, 1)).flatten()
    final_targets_original = target_scaler.inverse_transform(final_targets.reshape(-1, 1)).flatten()

    rmse_original = np.sqrt(mean_squared_error(final_targets_original, final_preds_original))
    mae_original = mean_absolute_error(final_targets_original, final_preds_original)
    r2_original = r2_score(final_targets_original, final_preds_original)

    print(f"\n✅ Original Scale Performance:")
    print(f"   RMSE: {rmse_original:.4f}%")
    print(f"   MAE: {mae_original:.4f}%")
    print(f"   R²: {r2_original:.6f}")
    print(f"   Direction Accuracy: {final_metrics['direction_acc']:.4f}")

    print(f"\n📊 Gating Weights:")
    print(f"   Expert 1 (Trend): {final_gating_weights[:, 0].mean():.3f} ± {final_gating_weights[:, 0].std():.3f}")
    print(f"   Expert 2 (Cyclic): {final_gating_weights[:, 1].mean():.3f} ± {final_gating_weights[:, 1].std():.3f}")
    print(f"   Expert 3 (High-Freq): {final_gating_weights[:, 2].mean():.3f} ± {final_gating_weights[:, 2].std():.3f}")

    print(f"\n📊 Comparison with Baseline:")
    print(f"   Baseline RMSE: 2.37%")
    print(f"   MoE RMSE: {rmse_original:.4f}%")
    print(f"   Improvement: {(2.37 - rmse_original)/2.37*100:.2f}%")
    print(f"   Baseline Direction: 63.64%")
    print(f"   MoE Direction: {final_metrics['direction_acc']*100:.2f}%")
    print(f"   Improvement: +{(final_metrics['direction_acc']*100 - 63.64):.2f}%")

    # Visualizations (可選)
    print("\n📊 Generating visualizations...")
    plot_training_history(training_history, '../results/training_history.png')
    plot_predictions(final_targets_original, final_preds_original, '../results/predictions.png')
    plot_gating_weights(final_gating_weights, '../results/gating_weights.png')

    print("\n✅ All done! Check the generated plots.")

    # Save results
    results_df = save_results_to_csv(
        final_targets, final_preds, final_gating_weights,
        final_expert_preds, scalers, '../results/modwt_moe_results.csv'
    )

    analyze_gating_dynamics(final_gating_weights, final_targets_original)

    plot_gating_by_regime(final_gating_weights, final_targets_original)