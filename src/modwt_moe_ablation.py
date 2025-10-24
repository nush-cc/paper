"""
MODWT-MoE 波動率預測 - 消融實驗版本
訓練邏輯與原本相同，僅修改 Gating 機制進行對比

可執行的版本，包含：
1. Soft Gating（原始版本）
2. Hard Gating - Learned（硬切換-學習型）
3. Hard Gating - Rule-based（硬切換-規則型）
4. Single Expert 版本（只用一個專家）
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

        cA = pywt.upcoef('a', coeffs[0], self.wavelet, level=self.level, take=target_len)[:target_len]
        components = {'cA4_trend': cA}

        for i in range(1, len(coeffs)):
            detail_level = self.level - i + 1
            cD = pywt.upcoef('d', coeffs[i], self.wavelet, level=detail_level, take=target_len)[:target_len]
            components[f'cD{detail_level}'] = cD

        self.components_names = list(components.keys())

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

        self.expert1_data = []
        self.expert2_data = []
        self.expert3_data = []
        self.targets = []

        cA4 = components_dict['cA4_trend']
        cD4 = components_dict['cD4']
        cD3 = components_dict['cD3']
        cD2 = components_dict['cD2']
        cD1 = components_dict['cD1']

        for i in range(len(cA4) - window - forecast_horizon + 1):
            self.expert1_data.append(cA4[i:i+window])

            expert2_window = np.stack([cD4[i:i+window], cD3[i:i+window]], axis=1)
            self.expert2_data.append(expert2_window)

            expert3_window = np.stack([cD2[i:i+window], cD1[i:i+window]], axis=1)
            self.expert3_data.append(expert3_window)

            self.targets.append(target[i + window + forecast_horizon - 1])

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


# ==================== Gating Networks ====================
class SoftGatingNetwork(nn.Module):
    """原始 Soft Gating Network"""

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


class HardGatingNetwork(nn.Module):
    """硬切換 Gating Network - 神經網絡決策"""

    def __init__(self, input_size=13, hidden_size=32, num_experts=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_experts)
        )
        self.num_experts = num_experts

    def forward(self, features):
        logits = self.network(features)
        selected_expert = torch.argmax(logits, dim=1)

        gating_weights = torch.zeros(features.shape[0], self.num_experts, device=features.device)
        gating_weights.scatter_(1, selected_expert.unsqueeze(1), 1.0)

        return gating_weights


class RuleBasedGatingNetwork(nn.Module):
    """硬切換 Gating Network - 規則決策"""

    def __init__(self, num_experts=3):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, features):
        batch_size = features.shape[0]
        selected_expert = torch.zeros(batch_size, dtype=torch.long, device=features.device)

        trend_strength = features[:, 11]  # trend_r2
        cycle_energy = features[:, 4] + features[:, 5]  # cD4 + cD3
        highfreq_energy = features[:, 6] + features[:, 7]  # cD2 + cD1

        for i in range(batch_size):
            if trend_strength[i] > 0.5:
                selected_expert[i] = 0  # Expert 1
            elif cycle_energy[i] > highfreq_energy[i]:
                selected_expert[i] = 1  # Expert 2
            else:
                selected_expert[i] = 2  # Expert 3

        gating_weights = torch.zeros(batch_size, self.num_experts, device=features.device)
        gating_weights.scatter_(1, selected_expert.unsqueeze(1), 1.0)

        return gating_weights


def extract_gating_features(expert1_input, expert2_input, expert3_input):
    """Extract statistical features for gating network"""

    features_list = []
    cA4 = expert1_input.squeeze(-1)

    current_level = cA4[:, -5:].mean(dim=1, keepdim=True)
    features_list.append(current_level)

    change_rate = (cA4[:, -1] - cA4[:, -5]) / 5
    features_list.append(change_rate.unsqueeze(1))

    recent_change = (cA4[:, -1] - cA4[:, -3]) / 2
    past_change = (cA4[:, -3] - cA4[:, -5]) / 2
    acceleration = recent_change - past_change
    features_list.append(acceleration.unsqueeze(1))

    recent_std = cA4[:, -10:].std(dim=1, keepdim=True)
    features_list.append(recent_std)

    cD4_energy = (expert2_input[:, :, 0] ** 2).mean(dim=1, keepdim=True)
    cD3_energy = (expert2_input[:, :, 1] ** 2).mean(dim=1, keepdim=True)
    cD2_energy = (expert3_input[:, :, 0] ** 2).mean(dim=1, keepdim=True)
    cD1_energy = (expert3_input[:, :, 1] ** 2).mean(dim=1, keepdim=True)
    features_list.extend([cD4_energy, cD3_energy, cD2_energy, cD1_energy])

    x = torch.arange(cA4.shape[1], device=cA4.device).float()
    x_mean = x.mean()
    y_mean = cA4.mean(dim=1, keepdim=True)
    numerator = ((x - x_mean) * (cA4 - y_mean)).sum(dim=1)
    denominator = ((x - x_mean) ** 2).sum()
    slope_value = numerator / denominator
    slope = torch.abs(slope_value).unsqueeze(1)
    features_list.append(slope)

    overall_volatility = cA4.std(dim=1, keepdim=True)
    features_list.append(overall_volatility)

    short_vol = cA4[:, -5:].std(dim=1, keepdim=True)
    long_vol = cA4[:, -20:].std(dim=1, keepdim=True)
    vol_ratio = short_vol / (long_vol + 1e-8)
    features_list.append(vol_ratio)

    y_pred = slope_value.unsqueeze(-1) * (torch.arange(cA4.shape[1], device=cA4.device).float() - x_mean).unsqueeze(0) + y_mean
    ss_res = ((cA4 - y_pred) ** 2).sum(dim=1, keepdim=True)
    ss_tot = ((cA4 - y_mean) ** 2).sum(dim=1, keepdim=True)
    trend_r2 = 1 - ss_res / (ss_tot + 1e-8)
    features_list.append(trend_r2)

    vol_direction = (cA4[:, -1] > cA4[:, -5]).float().unsqueeze(1)
    features_list.append(vol_direction)

    features = torch.cat(features_list, dim=1)
    return features


# ==================== MoE Models ====================
class MODWTMoESoft(nn.Module):
    """原始 Soft Gating 版本"""

    def __init__(self):
        super().__init__()
        self.expert1 = TrendExpert(input_size=1, hidden_size=32, num_layers=2, dropout=0.2)
        self.expert2 = CyclicExpert(input_size=2, hidden_size=64, num_layers=2, dropout=0.3)
        self.expert3 = HighFreqExpert(input_size=2, hidden_size=32, num_layers=2, dropout=0.4)
        self.gating = SoftGatingNetwork(input_size=13, hidden_size=32, num_experts=3)

    def forward(self, expert1_input, expert2_input, expert3_input):
        pred1 = self.expert1(expert1_input)
        pred2 = self.expert2(expert2_input)
        pred3 = self.expert3(expert3_input)

        expert_preds = torch.cat([pred1, pred2, pred3], dim=1)

        gating_features = extract_gating_features(expert1_input, expert2_input, expert3_input)
        gating_weights = self.gating(gating_features)

        final_pred = (expert_preds * gating_weights).sum(dim=1, keepdim=True)

        return final_pred, expert_preds, gating_weights


class MODWTMoEHardLearned(nn.Module):
    """硬切換版本 - 神經網絡決策"""

    def __init__(self):
        super().__init__()
        self.expert1 = TrendExpert(input_size=1, hidden_size=32, num_layers=2, dropout=0.2)
        self.expert2 = CyclicExpert(input_size=2, hidden_size=64, num_layers=2, dropout=0.3)
        self.expert3 = HighFreqExpert(input_size=2, hidden_size=32, num_layers=2, dropout=0.4)
        self.gating = HardGatingNetwork(input_size=13, hidden_size=32, num_experts=3)

    def forward(self, expert1_input, expert2_input, expert3_input):
        pred1 = self.expert1(expert1_input)
        pred2 = self.expert2(expert2_input)
        pred3 = self.expert3(expert3_input)

        expert_preds = torch.cat([pred1, pred2, pred3], dim=1)

        gating_features = extract_gating_features(expert1_input, expert2_input, expert3_input)
        gating_weights = self.gating(gating_features)

        final_pred = (expert_preds * gating_weights).sum(dim=1, keepdim=True)

        return final_pred, expert_preds, gating_weights


class MODWTMoEHardRule(nn.Module):
    """硬切換版本 - 規則決策"""

    def __init__(self):
        super().__init__()
        self.expert1 = TrendExpert(input_size=1, hidden_size=32, num_layers=2, dropout=0.2)
        self.expert2 = CyclicExpert(input_size=2, hidden_size=64, num_layers=2, dropout=0.3)
        self.expert3 = HighFreqExpert(input_size=2, hidden_size=32, num_layers=2, dropout=0.4)
        self.gating = RuleBasedGatingNetwork(num_experts=3)

    def forward(self, expert1_input, expert2_input, expert3_input):
        pred1 = self.expert1(expert1_input)
        pred2 = self.expert2(expert2_input)
        pred3 = self.expert3(expert3_input)

        expert_preds = torch.cat([pred1, pred2, pred3], dim=1)

        gating_features = extract_gating_features(expert1_input, expert2_input, expert3_input)
        gating_weights = self.gating(gating_features)

        final_pred = (expert_preds * gating_weights).sum(dim=1, keepdim=True)

        return final_pred, expert_preds, gating_weights


class MODWTExpert1Only(nn.Module):
    """只用 Expert 1"""

    def __init__(self):
        super().__init__()
        self.expert1 = TrendExpert(input_size=1, hidden_size=32, num_layers=2, dropout=0.2)

    def forward(self, expert1_input, expert2_input, expert3_input):
        pred1 = self.expert1(expert1_input)
        expert_preds = torch.cat([pred1, pred1, pred1], dim=1)
        gating_weights = torch.ones(expert1_input.shape[0], 3, device=expert1_input.device)
        gating_weights[:, 0] = 1.0
        gating_weights[:, 1:] = 0.0
        return pred1, expert_preds, gating_weights


class MODWTExpert2Only(nn.Module):
    """只用 Expert 2"""

    def __init__(self):
        super().__init__()
        self.expert2 = CyclicExpert(input_size=2, hidden_size=64, num_layers=2, dropout=0.3)

    def forward(self, expert1_input, expert2_input, expert3_input):
        pred2 = self.expert2(expert2_input)
        expert_preds = torch.cat([pred2, pred2, pred2], dim=1)
        gating_weights = torch.ones(expert2_input.shape[0], 3, device=expert2_input.device)
        gating_weights[:, 1] = 1.0
        gating_weights[:, [0, 2]] = 0.0
        return pred2, expert_preds, gating_weights


class MODWTExpert3Only(nn.Module):
    """只用 Expert 3"""

    def __init__(self):
        super().__init__()
        self.expert3 = HighFreqExpert(input_size=2, hidden_size=32, num_layers=2, dropout=0.4)

    def forward(self, expert1_input, expert2_input, expert3_input):
        pred3 = self.expert3(expert3_input)
        expert_preds = torch.cat([pred3, pred3, pred3], dim=1)
        gating_weights = torch.ones(expert3_input.shape[0], 3, device=expert3_input.device)
        gating_weights[:, 2] = 1.0
        gating_weights[:, :2] = 0.0
        return pred3, expert_preds, gating_weights


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
        huber_loss = self.huber(predictions, targets)

        if predictions.shape[0] > 1:
            pred_diff = predictions[1:] - predictions[:-1]
            target_diff = targets[1:] - targets[:-1]
            pred_direction = (pred_diff > 0).float()
            target_direction = (target_diff > 0).float()
            direction_loss = nn.functional.binary_cross_entropy(pred_direction, target_direction)
        else:
            direction_loss = torch.tensor(0.0, device=predictions.device)

        expert_corr = torch.corrcoef(expert_preds.T)
        expert_corr = expert_corr[~torch.eye(3, dtype=torch.bool, device=expert_preds.device)]
        diversity_loss = expert_corr.abs().mean()

        total_loss = self.alpha * huber_loss + self.beta * direction_loss - self.gamma * diversity_loss

        return total_loss, {
            'huber': huber_loss.item(),
            'direction': direction_loss.item(),
            'diversity': diversity_loss.item()
        }


# ==================== Training ====================
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    loss_components = {'huber': 0.0, 'direction': 0.0, 'diversity': 0.0}
    n_samples = 0

    for batch in tqdm(train_loader, desc="Training"):
        expert1_input = batch['expert1'].to(device)
        expert2_input = batch['expert2'].to(device)
        expert3_input = batch['expert3'].to(device)
        targets = batch['target'].to(device)

        predictions, expert_preds, _ = model(expert1_input, expert2_input, expert3_input)

        loss, components = criterion(predictions, targets, expert_preds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.shape[0]
        for key in components.keys():
            loss_components[key] += components[key] * targets.shape[0]
        n_samples += targets.shape[0]

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

    df = df.copy()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['log_return'].rolling(vol_window).std() * np.sqrt(252) * 100
    df = df.dropna()

    volatility = df['volatility'].values
    print(f"\n✅ Volatility: {len(volatility)} samples, Mean: {volatility.mean():.4f}%")

    print(f"\n🔧 Performing MODWT decomposition...")
    decomposer = MODWTDecomposer(wavelet=wavelet, level=level)
    components = decomposer.decompose(volatility)
    energies = decomposer.get_component_energies(components)

    print(f"\n📊 Component Energies:")
    for name, energy in energies.items():
        print(f"   {name}: {energy:.2f}%")

    split_idx = int(len(volatility) * train_ratio)
    train_volatility = volatility[:split_idx]
    test_volatility = volatility[split_idx:]

    train_components = {k: v[:split_idx] for k, v in components.items()}
    test_components = {k: v[split_idx:] for k, v in components.items()}

    print(f"\n✅ Train: {len(train_volatility)}, Test: {len(test_volatility)}")

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

    # 這裡的 model 由外部傳入
    # criterion 和 optimizer 定義在這裡
    criterion = CombinedLoss(huber_delta=1.0, alpha=1.0, beta=0.2, gamma=0.05)

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

    return history, criterion


def run_experiment(model_name, model, train_loader, test_loader, num_epochs, device):
    """運行一個實驗"""

    print(f"\n{'='*80}")
    print(f"🧪 Experiment: {model_name}")
    print(f"{'='*80}\n")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0026600047005483534, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history, criterion = train_modwt_moe(train_loader, test_loader, num_epochs, device)

    best_test_rmse = float('inf')
    patience_counter = 0
    early_stop_patience = 15

    print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🔧 Training on {device}, Epochs: {num_epochs}\n")

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
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.6f} (Huber: {loss_components['huber']:.6f})")
            print(f"  Test RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.6f}, Dir Acc: {metrics['direction_acc']:.4f}\n")

        if metrics['rmse'] < best_test_rmse:
            best_test_rmse = metrics['rmse']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"⚠️ Early stopping at epoch {epoch+1}\n")
            break

    # 恢復最佳模型
    model.load_state_dict(best_model_state)
    final_metrics, final_preds, final_targets, final_expert_preds, final_gating_weights = evaluate(
        model, test_loader, device
    )

    print(f"✅ Training complete! Best RMSE: {best_test_rmse:.6f}\n")

    return model, history, final_metrics, final_preds, final_targets, final_expert_preds, final_gating_weights


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

    # 消融實驗版本列表
    models_config = [
        ("Soft Gating (Original)", MODWTMoESoft, 50),
        ("Hard Gating - Learned", MODWTMoEHardLearned, 50),
        ("Hard Gating - Rule-based", MODWTMoEHardRule, 50),
        ("Expert 1 Only", MODWTExpert1Only, 50),
        ("Expert 2 Only", MODWTExpert2Only, 50),
        ("Expert 3 Only", MODWTExpert3Only, 50),
    ]

    # 保存結果
    results_summary = []

    # 執行所有實驗
    for model_name, model_class, num_epochs in models_config:
        model = model_class()
        trained_model, history, final_metrics, final_preds, final_targets, final_expert_preds, final_gating_weights = run_experiment(
            model_name, model, train_loader, test_loader, num_epochs, DEVICE
        )

        # Inverse transform
        target_scaler = scalers['cA4_trend']
        final_preds_original = target_scaler.inverse_transform(final_preds.reshape(-1, 1)).flatten()
        final_targets_original = target_scaler.inverse_transform(final_targets.reshape(-1, 1)).flatten()

        rmse_original = np.sqrt(mean_squared_error(final_targets_original, final_preds_original))
        mae_original = mean_absolute_error(final_targets_original, final_preds_original)
        r2_original = r2_score(final_targets_original, final_preds_original)

        print(f"📊 {model_name} - Final Results:")
        print(f"   RMSE: {rmse_original:.4f}%")
        print(f"   MAE: {mae_original:.4f}%")
        print(f"   R²: {r2_original:.6f}")
        print(f"   Direction Accuracy: {final_metrics['direction_acc']:.4f}\n")

        results_summary.append({
            'Model': model_name,
            'RMSE': rmse_original,
            'MAE': mae_original,
            'R2': r2_original,
            'Direction_Acc': final_metrics['direction_acc']
        })

    # 列印對比表
    print("\n" + "=" * 80)
    print("📊 ABLATION STUDY RESULTS")
    print("=" * 80)

    results_df = pd.DataFrame(results_summary)
    print(results_df.to_string(index=False))

    # 保存結果
    results_df.to_csv('../results/ablation_study_results.csv', index=False)
    print(f"\n💾 Results saved to ../results/ablation_study_results.csv")