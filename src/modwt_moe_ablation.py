"""
MODWT-MoE 波動率預測 - 消融實驗版本 (Updated for 75/10/15 split)
訓練邏輯與原本相同，僅修改 Gating 機制進行對比

可執行的版本，包含：
1. Soft Gating（原始版本）
2. Hard Gating - Learned（硬切換-學習型）
3. Hard Gating - Rule-based（硬切換-規則型）
4. Single Expert 版本（只用一個專家）

修正：使用與主模型相同的 75/10/15 切分
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

    def __init__(self, wavelet='sym4', level=4):
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

        print(f"   ✅ Dataset Created: {len(self.targets)} samples")

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
    """Expert 2: Cyclic patterns (cD4 + cD3)"""

    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
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


class HighFreqExpert(nn.Module):
    """Expert 3: High-frequency noise (cD2 + cD1)"""

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
    """Soft Gating Network - 原始版本"""

    def __init__(self, input_size=13, hidden_size=32, num_experts=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, features):
        return self.fc(features)


class HardGatingNetwork(nn.Module):
    """硬切換 Gating Network - 學習型"""

    def __init__(self, input_size=13, hidden_size=32, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_experts)
        )

    def forward(self, features):
        logits = self.fc(features)
        selected_expert = torch.argmax(logits, dim=1)

        batch_size = features.shape[0]
        gating_weights = torch.zeros(batch_size, self.num_experts, device=features.device)
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
        gating_weights[:, [0, 1]] = 0.0
        return pred3, expert_preds, gating_weights


# ==================== Loss Function ====================
class CombinedLoss(nn.Module):
    """Combined loss with Huber + Direction + Diversity"""

    def __init__(self, huber_delta=1.0, alpha=1.0, beta=0.2, gamma=0.05):
        super().__init__()
        self.huber_delta = huber_delta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.huber = nn.HuberLoss(delta=huber_delta)

    def forward(self, predictions, targets, expert_predictions):
        huber_loss = self.huber(predictions, targets)

        pred_direction = (predictions[1:] - predictions[:-1]).sign()
        true_direction = (targets[1:] - targets[:-1]).sign()
        direction_loss = 1.0 - (pred_direction == true_direction).float().mean()

        num_experts = expert_predictions.shape[1]
        diversity_loss = 0.0
        count = 0
        for i in range(num_experts):
            for j in range(i+1, num_experts):
                similarity = torch.cosine_similarity(
                    expert_predictions[:, i:i+1],
                    expert_predictions[:, j:j+1],
                    dim=0
                ).mean()
                diversity_loss += torch.abs(similarity)
                count += 1
        diversity_loss = diversity_loss / count if count > 0 else 0.0

        total_loss = self.alpha * huber_loss + self.beta * direction_loss + self.gamma * diversity_loss

        return total_loss, {
            'huber': huber_loss.item(),
            'direction': direction_loss.item(),
            'diversity': diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss
        }


# ==================== Training ====================
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in train_loader:
        expert1_input = batch['expert1'].to(device)
        expert2_input = batch['expert2'].to(device)
        expert3_input = batch['expert3'].to(device)
        targets = batch['target'].to(device)

        predictions, expert_preds, _ = model(expert1_input, expert2_input, expert3_input)

        loss, _ = criterion(predictions, targets, expert_preds)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * targets.shape[0]
        n_samples += targets.shape[0]

    return total_loss / n_samples


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
                       wavelet='sym4', level=4, train_ratio=0.75, val_ratio=0.10):
    """
    Complete data preparation pipeline
    使用與主模型相同的 75/10/15 切分
    """

    print("=" * 80)
    print("📊 Preparing MODWT-MoE Data (75/10/15 Split)")
    print("=" * 80)

    # Calculate volatility (與主模型相同)
    df = df.copy()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['log_return'].rolling(vol_window).std() * np.sqrt(252) * 100  # 年化
    df = df.dropna().reset_index(drop=True)

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

    # Split data (與主模型相同：75/10/15)
    total_len = len(volatility)
    train_split_idx = int(total_len * train_ratio)
    val_split_idx = int(total_len * (train_ratio + val_ratio))

    train_volatility = volatility[:train_split_idx]
    val_volatility = volatility[train_split_idx:val_split_idx]
    test_volatility = volatility[val_split_idx:]

    train_components = {k: v[:train_split_idx] for k, v in components.items()}
    val_components = {k: v[train_split_idx:val_split_idx] for k, v in components.items()}
    test_components = {k: v[val_split_idx:] for k, v in components.items()}

    print(f"\n✅ Train: {len(train_volatility)}, Val: {len(val_volatility)}, Test: {len(test_volatility)}")

    # Scale data (只在 train 上 fit)
    scalers = {}
    train_components_scaled = {}
    val_components_scaled = {}
    test_components_scaled = {}

    for comp_name in components.keys():
        scaler = StandardScaler()
        train_components_scaled[comp_name] = scaler.fit_transform(
            train_components[comp_name].reshape(-1, 1)
        ).flatten()
        val_components_scaled[comp_name] = scaler.transform(
            val_components[comp_name].reshape(-1, 1)
        ).flatten()
        test_components_scaled[comp_name] = scaler.transform(
            test_components[comp_name].reshape(-1, 1)
        ).flatten()
        scalers[comp_name] = scaler

    # Scale targets
    target_scaler = scalers['cA4_trend']
    train_target_scaled = target_scaler.transform(train_volatility.reshape(-1, 1)).flatten()
    val_target_scaled = target_scaler.transform(val_volatility.reshape(-1, 1)).flatten()
    test_target_scaled = target_scaler.transform(test_volatility.reshape(-1, 1)).flatten()

    # Create datasets
    print(f"\n🔧 Creating datasets...")
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

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"\n✅ DataLoaders created: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches")

    return train_loader, val_loader, test_loader, scalers, components, energies


# ==================== Experiment Runner ====================
def run_experiment(model_name, model, train_loader, val_loader, test_loader,
                   num_epochs, device):
    """Run experiment for one model variant with early stopping"""

    print(f"\n{'='*80}")
    print(f"🧪 Training: {model_name}")
    print(f"{'='*80}")

    model = model.to(device)
    criterion = CombinedLoss(huber_delta=1.0, alpha=1.0, beta=0.2, gamma=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 10  # 與主模型相同

    for epoch in range(num_epochs):
        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation
        val_metrics, _, _, _, _ = evaluate(model, val_loader, device)
        val_loss = val_metrics['rmse']

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch [{epoch+1:3d}/{num_epochs}] "
                  f"Train Loss: {train_loss:.6f}, Val RMSE: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   ⚠️ Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_state)

    # Final evaluation on test set
    test_metrics, test_preds, test_targets, test_expert_preds, test_gating = evaluate(
        model, test_loader, device
    )

    print(f"   ✅ Best Val RMSE: {best_val_loss:.6f}")
    print(f"   ✅ Test RMSE: {test_metrics['rmse']:.6f}")

    return model, test_metrics, test_preds, test_targets, test_gating


# ==================== Main Execution ====================
if __name__ == "__main__":
    # Create results directory
    os.makedirs('../results', exist_ok=True)

    # Load data
    print("📂 Loading data...")
    df = pd.read_csv("../dataset/USD_TWD.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"✅ Loaded {len(df)} days of data\n")

    # Prepare data (共用同一份資料)
    train_loader, val_loader, test_loader, scalers, components, energies = prepare_modwt_data(
        df,
        vol_window=7,
        lookback=30,
        forecast_horizon=1,
        wavelet='sym4',
        level=4,
        train_ratio=0.75,
        val_ratio=0.10
    )

    # Define all model variants
    models_to_test = {
        'Soft Gating (Ours)': MODWTMoESoft(),
        'Hard Gating - Learned': MODWTMoEHardLearned(),
        'Hard Gating - Rule': MODWTMoEHardRule(),
        'Expert 1 Only': MODWTExpert1Only(),
        'Expert 2 Only': MODWTExpert2Only(),
        'Expert 3 Only': MODWTExpert3Only(),
    }

    # Run experiments
    results = {}

    for model_name, model in models_to_test.items():
        trained_model, metrics, preds, targets, gating = run_experiment(
            model_name, model, train_loader, val_loader, test_loader,
            num_epochs=50, device=DEVICE
        )

        # Inverse transform to original scale
        preds_original = scalers['cA4_trend'].inverse_transform(preds.reshape(-1, 1)).flatten()
        targets_original = scalers['cA4_trend'].inverse_transform(targets.reshape(-1, 1)).flatten()

        # Recalculate metrics on original scale
        rmse = np.sqrt(mean_squared_error(targets_original, preds_original))
        mae = mean_absolute_error(targets_original, preds_original)
        r2 = r2_score(targets_original, preds_original)

        # Direction accuracy (already calculated)
        direction_acc = metrics['direction_acc']

        results[model_name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_acc': direction_acc,
            'model': trained_model,
            'predictions': preds_original,
            'targets': targets_original,
            'gating_weights': gating
        }

        print(f"\n   📊 Final Results (Original Scale):")
        print(f"      RMSE: {rmse:.4f}%")
        print(f"      MAE: {mae:.4f}%")
        print(f"      R²: {r2:.6f}")
        print(f"      Direction Accuracy: {direction_acc:.4f}")

    # ==================== Compare Results ====================
    print("\n" + "="*80)
    print("📊 ABLATION STUDY RESULTS - FINAL COMPARISON")
    print("="*80)

    print(f"\n{'Model Variant':<30} {'RMSE (%)':<12} {'MAE (%)':<12} {'R²':<10} {'Direction':<10}")
    print("-"*80)

    for model_name, result in results.items():
        print(f"{model_name:<30} "
              f"{result['rmse']:<12.4f} "
              f"{result['mae']:<12.4f} "
              f"{result['r2']:<10.6f} "
              f"{result['direction_acc']:<10.4f}")

    # Find best models
    best_rmse_model = min(results.items(), key=lambda x: x[1]['rmse'])
    best_r2_model = max(results.items(), key=lambda x: x[1]['r2'])
    best_dir_model = max(results.items(), key=lambda x: x[1]['direction_acc'])

    print("\n" + "="*80)
    print("🏆 Best Models:")
    print(f"   Lowest RMSE: {best_rmse_model[0]} ({best_rmse_model[1]['rmse']:.4f}%)")
    print(f"   Highest R²: {best_r2_model[0]} ({best_r2_model[1]['r2']:.6f})")
    print(f"   Best Direction: {best_dir_model[0]} ({best_dir_model[1]['direction_acc']:.4f})")

    # ==================== Visualization ====================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    model_names = list(results.keys())
    rmse_values = [results[m]['rmse'] for m in model_names]
    r2_values = [results[m]['r2'] for m in model_names]
    dir_values = [results[m]['direction_acc'] for m in model_names]

    # Highlight the best (Soft Gating)
    colors = ['red' if 'Soft Gating' in m else 'skyblue' for m in model_names]

    # RMSE comparison
    axes[0].barh(model_names, rmse_values, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('RMSE (%)', fontsize=12)
    axes[0].set_title('RMSE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)

    # R² comparison
    axes[1].barh(model_names, r2_values, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('R²', fontsize=12)
    axes[1].set_title('R² Comparison (Higher is Better)', fontsize=13, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    # Direction accuracy comparison
    axes[2].barh(model_names, dir_values, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_xlabel('Direction Accuracy', fontsize=12)
    axes[2].set_title('Direction Accuracy (Higher is Better)', fontsize=13, fontweight='bold')
    axes[2].axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random (50%)')
    axes[2].legend()
    axes[2].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/ablation_study_comparison.png', dpi=300, bbox_inches='tight')
    print("\n📊 Saved: ../results/ablation_study_comparison.png")
    plt.close()

    print("\n✅ Ablation study complete!")