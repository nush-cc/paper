"""
Production-Ready MODWT-MoE Volatility Forecasting Model
========================================================

Heterogeneous Hybrid Architecture combining the strengths of both LSTM and GRU:

1. RawLSTM: Baseline branch using LSTM (optimized for low RMSE)
2. AttentionExperts: TrendExpert, CyclicExpert, HighFreqExpert (GRU-based with attention - optimized for direction accuracy)
3. HybridMODWTMoE: Hybrid architecture with zero-initialization and context-aware gating
   - Base branch (LSTM) predicts stable trend baseline (RMSE anchor)
   - Expert branches (GRU) predict residuals from wavelet components (direction accuracy boost)
   - Output: P = P_lstm + branch_weight * P_moe
4. Two-Stage Curriculum Learning: Stage 1 (baseline only) → Stage 2 (joint + auxiliary loss)
5. Walk-Forward Rolling Decomposition: Prevents data leakage
6. Complete evaluation and visualization pipelines

Run: python main.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

# Import modwtpy
try:
    from modwt import modwt, imodwt, modwtmra
    print("[Setup] modwtpy imported successfully.")
except ImportError:
    print("[Setup] Error: modwtpy not found. Please install it:")
    print("        pip install modwtpy")
    print("        or: pip install git+https://github.com/pistonly/modwtpy.git")
    raise

# ==================== Configuration ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = np.random.randint(1, 10000)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
print(f"[Setup] Device: {DEVICE} | Seed: {SEED}")


# ==================== MODWT Decomposer with MRA ====================
class MODWTDecomposer:
    """Perform MODWT decomposition using modwtpy with MRA"""

    def __init__(self, wavelet='db4', level=4):
        self.wavelet = wavelet
        self.level = level
        self.components_names = None

    def decompose(self, signal, use_mra=True):
        """Decompose signal using MODWT"""

        try:
            w = modwt(signal, self.wavelet, self.level)
        except Exception as e:
            print(f"  [Error] MODWT failed: {e}")
            raise

        if use_mra:
            mra = modwtmra(w, self.wavelet)
            components = {}
            for i in range(self.level):
                components[f'cD{i+1}'] = mra[i]
            components[f'cA{self.level}_trend'] = mra[-1]
        else:
            components = {}
            for i in range(self.level):
                components[f'cD{i+1}'] = w[i]
            components[f'cA{self.level}_trend'] = w[-1]

        self.components_names = list(components.keys())
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
    """Dataset for MODWT-based models with walk-forward rolling decomposition"""

    def __init__(self, expert1_features, expert2_features, expert3_features, targets, raw_input=None):
        self.expert1_data = torch.FloatTensor(expert1_features)
        self.expert2_data = torch.FloatTensor(expert2_features)
        self.expert3_data = torch.FloatTensor(expert3_features)
        self.targets = torch.FloatTensor(targets)

        if raw_input is not None:
            self.raw_data = torch.FloatTensor(raw_input)
        else:
            self.raw_data = None

        assert len(self.expert1_data) == len(self.expert2_data) == len(self.expert3_data) == len(self.targets)
        if self.raw_data is not None:
            assert len(self.raw_data) == len(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        batch = {
            'expert1': self.expert1_data[idx],
            'expert2': self.expert2_data[idx],
            'expert3': self.expert3_data[idx],
            'target': self.targets[idx]
        }
        if self.raw_data is not None:
            batch['raw_input'] = self.raw_data[idx]
        return batch


# ==================== Attention Expert Networks ====================
class AttentionExpert(nn.Module):
    """Attention-based Expert Network with Temporal Attention Mechanism (GRU-based)"""

    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2,
                 attention_size=16, expert_name="Expert"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.expert_name = expert_name

        # GRU backbone (for high-frequency residual predictions and direction accuracy)
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

        # Temporal Attention (Bahdanau style)
        self.attention_query = nn.Linear(hidden_size, attention_size)
        self.attention_key = nn.Linear(hidden_size, attention_size)
        self.attention_score = nn.Linear(attention_size, 1)

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_size]

        Returns:
            prediction: [batch_size, 1]
            attention_weights: [batch_size, seq_len]
        """
        # GRU forward pass (returns: output, h_n where h_n is the final hidden state)
        gru_out, h_n = self.gru(x)  # gru_out: [batch, seq_len, hidden_size], h_n: [num_layers, batch, hidden_size]

        # Temporal Attention
        query = self.attention_query(gru_out)
        key = self.attention_key(gru_out)
        attention_logits = torch.tanh(query + key)
        attention_logits = self.attention_score(attention_logits)

        # Softmax attention weights
        attention_weights = torch.softmax(attention_logits.squeeze(-1), dim=1)

        # Apply attention
        context_vector = torch.sum(
            gru_out * attention_weights.unsqueeze(-1),
            dim=1
        )

        context_vector = self.dropout(context_vector)
        prediction = self.fc(context_vector)

        return prediction, attention_weights


class TrendExpert(AttentionExpert):
    """Expert 1: Trend prediction (cA4 only)"""
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.2):
        super().__init__(input_size, hidden_size, num_layers, dropout,
                        attention_size=16, expert_name="TrendExpert")


class CyclicExpert(AttentionExpert):
    """Expert 2: Cyclic prediction (cD4, cD3)"""
    def __init__(self, input_size=2, hidden_size=32, num_layers=2, dropout=0.3):
        super().__init__(input_size, hidden_size, num_layers, dropout,
                        attention_size=16, expert_name="CyclicExpert")


class HighFreqExpert(AttentionExpert):
    """Expert 3: High-frequency prediction (cD2, cD1)"""
    def __init__(self, input_size=2, hidden_size=32, num_layers=2, dropout=0.4):
        super().__init__(input_size, hidden_size, num_layers, dropout,
                        attention_size=16, expert_name="HighFreqExpert")


# ==================== Gating Network ====================
class GatingNetwork(nn.Module):
    """Gating network for expert weighting in Mixture of Experts"""

    def __init__(self, input_size=5, hidden_size=128, num_experts=3, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
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


# ==================== RawLSTM: Baseline Branch ====================
class RawLSTM(nn.Module):
    """Baseline LSTM branch - learns from raw signal"""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, 1]

        Returns:
            prediction: [batch_size, 1]
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        last_hidden = self.dropout(last_hidden)
        prediction = self.fc(last_hidden)
        return prediction


# ==================== HybridMODWTMoE: Heterogeneous Hybrid ====================
class HybridMODWTMoE(nn.Module):
    """
    Heterogeneous Hybrid Architecture: LSTM baseline + GRU-based MoE specialists

    Combines LSTM's stable trend prediction with GRU's directional agility.

    - Branch A (RawLSTM): LSTM learns base prediction from raw signal (RMSE anchor)
    - Branch B (MoE): GRU experts learn residuals from wavelet components (direction accuracy boost)
    - Output: P = P_lstm + branch_weight * P_moe

    Key features:
    - Zero initialization (branch_weight starts at 0.0)
    - Context-aware gating (11 features total)
    - Heterogeneous RNN design: LSTM base + GRU experts
    - Two-Stage Curriculum Learning: Stage 1 trains LSTM-only, Stage 2 trains jointly with auxiliary loss
    """

    def __init__(self):
        super().__init__()
        # Branch A: LSTM baseline
        self.base_branch = RawLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)

        # Branch B: Experts (wavelet-only inputs)
        self.expert1 = TrendExpert(input_size=1)      # cA4 only
        self.expert2 = CyclicExpert(input_size=2)     # cD4, cD3 only
        self.expert3 = HighFreqExpert(input_size=2)   # cD2, cD1 only

        # Context-aware gating (total input: 11)
        # Last wavelets (5) + Mean wavelets (5) + Raw last (1)
        self.gating = GatingNetwork(input_size=11, hidden_size=128, num_experts=3, dropout=0.1)

        # Zero initialization
        self.branch_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, raw_input, expert1_input, expert2_input, expert3_input):
        """
        Args:
            raw_input: [batch, seq_len, 1]
            expert1_input: [batch, seq_len, 1]
            expert2_input: [batch, seq_len, 2]
            expert3_input: [batch, seq_len, 2]

        Returns:
            output, base_pred, moe_pred, weights, predictions, attention_weights
        """
        # Branch A: LSTM baseline
        base_pred = self.base_branch(raw_input)

        # Branch B: Experts
        pred1, attn1 = self.expert1(expert1_input)
        pred2, attn2 = self.expert2(expert2_input)
        pred3, attn3 = self.expert3(expert3_input)

        attention_weights = {
            'expert1': attn1,
            'expert2': attn2,
            'expert3': attn3
        }

        # Context-aware gating input
        # Part 1: Last timestep wavelets
        e1_last = expert1_input[:, -1, :]
        e2_last = expert2_input[:, -1, :]
        e3_last = expert3_input[:, -1, :]
        last_wavelets = torch.cat([e1_last, e2_last, e3_last], dim=1)

        # Part 2: Mean pooling wavelets
        e1_mean = torch.mean(expert1_input, dim=1)
        e2_mean = torch.mean(expert2_input, dim=1)
        e3_mean = torch.mean(expert3_input, dim=1)
        mean_wavelets = torch.cat([e1_mean, e2_mean, e3_mean], dim=1)

        # Part 3: Raw data last timestep
        raw_last = raw_input[:, -1, :]

        # Concatenate all context
        gate_input = torch.cat([last_wavelets, mean_wavelets, raw_last], dim=1)

        # Gating weights
        weights = self.gating(gate_input)

        # MoE prediction
        predictions = torch.stack([pred1, pred2, pred3], dim=2)
        moe_pred = torch.sum(predictions * weights.unsqueeze(1), dim=2)

        # Final output
        output = base_pred + self.branch_weight * moe_pred

        return output, base_pred, moe_pred, weights, predictions.squeeze(1), attention_weights


# ==================== Data Preparation ====================
def prepare_modwt_data(df, vol_window=7, lookback=30, forecast_horizon=1,
                       wavelet='db4', level=4,
                       train_ratio=0.80, batch_size=32,
                       use_robust_scaler=False, buffer_size=200):
    """Prepare MODWT data using walk-forward rolling decomposition"""

    print("[Data Preparation] MODWT with Walk-Forward Rolling Decomposition")

    # Step 1: Calculate volatility
    df = df.copy()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['log_return'].rolling(vol_window).std() * np.sqrt(252) * 100
    df = df.dropna().reset_index(drop=True)

    volatility = df['Volatility'].values
    print(f"  Volatility: min={volatility.min():.4f}%, max={volatility.max():.4f}%, "
          f"mean={volatility.mean():.4f}%, std={volatility.std():.4f}%")

    # Step 2: Split into train/test
    total_len = len(volatility)
    train_split_idx = int(total_len * train_ratio)

    train_vol = volatility[:train_split_idx]
    test_vol = volatility[train_split_idx:]

    print(f"  Split: train={len(train_vol)}, test={len(test_vol)}")

    # Step 3: Center and scale
    train_mean = train_vol.mean()
    train_vol_centered = train_vol - train_mean
    test_vol_centered = test_vol - train_mean

    if use_robust_scaler:
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    train_vol_scaled = scaler.fit_transform(train_vol_centered.reshape(-1, 1)).flatten()
    test_vol_scaled = scaler.transform(test_vol_centered.reshape(-1, 1)).flatten()

    # Step 4: Walk-forward rolling decomposition
    print(f"  [Walk-Forward Rolling MODWT] buffer_size={buffer_size}, lookback={lookback}")

    decomposer = MODWTDecomposer(wavelet=wavelet, level=level)

    def generate_wf_features(vol_series, decomposer, lookback, buffer_size):
        """Generate features using walk-forward rolling decomposition"""
        num_samples = len(vol_series) - lookback - 1

        expert1_data = np.zeros((num_samples, lookback, 1))
        expert2_data = np.zeros((num_samples, lookback, 2))
        expert3_data = np.zeros((num_samples, lookback, 2))
        raw_data = np.zeros((num_samples, lookback, 1))

        for t in range(lookback, lookback + num_samples):
            buffer_start = max(0, t - buffer_size)
            buffer_end = t + 1
            buffer_vol = vol_series[buffer_start:buffer_end]

            try:
                components = decomposer.decompose(buffer_vol, use_mra=True)
            except Exception as e:
                print(f"    Warning: MODWT failed at t={t}, using zeros")
                components = {f'cD{i+1}': np.zeros(len(buffer_vol)) for i in range(level)}
                components[f'cA{level}_trend'] = np.zeros(len(buffer_vol))

            # Extract lookback sequence
            seq_start = t - lookback + 1
            for seq_idx, time_pos in enumerate(range(seq_start, t + 1)):
                buffer_time_idx = time_pos - buffer_start

                if buffer_time_idx < 0 or buffer_time_idx >= len(buffer_vol):
                    continue

                expert1_data[t - lookback, seq_idx, 0] = components[f'cA{level}_trend'][buffer_time_idx]
                expert2_data[t - lookback, seq_idx, 0] = components['cD4'][buffer_time_idx]
                expert2_data[t - lookback, seq_idx, 1] = components['cD3'][buffer_time_idx]
                expert3_data[t - lookback, seq_idx, 0] = components['cD2'][buffer_time_idx]
                expert3_data[t - lookback, seq_idx, 1] = components['cD1'][buffer_time_idx]
                raw_data[t - lookback, seq_idx, 0] = vol_series[time_pos]

        return expert1_data, expert2_data, expert3_data, raw_data

    # Generate features
    print(f"  Generating training features...")
    train_e1, train_e2, train_e3, train_raw = generate_wf_features(
        train_vol_scaled, decomposer, lookback, buffer_size
    )

    print(f"  Generating test features...")
    test_e1, test_e2, test_e3, test_raw = generate_wf_features(
        test_vol_scaled, decomposer, lookback, buffer_size
    )

    # Create targets
    train_targets = train_vol_scaled[lookback + 1:lookback + 1 + len(train_e1)].reshape(-1, 1)
    test_targets = test_vol_scaled[lookback + 1:lookback + 1 + len(test_e1)].reshape(-1, 1)

    print(f"  Train features: {train_e1.shape[0]} samples")
    print(f"  Test features: {test_e1.shape[0]} samples")

    # Create datasets and loaders
    train_dataset = MODWTVolatilityDataset(train_e1, train_e2, train_e3, train_targets, raw_input=train_raw)
    test_dataset = MODWTVolatilityDataset(test_e1, test_e2, test_e3, test_targets, raw_input=test_raw)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    scalers = {
        'target': scaler,
        'volatility_mean': train_mean
    }

    # Calculate component energies
    sample_vol = train_vol_scaled[:buffer_size + 1]
    components = decomposer.decompose(sample_vol, use_mra=True)
    energies = decomposer.get_component_energies(components)

    return train_loader, test_loader, scalers, train_mean, energies


# ==================== Two-Stage Curriculum Training ====================
def train_hybrid_moe_curriculum(train_loader, test_loader, num_epochs=100,
                                 lr=0.001, device=DEVICE):
    """
    Train HybridMODWTMoE Heterogeneous Hybrid with Two-Stage Curriculum Learning + Auxiliary Loss

    Architecture:
    - Base branch: LSTM (optimized for RMSE)
    - Expert branches: GRU (optimized for direction accuracy)

    Stage 1 (First 50%): Freeze GRU experts, train LSTM baseline only
    - Loss: criterion(base_pred, target)

    Stage 2 (Last 50%): Unfreeze all, joint training with auxiliary loss
    - Loss: criterion(output, target) + 0.5 * criterion(base_pred, target)
    """

    print("[Training] HybridMODWTMoE Heterogeneous Hybrid (LSTM base + GRU experts) with Two-Stage Curriculum Learning")

    model = HybridMODWTMoE().to(device)
    criterion = HuberLoss(delta=1.0)

    history = {
        'train_loss': [],
        'test_loss': [],
        'train_rmse': [],
        'train_mae': [],
        'epochs': [],
        'base_pred_mean': [],
        'moe_pred_mean': [],
        'branch_weight': [],
        'stage': []
    }

    best_train_loss = float('inf')
    best_epoch = 1
    best_model_state = model.state_dict().copy()

    stage1_epochs = num_epochs // 2
    stage2_epochs = num_epochs - stage1_epochs

    # ==================== STAGE 1: LSTM-Only Training ====================
    print(f"\n[Stage 1] Training LSTM Baseline Only (Epochs 1-{stage1_epochs})")
    print("  Freezing GRU expert parameters to allow LSTM stabilization")

    model.expert1.requires_grad_(False)
    model.expert2.requires_grad_(False)
    model.expert3.requires_grad_(False)
    model.gating.requires_grad_(False)
    model.branch_weight.requires_grad_(False)

    optimizer = torch.optim.Adam(model.base_branch.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    for epoch in range(stage1_epochs):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        base_preds = []
        moe_preds = []

        for batch in train_loader:
            raw_input = batch['raw_input'].to(device)
            e1 = batch['expert1'].to(device)
            e2 = batch['expert2'].to(device)
            e3 = batch['expert3'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()

            output, base_pred, moe_pred, weights, expert_preds, _ = model(
                raw_input, e1, e2, e3
            )

            loss = criterion(base_pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.base_branch.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.append(base_pred.detach().cpu().numpy())
            train_targets.append(target.cpu().numpy())
            base_preds.append(base_pred.detach().cpu().numpy())
            moe_preds.append(moe_pred.detach().cpu().numpy())

        # Metrics
        avg_train_loss = np.mean(train_losses)
        train_preds_all = np.concatenate(train_preds, axis=0)
        train_targets_all = np.concatenate(train_targets, axis=0)
        base_preds_all = np.concatenate(base_preds, axis=0)
        moe_preds_all = np.concatenate(moe_preds, axis=0)

        train_rmse = np.sqrt(mean_squared_error(train_targets_all, train_preds_all))
        train_mae = mean_absolute_error(train_targets_all, train_preds_all)

        # Test
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in test_loader:
                raw_input = batch['raw_input'].to(device)
                e1 = batch['expert1'].to(device)
                e2 = batch['expert2'].to(device)
                e3 = batch['expert3'].to(device)
                target = batch['target'].to(device)

                output, base_pred, moe_pred, weights, expert_preds, _ = model(
                    raw_input, e1, e2, e3
                )
                loss = criterion(base_pred, target)
                test_losses.append(loss.item())

        avg_test_loss = np.mean(test_losses)
        scheduler.step(avg_train_loss)

        # History
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['train_rmse'].append(train_rmse)
        history['train_mae'].append(train_mae)
        history['epochs'].append(epoch + 1)
        history['base_pred_mean'].append(base_preds_all.mean())
        history['moe_pred_mean'].append(moe_preds_all.mean())
        history['branch_weight'].append(model.branch_weight.item())
        history['stage'].append(1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{stage1_epochs} | Loss: {avg_train_loss:.4f} | RMSE: {train_rmse:.4f}")

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()

    # ==================== STAGE 2: Heterogeneous Joint Training ====================
    print(f"\n[Stage 2] Joint Training with Auxiliary Loss (Epochs {stage1_epochs+1}-{num_epochs})")
    print("  Unfreezing GRU expert parameters for heterogeneous joint refinement")

    best_train_loss = float('inf')
    model.expert1.requires_grad_(True)
    model.expert2.requires_grad_(True)
    model.expert3.requires_grad_(True)
    model.gating.requires_grad_(True)
    model.branch_weight.requires_grad_(True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    for epoch in range(stage2_epochs):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        base_preds = []
        moe_preds = []

        for batch in train_loader:
            raw_input = batch['raw_input'].to(device)
            e1 = batch['expert1'].to(device)
            e2 = batch['expert2'].to(device)
            e3 = batch['expert3'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()

            output, base_pred, moe_pred, weights, expert_preds, _ = model(
                raw_input, e1, e2, e3
            )

            # Main + auxiliary loss
            main_loss = criterion(output, target)
            aux_loss = criterion(base_pred, target)
            loss = main_loss + 0.5 * aux_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.append(output.detach().cpu().numpy())
            train_targets.append(target.cpu().numpy())
            base_preds.append(base_pred.detach().cpu().numpy())
            moe_preds.append(moe_pred.detach().cpu().numpy())

        # Metrics
        avg_train_loss = np.mean(train_losses)
        train_preds_all = np.concatenate(train_preds, axis=0)
        train_targets_all = np.concatenate(train_targets, axis=0)
        base_preds_all = np.concatenate(base_preds, axis=0)
        moe_preds_all = np.concatenate(moe_preds, axis=0)

        train_rmse = np.sqrt(mean_squared_error(train_targets_all, train_preds_all))
        train_mae = mean_absolute_error(train_targets_all, train_preds_all)

        # Test
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in test_loader:
                raw_input = batch['raw_input'].to(device)
                e1 = batch['expert1'].to(device)
                e2 = batch['expert2'].to(device)
                e3 = batch['expert3'].to(device)
                target = batch['target'].to(device)

                output, base_pred, moe_pred, weights, expert_preds, _ = model(
                    raw_input, e1, e2, e3
                )
                loss = criterion(output, target)
                test_losses.append(loss.item())

        avg_test_loss = np.mean(test_losses)
        scheduler.step(avg_train_loss)

        # History
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['train_rmse'].append(train_rmse)
        history['train_mae'].append(train_mae)
        history['epochs'].append(stage1_epochs + epoch + 1)
        history['base_pred_mean'].append(base_preds_all.mean())
        history['moe_pred_mean'].append(moe_preds_all.mean())
        history['branch_weight'].append(model.branch_weight.item())
        history['stage'].append(2)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {stage1_epochs + epoch+1:3d}/{num_epochs} | Loss: {avg_train_loss:.4f} | RMSE: {train_rmse:.4f}")

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = stage1_epochs + epoch + 1
            best_model_state = model.state_dict().copy()

    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nTraining complete. Best model from epoch {best_epoch} loaded.")

    return model, history, best_epoch


# ==================== Evaluation ====================
def evaluate(model, data_loader, device):
    """Evaluate model"""
    model.eval()

    all_preds = []
    all_targets = []
    all_base_preds = []
    all_moe_preds = []
    all_expert_preds = []
    all_gating_weights = []
    all_attention_weights = {'expert1': [], 'expert2': [], 'expert3': []}

    with torch.no_grad():
        for batch in data_loader:
            raw_input = batch['raw_input'].to(device)
            e1 = batch['expert1'].to(device)
            e2 = batch['expert2'].to(device)
            e3 = batch['expert3'].to(device)
            target = batch['target'].to(device)

            output, base_pred, moe_pred, weights, expert_preds, attention_weights = model(
                raw_input, e1, e2, e3
            )

            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_base_preds.append(base_pred.cpu().numpy())
            all_moe_preds.append(moe_pred.cpu().numpy())
            all_expert_preds.append(expert_preds.cpu().numpy())
            all_gating_weights.append(weights.cpu().numpy())

            all_attention_weights['expert1'].append(attention_weights['expert1'].cpu().numpy())
            all_attention_weights['expert2'].append(attention_weights['expert2'].cpu().numpy())
            all_attention_weights['expert3'].append(attention_weights['expert3'].cpu().numpy())

    # Concatenate
    predictions = np.concatenate(all_preds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()
    base_predictions = np.concatenate(all_base_preds, axis=0).flatten()
    moe_predictions = np.concatenate(all_moe_preds, axis=0).flatten()
    expert_preds_all = np.concatenate(all_expert_preds, axis=0)
    gating_weights = np.concatenate(all_gating_weights, axis=0)

    attention_weights_concat = {
        'expert1': np.concatenate(all_attention_weights['expert1'], axis=0),
        'expert2': np.concatenate(all_attention_weights['expert2'], axis=0),
        'expert3': np.concatenate(all_attention_weights['expert3'], axis=0)
    }

    # Metrics
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    direction_true = np.sign(np.diff(targets))
    direction_pred = np.sign(np.diff(predictions))
    direction_acc = np.mean(direction_true == direction_pred)

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_acc': direction_acc
    }

    return (metrics, predictions, targets, base_predictions, moe_predictions,
            expert_preds_all, gating_weights, attention_weights_concat)


# ==================== Visualization ====================
def plot_training_curves(history, output_path='../results/training_history.png'):
    """Plot training curves"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    epochs = history['epochs']
    stages = history['stage']

    # Find stage transition
    stage_transition = None
    for i in range(1, len(stages)):
        if stages[i] != stages[i-1]:
            stage_transition = epochs[i-1]
            break

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    if stage_transition:
        axes[0, 0].axvline(stage_transition, color='green', linestyle='--', linewidth=2,
                          label=f'Stage Transition (Epoch {stage_transition})', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss (Two-Stage: LSTM-only → Heterogeneous Joint)', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(alpha=0.3)

    # RMSE
    axes[0, 1].plot(epochs, history['train_rmse'], 'g-', linewidth=2)
    if stage_transition:
        axes[0, 1].axvline(stage_transition, color='green', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('RMSE (%)', fontsize=12)
    axes[0, 1].set_title('Training RMSE Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # Branch contributions
    axes[1, 0].plot(epochs, history['base_pred_mean'], 'b-', label='LSTM Base Mean (RMSE anchor)', linewidth=2)
    axes[1, 0].plot(epochs, history['moe_pred_mean'], 'orange', label='GRU Experts Mean (direction boost)', linewidth=2)
    axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    if stage_transition:
        axes[1, 0].axvline(stage_transition, color='green', linestyle='--', linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Mean Prediction', fontsize=12)
    axes[1, 0].set_title('Branch Contribution Over Time', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(alpha=0.3)

    # Branch weight
    axes[1, 1].plot(epochs, history['branch_weight'], 'purple', linewidth=2)
    if stage_transition:
        axes[1, 1].axvline(stage_transition, color='green', linestyle='--', linewidth=2,
                          label='Stage Transition', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Branch Weight', fontsize=12)
    axes[1, 1].set_title('Learned MoE Weight (Starts at 0.0)', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_predictions(test_targets_original, test_preds_original, test_base_preds_original,
                     output_path='../results/predictions.png'):
    """Plot predictions"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    time_idx = np.arange(len(test_targets_original))

    # Predictions
    axes[0].plot(time_idx, test_targets_original, 'k-', alpha=0.6, linewidth=1.5, label='Actual')
    axes[0].plot(time_idx, test_preds_original, 'b-', alpha=0.7, linewidth=1.5, label='Hybrid Prediction')
    axes[0].fill_between(time_idx, test_targets_original, test_preds_original, alpha=0.2)
    axes[0].set_ylabel('Volatility (%)', fontsize=12)
    axes[0].set_title('Hybrid Model: Actual vs Predicted Volatility', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Errors
    axes[1].plot(time_idx, test_base_preds_original - test_targets_original, 'b-', alpha=0.7,
                linewidth=1.5, label='LSTM Baseline Error')
    axes[1].plot(time_idx, test_preds_original - test_targets_original, 'orange', alpha=0.7,
                linewidth=1.5, label='Hybrid Model Error')
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_ylabel('Prediction Error (%)', fontsize=12)
    axes[1].set_title('Error Comparison: LSTM Baseline vs Hybrid', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ==================== Main ====================
if __name__ == "__main__":
    os.makedirs('../results', exist_ok=True)

    print("\n" + "="*80)
    print("[MODWT-MoE Volatility Forecasting - Production Model]")
    print("="*80)

    # Load data
    print("\n[Loading Data]")
    df = pd.read_csv("../dataset/USD_TWD.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  Loaded {len(df)} days")

    # Prepare data
    print("\n[Data Preparation]")
    train_loader, test_loader, scalers, volatility_mean, energies = prepare_modwt_data(
        df,
        vol_window=7,
        lookback=30,
        forecast_horizon=1,
        wavelet='db4',
        level=4,
        train_ratio=0.80,
        batch_size=32,
        use_robust_scaler=False,
        buffer_size=200
    )

    # Train
    print("\n[Training]")
    trained_model, training_history, best_epoch = train_hybrid_moe_curriculum(
        train_loader,
        test_loader,
        num_epochs=80,
        lr=0.001,
        device=DEVICE
    )

    # Evaluate
    print("\n[Evaluation]")
    (test_metrics, test_preds, test_targets, test_base_preds, test_moe_preds,
     test_expert_preds, test_gating_weights, test_attention_weights) = evaluate(
        trained_model, test_loader, DEVICE
    )

    # Inverse transform
    target_scaler = scalers['target']

    test_preds_centered = target_scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
    test_targets_centered = target_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()
    test_base_preds_centered = target_scaler.inverse_transform(test_base_preds.reshape(-1, 1)).flatten()
    test_moe_preds_centered = target_scaler.inverse_transform(test_moe_preds.reshape(-1, 1)).flatten()

    test_preds_original = test_preds_centered + volatility_mean
    test_targets_original = test_targets_centered + volatility_mean
    test_base_preds_original = test_base_preds_centered + volatility_mean
    test_moe_preds_original = test_moe_preds_centered + volatility_mean

    # Metrics
    rmse_original = np.sqrt(mean_squared_error(test_targets_original, test_preds_original))
    mae_original = mean_absolute_error(test_targets_original, test_preds_original)
    r2_original = r2_score(test_targets_original, test_preds_original)
    rmse_base = np.sqrt(mean_squared_error(test_targets_original, test_base_preds_original))
    rmse_moe = np.sqrt(mean_squared_error(test_targets_original, test_moe_preds_original))

    print(f"\n[Test Set Performance - Heterogeneous Hybrid (LSTM base + GRU experts)]")
    print(f"  Combined (Hybrid):  RMSE: {rmse_original:.4f}% | MAE: {mae_original:.4f}% | R²: {r2_original:.4f}")
    print(f"  LSTM Base Anchor:   RMSE: {rmse_base:.4f}%")
    print(f"  GRU Expert Boost:   RMSE: {rmse_moe:.4f}%")
    print(f"  Direction Accuracy: {test_metrics['direction_acc']*100:.2f}%")
    print(f"  RMSE Improvement:   {(rmse_base - rmse_original):.4f}% ({(rmse_base - rmse_original)/rmse_base*100:.1f}%)")

    # Visualizations
    print("\n[Visualizations]")
    plot_training_curves(training_history, '../results/training_history.png')
    plot_predictions(test_targets_original, test_preds_original, test_base_preds_original,
                     '../results/predictions.png')

    # Summary
    print(f"\n[Summary]")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Test RMSE: {rmse_original:.4f}%")
    print(f"  Test MAE: {mae_original:.4f}%")
    print(f"  Test R²: {r2_original:.4f}")
    print(f"  Direction Accuracy: {test_metrics['direction_acc']*100:.2f}%")
    print(f"\n  Results saved to ../results/")
    print("="*80 + "\n")