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
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
print(f"[Setup] Device: {DEVICE}")


# ==================== MODWT Decomposer with MRA ====================
class MODWTDecomposer:
    """Perform MODWT decomposition using modwtpy with MRA"""

    def __init__(self, wavelet='db4', level=4):
        self.wavelet = wavelet
        self.level = level
        self.components_names = None

    def decompose(self, signal, use_mra=True):
        """
        Decompose signal using MODWT

        Args:
            signal: 1D numpy array
            use_mra: if True, return MRA components (recommended)

        Returns:
            components: dict with keys like 'cA4_trend', 'cD4', 'cD3', etc.
        """

        # Perform MODWT
        try:
            w = modwt(signal, self.wavelet, self.level)
        except Exception as e:
            print(f"  [Error] MODWT failed: {e}")
            print(f"  Suggestion: Try using 'haar', 'db2', 'db4', or 'sym4'")
            raise

        # Use MRA for better interpretability
        if use_mra:
            mra = modwtmra(w, self.wavelet)
            # mra shape: [level+1, N]
            # mra[0] = D1, mra[1] = D2, ..., mra[-1] = S_J

            components = {}
            # Details (從細到粗)
            for i in range(self.level):
                components[f'cD{i+1}'] = mra[i]

            # Approximation (trend)
            components[f'cA{self.level}_trend'] = mra[-1]

        else:
            # Use raw MODWT coefficients
            components = {}
            # w shape: [level+1, N]
            # w[0] = w1, w[1] = w2, ..., w[-1] = v_J

            for i in range(self.level):
                components[f'cD{i+1}'] = w[i]
            components[f'cA{self.level}_trend'] = w[-1]

        self.components_names = list(components.keys())

        # Verify reconstruction (only for MRA)
        if use_mra:
            reconstructed = sum(components.values())
            recon_error = np.max(np.abs(signal - reconstructed))
        else:
            try:
                reconstructed = imodwt(w, self.wavelet)
                recon_error = np.max(np.abs(signal - reconstructed[:len(signal)]))
            except:
                recon_error = np.nan

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
    """
    Dataset for MODWT-based models with pre-calculated, leakage-free features.

    This dataset accepts pre-computed MODWT features that were generated using
    strict walk-forward rolling decomposition. This ensures that at time t,
    the model only sees information from t and its past, preventing look-ahead bias.

    Supports both traditional MODWTMoE and Hybrid (RawLSTM + MODWTMoE) architectures.
    """

    def __init__(self, expert1_features, expert2_features, expert3_features, targets, raw_input=None):
        """
        Args:
            expert1_features: Tensor of shape [num_samples, lookback, 1] - Trend (cA4)
            expert2_features: Tensor of shape [num_samples, lookback, 2] - Cyclic (cD4, cD3)
            expert3_features: Tensor of shape [num_samples, lookback, 2] - High-freq (cD2, cD1)
            targets: Tensor of shape [num_samples, 1] - Target volatility
            raw_input: Tensor of shape [num_samples, lookback, 1] - Raw scaled volatility (optional)
        """
        self.expert1_data = torch.FloatTensor(expert1_features)
        self.expert2_data = torch.FloatTensor(expert2_features)
        self.expert3_data = torch.FloatTensor(expert3_features)
        self.targets = torch.FloatTensor(targets)

        # Raw input for hybrid models (optional)
        if raw_input is not None:
            self.raw_data = torch.FloatTensor(raw_input)
        else:
            self.raw_data = None

        # Ensure all have same number of samples
        assert len(self.expert1_data) == len(self.expert2_data) == len(self.expert3_data) == len(self.targets), \
            "All input features and targets must have the same number of samples"

        if self.raw_data is not None:
            assert len(self.raw_data) == len(self.targets), \
                "Raw input must have same number of samples as targets"

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        batch = {
            'expert1': self.expert1_data[idx],
            'expert2': self.expert2_data[idx],
            'expert3': self.expert3_data[idx],
            'target': self.targets[idx]
        }

        # Include raw input if available (for hybrid models)
        if self.raw_data is not None:
            batch['raw_input'] = self.raw_data[idx]

        return batch


# ==================== Attention Expert Networks ====================
class AttentionExpert(nn.Module):
    """
    Attention-based Expert Network with Temporal Attention Mechanism.

    Uses GRU backbone with Bahdanau-style attention to focus on recent timesteps,
    offsetting the lag introduced by wavelet filtering.
    """

    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2,
                 attention_size=16, expert_name="Expert"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.expert_name = expert_name

        # GRU backbone (return sequences for attention)
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

        # Temporal Attention Mechanism (Bahdanau style)
        # Scores the importance of each timestep
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
            attention_weights: [batch_size, seq_len] - for visualization
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)  # [batch, seq_len, hidden_size]

        # Temporal Attention
        # Compute attention scores for each timestep
        query = self.attention_query(gru_out)  # [batch, seq_len, attention_size]
        key = self.attention_key(gru_out)      # [batch, seq_len, attention_size]

        # Attention score: tanh(W_q * h_t + W_k * h_t)
        attention_logits = torch.tanh(query + key)  # [batch, seq_len, attention_size]
        attention_logits = self.attention_score(attention_logits)  # [batch, seq_len, 1]

        # Softmax to get attention weights
        attention_weights = torch.softmax(attention_logits.squeeze(-1), dim=1)  # [batch, seq_len]

        # Apply attention: weighted sum of GRU hidden states
        context_vector = torch.sum(
            gru_out * attention_weights.unsqueeze(-1),  # broadcast to [batch, seq_len, hidden_size]
            dim=1  # sum over time axis
        )  # [batch, hidden_size]

        # Dropout and output
        context_vector = self.dropout(context_vector)
        prediction = self.fc(context_vector)  # [batch, 1]

        return prediction, attention_weights


class TrendExpert(AttentionExpert):
    """
    Expert 1: Trend prediction (Wavelet-based, no raw data injection).

    Input: [cA4 (Trend Component)] = 1 feature
    This expert specializes in long-term trend prediction from the
    trend component of the MODWT decomposition.
    """
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.2):
        super().__init__(input_size, hidden_size, num_layers, dropout,
                        attention_size=16, expert_name="TrendExpert")


class CyclicExpert(AttentionExpert):
    """
    Expert 2: Cyclic prediction (Wavelet-based, no raw data injection).

    Input: [cD4 (Detail 4), cD3 (Detail 3)] = 2 features
    This expert identifies cyclic patterns from the cyclic frequency bands
    of the MODWT decomposition.
    """
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__(input_size, hidden_size, num_layers, dropout,
                        attention_size=32, expert_name="CyclicExpert")


class HighFreqExpert(AttentionExpert):
    """
    Expert 3: High-frequency/noise prediction (Wavelet-based, no raw data injection).

    Input: [cD2 (Detail 2), cD1 (Detail 1)] = 2 features
    This expert captures high-frequency patterns and noise from the
    high-frequency components of the MODWT decomposition.
    """
    def __init__(self, input_size=2, hidden_size=32, num_layers=2, dropout=0.4):
        super().__init__(input_size, hidden_size, num_layers, dropout,
                        attention_size=16, expert_name="HighFreqExpert")


# ==================== Enhanced Gating Network ====================
class GatingNetwork(nn.Module):
    """
    Gating network for expert weighting in Mixture of Experts.

    Takes the concatenated last timestep features from all three experts
    and learns to weight their contributions.

    Input size can vary:
    - In MODWTMoE (with raw injection): 2 + 3 + 3 = 8
    - In HybridMODWTMoE (wavelet only): 1 + 2 + 2 = 5
    """

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


# ==================== MoE Model ====================
class MODWTMoE(nn.Module):
    """
    Complete MODWT-MoE model with Attention-based Experts and Raw Data Injection.

    Architecture:
    - Expert 1 (Trend):   Accepts [cA4, Raw] (2 features)
    - Expert 2 (Cyclic):  Accepts [cD4, cD3, Raw] (3 features)
    - Expert 3 (HighFreq): Accepts [cD2, cD1, Raw] (3 features)

    Each expert uses temporal attention to focus on recent timesteps,
    while the raw data injection enables immediate shock response.

    Gating Network:
    - Receives concatenated last timestep from all experts: 2 + 3 + 3 = 8 features
    - Learns to weight experts dynamically based on all available information
    """

    def __init__(self):
        super().__init__()
        # Initialize experts with updated input sizes (including raw data)
        self.expert1 = TrendExpert(input_size=2)      # cA4 + Raw
        self.expert2 = CyclicExpert(input_size=3)     # cD4 + cD3 + Raw
        self.expert3 = HighFreqExpert(input_size=3)   # cD2 + cD1 + Raw
        self.gating = GatingNetwork(input_size=8)     # 2 + 3 + 3 = 8

    def forward(self, expert1_input, expert2_input, expert3_input):
        """
        Forward pass with raw data injection.

        Args:
            expert1_input: [batch, seq_len, 2] - Trend expert input
            expert2_input: [batch, seq_len, 3] - Cyclic expert input
            expert3_input: [batch, seq_len, 3] - HighFreq expert input

        Returns:
            output: [batch, 1] - Final prediction
            weights: [batch, 3] - Gating weights for each expert
            predictions: [batch, 3] - Individual expert predictions
            attention_weights: dict - Attention weights from each expert
        """
        # Expert predictions with attention weights
        pred1, attn1 = self.expert1(expert1_input)  # [batch, 1], [batch, seq_len]
        pred2, attn2 = self.expert2(expert2_input)  # [batch, 1], [batch, seq_len]
        pred3, attn3 = self.expert3(expert3_input)  # [batch, 1], [batch, seq_len]

        # Store attention weights for visualization
        attention_weights = {
            'expert1': attn1,
            'expert2': attn2,
            'expert3': attn3
        }

        # Gating input: last timestep features from all experts (with raw data)
        e1_last = expert1_input[:, -1, :]  # [batch, 2] (cA4, Raw)
        e2_last = expert2_input[:, -1, :]  # [batch, 3] (cD4, cD3, Raw)
        e3_last = expert3_input[:, -1, :]  # [batch, 3] (cD2, cD1, Raw)

        # Concatenate all last timestep features
        gate_input = torch.cat([e1_last, e2_last, e3_last], dim=1)  # [batch, 8]

        # Gating weights (learned combination of experts)
        weights = self.gating(gate_input)

        # Weighted combination of expert predictions
        predictions = torch.stack([pred1, pred2, pred3], dim=2)  # [batch, 1, 3]
        output = torch.sum(predictions * weights.unsqueeze(1), dim=2)  # [batch, 1]

        return output, weights, predictions.squeeze(1), attention_weights


# ==================== Residual-MoE Hybrid Architecture ====================
class RawLSTM(nn.Module):
    """
    Branch A: The Baseline LSTM module.

    Learns to predict volatility directly from the raw (undecomposed) signal.
    This provides a strong baseline prediction P_base.

    Input: [batch, seq_len, 1] - Raw centered, scaled volatility
    Output: [batch, 1] - Base prediction
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM backbone
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

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
            x: [batch_size, seq_len, 1]

        Returns:
            prediction: [batch_size, 1] - Base prediction from raw signal
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)  # [batch, seq_len, hidden_size]

        # Use last hidden state
        last_hidden = h_n[-1]  # [batch, hidden_size]

        # Dropout and output
        last_hidden = self.dropout(last_hidden)
        prediction = self.fc(last_hidden)  # [batch, 1]

        return prediction


class HybridMODWTMoE(nn.Module):
    """
    Residual-MoE Hybrid Architecture with Zero-Initialization and Context-Aware Gating.

    Architecture:
    ┌─ Branch A (RawLSTM):       Learns base prediction P_base from raw signal
    ├─ Branch B (MODWTMoE):      Learns residual prediction P_moe from wavelet decomposition
    └─ Final Output:             P = P_base + P_moe (zero-initialized residual learning)

    Key Innovations:
    - Zero Initialization: branch_weight starts at 0.0, preventing MoE noise during early training
    - Context-Aware Gating: Gating network receives temporal context (mean pooling) + magnitude info (raw)
    - Regime Detection: Able to identify high-volatility markets and activate appropriate experts
    """

    def __init__(self):
        super().__init__()
        # Branch A: Raw signal baseline (similar to baseline LSTM)
        self.lstm_branch = RawLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)

        # Branch B: Wavelet-based specialists (unchanged architecture but reverted input sizes)
        # Now experts receive ONLY wavelet components (no raw data injection)
        self.expert1 = TrendExpert(input_size=1)      # cA4 only
        self.expert2 = CyclicExpert(input_size=2)     # cD4, cD3 only
        self.expert3 = HighFreqExpert(input_size=2)   # cD2, cD1 only

        # Context-Aware Gating Network with expanded input
        # Input composition (total = 11):
        # - Last timestep wavelets: 1 + 2 + 2 = 5 features
        # - Mean pooling wavelets: 1 + 2 + 2 = 5 features (temporal context)
        # - Last timestep raw volatility: 1 feature (magnitude awareness)
        self.gating = GatingNetwork(input_size=11, hidden_size=128, num_experts=3, dropout=0.1)

        # Zero-Initialization: Start exactly as LSTM baseline, gradually learn residual
        # This prevents MoE from adding noise during early training
        self.branch_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, raw_input, expert1_input, expert2_input, expert3_input):
        """
        Forward pass for hybrid architecture with context-aware gating.

        Args:
            raw_input: [batch, seq_len, 1] - Raw scaled volatility for LSTM branch
            expert1_input: [batch, seq_len, 1] - Trend (cA4) for Expert 1
            expert2_input: [batch, seq_len, 2] - Cyclic (cD4, cD3) for Expert 2
            expert3_input: [batch, seq_len, 2] - HighFreq (cD2, cD1) for Expert 3

        Returns:
            output: [batch, 1] - Final prediction (P_base + zero_initialized * P_moe)
            base_pred: [batch, 1] - Branch A prediction
            moe_pred: [batch, 1] - Branch B prediction
            weights: [batch, 3] - Gating weights from MoE
            predictions: [batch, 3] - Individual expert predictions
            attention_weights: dict - Attention maps from experts
        """
        # Branch A: LSTM baseline from raw signal
        base_pred = self.lstm_branch(raw_input)  # [batch, 1]

        # Branch B: MoE specialists from wavelet components
        # Expert predictions with attention
        pred1, attn1 = self.expert1(expert1_input)  # [batch, 1]
        pred2, attn2 = self.expert2(expert2_input)  # [batch, 1]
        pred3, attn3 = self.expert3(expert3_input)  # [batch, 1]

        attention_weights = {
            'expert1': attn1,
            'expert2': attn2,
            'expert3': attn3
        }

        # ==================== Context-Aware Gating Input ====================
        # Part 1: Last Timestep Features (regime identification at current time)
        e1_last = expert1_input[:, -1, :]  # [batch, 1]
        e2_last = expert2_input[:, -1, :]  # [batch, 2]
        e3_last = expert3_input[:, -1, :]  # [batch, 2]
        last_wavelets = torch.cat([e1_last, e2_last, e3_last], dim=1)  # [batch, 5]

        # Part 2: Mean Pooling over Time (temporal context and trend direction)
        e1_mean = torch.mean(expert1_input, dim=1)  # [batch, 1]
        e2_mean = torch.mean(expert2_input, dim=1)  # [batch, 2]
        e3_mean = torch.mean(expert3_input, dim=1)  # [batch, 2]
        mean_wavelets = torch.cat([e1_mean, e2_mean, e3_mean], dim=1)  # [batch, 5]

        # Part 3: Raw Data Last Timestep (magnitude awareness for volatility level)
        raw_last = raw_input[:, -1, :]  # [batch, 1]

        # Concatenate all context features
        gate_input = torch.cat([last_wavelets, mean_wavelets, raw_last], dim=1)  # [batch, 11]

        # Expert weighting based on rich context
        weights = self.gating(gate_input)  # [batch, 3]

        # MoE prediction: weighted combination of experts
        predictions = torch.stack([pred1, pred2, pred3], dim=2)  # [batch, 1, 3]
        moe_pred = torch.sum(predictions * weights.unsqueeze(1), dim=2)  # [batch, 1]

        # Final output: Zero-initialized residual learning
        # Starts as pure LSTM (branch_weight = 0.0), gradually learns to use MoE refinement
        output = base_pred + self.branch_weight * moe_pred  # [batch, 1]

        return output, base_pred, moe_pred, weights, predictions.squeeze(1), attention_weights


# ==================== Data Preparation with Walk-Forward Rolling Decomposition ====================
def prepare_modwt_data(df, vol_window=7, lookback=30, forecast_horizon=1,
                       wavelet='db4', level=4,
                       train_ratio=0.80, batch_size=32,
                       use_robust_scaler=False, buffer_size=200):
    """
    準備 MODWT 資料，使用嚴格的 Walk-Forward 滾動分解來消除數據泄漏。

    ============ 數據泄漏解決方案 ============
    問題：全局分解（decomposing the entire train set at once）會導致數據泄漏，
    因為 MODWT 使用非因果小波濾波器，會混合來自過去和未來的信息。

    解決方案：對每個時間點 t，只使用 t 及之前的數據進行本地分解：
    1. 提取一個歷史緩衝區 [t-buffer_size, ..., t]
    2. 對該緩衝區進行 MODWT 分解
    3. 從結果的最後 lookback 個點提取特徵向量
    4. 使用 t+forecast_horizon 的目標值進行監督學習

    這樣確保模型在時間 t 只能看到 t 及其過去的信息，模擬真實的實盤交易環境。
    ==========================================

    Args:
        df: DataFrame with 'Close' column
        vol_window: 波動率計算窗口
        lookback: 特徵序列長度
        forecast_horizon: 預測步長
        wavelet: MODWT 小波類型
        level: MODWT 分解層級
        train_ratio: 訓練集比例
        batch_size: DataLoader batch 大小
        use_robust_scaler: 是否使用 RobustScaler
        buffer_size: 用於本地 MODWT 分解的歷史緩衝區大小
    """

    print("[Data Preparation - Walk-Forward Rolling Decomposition]")

    # Step 1: 計算波動率
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['log_return'].rolling(window=vol_window).std() * np.sqrt(252) * 100
    df = df.dropna().reset_index(drop=True)
    volatility = df['Volatility'].values

    # Step 2: 時間序列切分 (80/20)
    total_len = len(volatility)
    train_split_idx = int(total_len * train_ratio)

    train_volatility = volatility[:train_split_idx]
    test_volatility = volatility[train_split_idx:]

    # Step 3: 計算訓練集的均值和 Scaler（只用訓練集來 fit）
    train_mean = train_volatility.mean()
    train_volatility_centered = train_volatility - train_mean

    # 初始化 Scaler（只在訓練集上 fit）
    ScalerClass = RobustScaler if use_robust_scaler else StandardScaler
    target_scaler = ScalerClass()
    target_scaler.fit(train_volatility_centered.reshape(-1, 1))

    # Step 4: 執行嚴格的 Walk-Forward 滾動分解
    decomposer = MODWTDecomposer(wavelet=wavelet, level=level)

    print(f"  Generating walk-forward features (buffer_size={buffer_size}, lookback={lookback})...")

    # 生成訓練集特徵（包含原始輸入用於混合架構）
    train_expert1, train_expert2, train_expert3, train_raw, train_targets = _generate_wf_features(
        train_volatility, decomposer, lookback, forecast_horizon, buffer_size,
        target_scaler, train_mean, is_train=True
    )

    # 生成測試集特徵
    test_expert1, test_expert2, test_expert3, test_raw, test_targets = _generate_wf_features(
        test_volatility, decomposer, lookback, forecast_horizon, buffer_size,
        target_scaler, train_mean, is_train=False
    )

    print(f"  Train samples: {len(train_targets)} | Test samples: {len(test_targets)}")

    # Step 5: 建立 Dataset 和 DataLoader
    train_dataset = MODWTVolatilityDataset(
        train_expert1, train_expert2, train_expert3, train_targets, raw_input=train_raw
    )

    test_dataset = MODWTVolatilityDataset(
        test_expert1, test_expert2, test_expert3, test_targets, raw_input=test_raw
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"  Train: {len(train_loader)} batches | Test: {len(test_loader)} batches")

    # 返回 scaler 和額外信息用於逆轉換
    scalers = {
        'target': target_scaler,
        'volatility_mean': train_mean
    }

    # 估計訓練集的能量分布（用於信息目的）
    train_energies = {'cA4_trend': 25, 'cD4': 25, 'cD3': 25, 'cD2': 15, 'cD1': 10}  # 估計值

    return train_loader, test_loader, scalers, None, train_energies


def _generate_wf_features(volatility, decomposer, lookback, forecast_horizon, buffer_size,
                          target_scaler, train_mean, is_train=True):
    """
    使用 Walk-Forward 滾動分解為單個數據集生成特徵（無原始數據注入）。

    對每個時間點 t：
    1. 提取緩衝區 [max(0, t-buffer_size), ..., t]
    2. 對緩衝區進行 MODWT 分解
    3. 提取最後 lookback 個特徵
    4. 分別提取原始波動率序列（給混合架構使用）
    5. 創建訓練對 (features, target_at_t+forecast_horizon)

    特徵設計（適用於 Residual-MoE Hybrid）：
    - Expert 1 (Trend):   [cA4] = 1 feature
    - Expert 2 (Cyclic):  [cD4, cD3] = 2 features
    - Expert 3 (HighFreq):[cD2, cD1] = 2 features
    - Raw Input (LSTM):   [Raw Volatility] = 1 feature (separate)

    目的：
    - 讓 LSTM 從原始信號學習基準預測（P_base）
    - 讓 MoE 專家從小波分解學習殘差預測（P_moe）
    - 最終輸出 = P_base + P_moe

    Args:
        volatility: 波動率序列
        decomposer: MODWTDecomposer 實例
        lookback: 特徵序列長度
        forecast_horizon: 預測步長
        buffer_size: 本地分解的緩衝區大小
        target_scaler: 預擬合的目標 scaler
        train_mean: 訓練集均值（用於中心化）
        is_train: 是否為訓練集

    Returns:
        expert1_features: [num_samples, lookback, 1] (cA4 only)
        expert2_features: [num_samples, lookback, 2] (cD4, cD3)
        expert3_features: [num_samples, lookback, 2] (cD2, cD1)
        raw_inputs: [num_samples, lookback, 1] (Raw centered volatility)
        targets: [num_samples, 1]
    """
    expert1_list = []
    expert2_list = []
    expert3_list = []
    raw_list = []
    targets_list = []

    n = len(volatility)

    # 有效的時間範圍：需要足夠的歷史以形成緩衝區，且需要未來的目標值
    min_idx = buffer_size - 1  # 最早可以開始分解的索引（至少有 buffer_size 個點）
    max_idx = n - forecast_horizon  # 最後可以開始分解的索引（需要有目標值）

    for t in range(min_idx, max_idx):
        # Step 1: 提取歷史緩衝區 [t-buffer_size+1, ..., t]
        buffer_start = max(0, t - buffer_size + 1)
        buffer_end = t + 1
        history_buffer = volatility[buffer_start:buffer_end] - train_mean  # 中心化

        # Step 2: 對本地緩衝區進行 MODWT 分解
        try:
            components = decomposer.decompose(history_buffer, use_mra=True)
        except Exception as e:
            print(f"  Warning: MODWT failed at time {t}: {e}. Skipping this sample.")
            continue

        # Step 3: 提取最後 lookback 個點
        # 確保我們有足夠的點
        if len(history_buffer) < lookback:
            continue  # 跳過不足 lookback 的樣本

        cA4 = components['cA4_trend'][-lookback:]
        cD4 = components['cD4'][-lookback:]
        cD3 = components['cD3'][-lookback:]
        cD2 = components['cD2'][-lookback:]
        cD1 = components['cD1'][-lookback:]

        # 調整形狀以適應 lookback
        if len(cA4) < lookback:
            # 用零填充到 lookback（不應該發生，但防守性編程）
            cA4 = np.pad(cA4, (lookback - len(cA4), 0), mode='edge')
            cD4 = np.pad(cD4, (lookback - len(cD4), 0), mode='edge')
            cD3 = np.pad(cD3, (lookback - len(cD3), 0), mode='edge')
            cD2 = np.pad(cD2, (lookback - len(cD2), 0), mode='edge')
            cD1 = np.pad(cD1, (lookback - len(cD1), 0), mode='edge')

        # Step 4: 提取原始波動率（已中心化）的最後 lookback 個點
        raw_lookback = history_buffer[-lookback:]  # [lookback,]

        # Step 5: 構建專家特徵（WAVELET ONLY，不注入原始數據）
        # Expert 1: Trend (cA4) only
        expert1 = cA4.reshape(-1, 1)  # [lookback, 1]

        # Expert 2: Cyclic (cD4, cD3)
        expert2 = np.stack([cD4, cD3], axis=1)  # [lookback, 2]

        # Expert 3: HighFreq (cD2, cD1)
        expert3 = np.stack([cD2, cD1], axis=1)  # [lookback, 2]

        # Raw input for LSTM branch
        raw_input = raw_lookback.reshape(-1, 1)  # [lookback, 1]

        expert1_list.append(expert1)
        expert2_list.append(expert2)
        expert3_list.append(expert3)
        raw_list.append(raw_input)

        # Step 6: 目標值（在 t+forecast_horizon 處的波動率，中心化和縮放）
        target_idx = t + forecast_horizon
        target_vol = volatility[target_idx] - train_mean
        target_scaled = target_scaler.transform([[target_vol]])[0, 0]
        targets_list.append([target_scaled])

    # 轉換為 numpy 數組
    expert1_features = np.array(expert1_list)  # [num_samples, lookback, 1]
    expert2_features = np.array(expert2_list)  # [num_samples, lookback, 2]
    expert3_features = np.array(expert3_list)  # [num_samples, lookback, 2]
    raw_inputs = np.array(raw_list)  # [num_samples, lookback, 1]
    targets = np.array(targets_list)  # [num_samples, 1]

    return expert1_features, expert2_features, expert3_features, raw_inputs, targets


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


# ==================== Training Function ====================
def train_modwt_moe(train_loader, test_loader, num_epochs=100,
                    lr=0.001, delta=1.0, device=DEVICE):
    """訓練 MODWT-MoE 模型 (無 early stopping 版本)"""

    print("[Training]")

    model = MODWTMoE().to(device)
    criterion = HuberLoss(delta=delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {
        'train_loss': [],
        'test_loss': [],
        'train_rmse': [],
        'train_mae': [],
        'epochs': []
    }

    best_train_loss = float('inf')
    best_model_state = None
    best_epoch = 0

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        for batch in train_loader:
            e1 = batch['expert1'].to(device)
            e2 = batch['expert2'].to(device)
            e3 = batch['expert3'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()
            output, weights, expert_preds, _ = model(e1, e2, e3)  # Unpack attention_weights but don't use in training
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.append(output.detach().cpu().numpy())
            train_targets.append(target.cpu().numpy())

        avg_train_loss = np.mean(train_losses)
        train_preds = np.concatenate(train_preds, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)

        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        train_mae = mean_absolute_error(train_targets, train_preds)

        # Test Phase (僅用於監控，不作為停止條件)
        model.eval()
        test_losses = []

        with torch.no_grad():
            for batch in test_loader:
                e1 = batch['expert1'].to(device)
                e2 = batch['expert2'].to(device)
                e3 = batch['expert3'].to(device)
                target = batch['target'].to(device)

                output, _, _, _ = model(e1, e2, e3)
                loss = criterion(output, target)
                test_losses.append(loss.item())

        avg_test_loss = np.mean(test_losses)

        scheduler.step(avg_train_loss)

        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['train_rmse'].append(train_rmse)
        history['train_mae'].append(train_mae)
        history['epochs'].append(epoch + 1)

        # Print every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_train_loss:.4f} | RMSE: {train_rmse:.4f}")

        # 保存最佳模型 (基於 Train Loss)
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()

    model.load_state_dict(best_model_state)
    print(f"\nTraining complete. Best model from epoch {best_epoch} loaded.")

    return model, history, best_epoch


# ==================== Evaluation Function ====================
def evaluate(model, data_loader, device):
    """Evaluate model on a dataset"""
    model.eval()

    all_preds = []
    all_targets = []
    all_expert_preds = []
    all_gating_weights = []
    all_attention_weights = {'expert1': [], 'expert2': [], 'expert3': []}

    with torch.no_grad():
        for batch in data_loader:
            e1 = batch['expert1'].to(device)
            e2 = batch['expert2'].to(device)
            e3 = batch['expert3'].to(device)
            target = batch['target'].to(device)

            output, weights, expert_preds, attention_weights = model(e1, e2, e3)

            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_expert_preds.append(expert_preds.cpu().numpy())
            all_gating_weights.append(weights.cpu().numpy())

            # Collect attention weights from each expert
            all_attention_weights['expert1'].append(attention_weights['expert1'].cpu().numpy())
            all_attention_weights['expert2'].append(attention_weights['expert2'].cpu().numpy())
            all_attention_weights['expert3'].append(attention_weights['expert3'].cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    expert_preds = np.concatenate(all_expert_preds, axis=0)
    gating_weights = np.concatenate(all_gating_weights, axis=0)

    # Concatenate attention weights
    attention_weights_concat = {
        'expert1': np.concatenate(all_attention_weights['expert1'], axis=0),  # [num_samples, seq_len]
        'expert2': np.concatenate(all_attention_weights['expert2'], axis=0),
        'expert3': np.concatenate(all_attention_weights['expert3'], axis=0)
    }

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    direction_true = np.sign(np.diff(targets.flatten()))
    direction_pred = np.sign(np.diff(predictions.flatten()))
    direction_acc = np.mean(direction_true == direction_pred)

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_acc': direction_acc
    }

    return metrics, predictions.flatten(), targets.flatten(), expert_preds, gating_weights, attention_weights_concat

def walk_forward_validation(df,
                            vol_window=7,
                            lookback=30,
                            forecast_horizon=1,
                            wavelet='db4',
                            level=4,
                            train_window=4000,
                            test_window=500,
                            step_size=500,
                            num_epochs=50,
                            batch_size=32,
                            lr=0.001,
                            use_robust_scaler=False,
                            device=DEVICE):
    """
    Walk-Forward Validation for MODWT-MoE

    參數:
        df: 原始數據 DataFrame
        train_window: 每個 fold 的訓練窗口大小 (天數)
        test_window: 每個 fold 的測試窗口大小 (天數)
        step_size: 每次滾動的步長 (天數)
        num_epochs: 每個 fold 訓練多少 epoch
        其他參數與原本相同

    返回:
        results_df: 包含所有 fold 結果的 DataFrame
        all_predictions: 所有 fold 的預測結果
        all_models: 所有訓練好的模型 (可選)
    """

    print("\n[Walk-Forward Validation]")

    total_len = len(df)
    max_start = total_len - train_window - test_window
    num_folds = max_start // step_size + 1

    print(f"  Window: train={train_window} | test={test_window} | step={step_size}")
    print(f"  Total: {total_len} days | Folds: {num_folds}")

    all_results = []
    all_predictions = []
    all_models = []

    for fold in range(num_folds):
        fold_start_time = pd.Timestamp.now()

        # 計算這個 fold 的時間範圍
        train_start = fold * step_size
        train_end = train_start + train_window
        test_end = train_end + test_window

        # 檢查是否超出範圍
        if test_end > total_len:
            print(f"\n[Fold {fold+1}] Insufficient data, skipping...")
            break

        print(f"\nFold {fold+1}/{num_folds}")

        # 切分數據
        fold_df = df.iloc[train_start:test_end].copy().reset_index(drop=True)

        # 準備數據 (內部會再按 train_window/(train_window+test_window) 切分)
        train_ratio = train_window / (train_window + test_window)

        try:
            train_loader, test_loader, scalers, components, energies = prepare_modwt_data(
                fold_df,
                vol_window=vol_window,
                lookback=lookback,
                forecast_horizon=forecast_horizon,
                wavelet=wavelet,
                level=level,
                train_ratio=train_ratio,
                batch_size=batch_size,
                use_robust_scaler=use_robust_scaler
            )

            # 訓練模型
            model, history, best_epoch = train_modwt_moe(
                train_loader,
                test_loader,
                num_epochs=num_epochs,
                lr=lr,
                device=device
            )

            # 評估
            test_metrics, test_preds, test_targets, test_expert_preds, test_gating_weights, test_attention_weights = evaluate(
                model, test_loader, device
            )

            # Inverse transform
            target_scaler = scalers['target']
            volatility_mean = scalers['volatility_mean']

            test_preds_centered = target_scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
            test_targets_centered = target_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()

            test_preds_original = test_preds_centered + volatility_mean
            test_targets_original = test_targets_centered + volatility_mean

            rmse_original = np.sqrt(mean_squared_error(test_targets_original, test_preds_original))
            mae_original = mean_absolute_error(test_targets_original, test_preds_original)
            r2_original = r2_score(test_targets_original, test_preds_original)

            # 保存結果
            fold_result = {
                'fold': fold + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'best_epoch': best_epoch,
                'rmse': rmse_original,
                'mae': mae_original,
                'r2': r2_original,
                'direction_acc': test_metrics['direction_acc'],
                'expert1_weight': test_gating_weights[:, 0].mean(),
                'expert2_weight': test_gating_weights[:, 1].mean(),
                'expert3_weight': test_gating_weights[:, 2].mean(),
            }

            all_results.append(fold_result)

            # 保存預測結果
            all_predictions.append({
                'fold': fold + 1,
                'predictions': test_preds_original,
                'targets': test_targets_original,
                'gating_weights': test_gating_weights,
                'expert_preds': test_expert_preds
            })

            # 可選: 保存模型
            all_models.append({
                'fold': fold + 1,
                'model': model.state_dict(),
                'scalers': scalers
            })

            # 打印這個 fold 的結果
            print(f"  RMSE: {rmse_original:.4f}% | MAE: {mae_original:.4f}% | R²: {r2_original:.4f}")

        except Exception as e:
            print(f"\n[Error] Fold {fold+1} failed with error: {e}")
            continue

    # 匯總所有結果
    results_df = pd.DataFrame(all_results)

    print("\n[WFV Summary]")
    print(f"  RMSE:  {results_df['rmse'].mean():.4f}% ± {results_df['rmse'].std():.4f}%")
    print(f"  MAE:   {results_df['mae'].mean():.4f}% ± {results_df['mae'].std():.4f}%")
    print(f"  R²:    {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}")
    print(f"  Dir Acc: {results_df['direction_acc'].mean()*100:.2f}% ± {results_df['direction_acc'].std()*100:.2f}%")

    return results_df, all_predictions, all_models

# ==================== Visualization Functions ====================
def plot_training_history(history, save_path='training_history.png'):
    """Plot training history (Train/Test only)"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    epochs = history['epochs']

    # Loss curve
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['test_loss'], 'r-', label='Test Loss (Monitor)', linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training/Test Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(alpha=0.3)

    # Train RMSE
    axes[0, 1].plot(epochs, history['train_rmse'], 'purple', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('RMSE', fontsize=12)
    axes[0, 1].set_title('Training RMSE', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # Train MAE
    axes[1, 0].plot(epochs, history['train_mae'], 'orange', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('MAE', fontsize=12)
    axes[1, 0].set_title('Training MAE', fontsize=14, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # Log scale loss
    axes[1, 1].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, history['test_loss'], 'r-', label='Test (Monitor)', linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title('Loss Comparison (Log Scale)', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_predictions(true_values, predictions, save_path='predictions.png'):
    """Plot predictions vs actual"""

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    axes[0].plot(true_values, label='Actual', color='blue', linewidth=1.5, alpha=0.7)
    axes[0].plot(predictions, label='Predicted', color='red', linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('Volatility (%)', fontsize=12)
    axes[0].set_title('Volatility Predictions vs Actual', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

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
    plt.close()


def plot_gating_weights(gating_weights, save_path='gating_weights.png'):
    """Plot gating weights over time"""

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    time = np.arange(len(gating_weights))

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
    plt.close()


def plot_attention_maps(attention_weights, save_path='attention_maps.png'):
    """
    Visualize temporal attention mechanisms from all three experts.

    Shows how each expert focuses on different timesteps in the lookback window (0-30 days).
    This demonstrates that the attention mechanism helps offset wavelet-induced lag by
    learning to focus more on recent timesteps.

    Args:
        attention_weights: Dict with keys 'expert1', 'expert2', 'expert3'
                          Each is [num_samples, seq_len] array
        save_path: Where to save the figure
    """

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Extract attention weights
    attn1 = attention_weights['expert1']  # [num_samples, 30]
    attn2 = attention_weights['expert2']  # [num_samples, 30]
    attn3 = attention_weights['expert3']  # [num_samples, 30]

    # Compute mean attention across all samples
    mean_attn1 = attn1.mean(axis=0)
    mean_attn2 = attn2.mean(axis=0)
    mean_attn3 = attn3.mean(axis=0)

    # Compute standard deviation for uncertainty
    std_attn1 = attn1.std(axis=0)
    std_attn2 = attn2.std(axis=0)
    std_attn3 = attn3.std(axis=0)

    timesteps = np.arange(len(mean_attn1))

    # ==================== Subplot 1: Expert 1 (Trend) ====================
    axes[0, 0].plot(timesteps, mean_attn1, 'g-', linewidth=2.5, label='Mean Attention')
    axes[0, 0].fill_between(timesteps, mean_attn1 - std_attn1, mean_attn1 + std_attn1,
                            color='green', alpha=0.3, label='±1 Std Dev')
    axes[0, 0].set_xlabel('Lookback Days (0=oldest, 29=most recent)', fontsize=11)
    axes[0, 0].set_ylabel('Attention Weight', fontsize=11)
    axes[0, 0].set_title('Expert 1 (Trend) - Temporal Attention', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].set_xlim([-0.5, len(mean_attn1) - 0.5])

    # ==================== Subplot 2: Expert 2 (Cyclic) ====================
    axes[0, 1].plot(timesteps, mean_attn2, 'b-', linewidth=2.5, label='Mean Attention')
    axes[0, 1].fill_between(timesteps, mean_attn2 - std_attn2, mean_attn2 + std_attn2,
                            color='blue', alpha=0.3, label='±1 Std Dev')
    axes[0, 1].set_xlabel('Lookback Days (0=oldest, 29=most recent)', fontsize=11)
    axes[0, 1].set_ylabel('Attention Weight', fontsize=11)
    axes[0, 1].set_title('Expert 2 (Cyclic) - Temporal Attention', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].set_xlim([-0.5, len(mean_attn2) - 0.5])

    # ==================== Subplot 3: Expert 3 (High-Freq) ====================
    axes[1, 0].plot(timesteps, mean_attn3, 'orange', linewidth=2.5, label='Mean Attention')
    axes[1, 0].fill_between(timesteps, mean_attn3 - std_attn3, mean_attn3 + std_attn3,
                            color='orange', alpha=0.3, label='±1 Std Dev')
    axes[1, 0].set_xlabel('Lookback Days (0=oldest, 29=most recent)', fontsize=11)
    axes[1, 0].set_ylabel('Attention Weight', fontsize=11)
    axes[1, 0].set_title('Expert 3 (High-Freq) - Temporal Attention', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].set_xlim([-0.5, len(mean_attn3) - 0.5])

    # ==================== Subplot 4: All Experts Combined ====================
    axes[1, 1].plot(timesteps, mean_attn1, 'g-', linewidth=2, label='Expert 1 (Trend)', alpha=0.8)
    axes[1, 1].plot(timesteps, mean_attn2, 'b-', linewidth=2, label='Expert 2 (Cyclic)', alpha=0.8)
    axes[1, 1].plot(timesteps, mean_attn3, color='orange', linewidth=2, label='Expert 3 (High-Freq)', alpha=0.8)
    axes[1, 1].set_xlabel('Lookback Days (0=oldest, 29=most recent)', fontsize=11)
    axes[1, 1].set_ylabel('Attention Weight', fontsize=11)
    axes[1, 1].set_title('All Experts - Temporal Attention Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(fontsize=10, loc='best')
    axes[1, 1].set_xlim([-0.5, len(mean_attn1) - 0.5])

    # Add overall title
    fig.suptitle('Temporal Attention Mechanisms: Focus on Recent Timesteps Offsets Wavelet Lag',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Attention maps saved to {save_path}")


def save_results_to_csv(targets, predictions, gating_weights, expert_preds, scalers, save_path):
    """Save results to CSV"""

    target_scaler = scalers['target']
    volatility_mean = scalers['volatility_mean']

    targets_centered = target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
    predictions_centered = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # 加回均值
    targets_original = targets_centered + volatility_mean
    predictions_original = predictions_centered + volatility_mean

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

    return results_df


def analyze_gating_dynamics(gating_weights, volatility):
    """分析 Gating 動態"""

    low_vol = volatility < np.percentile(volatility, 33)
    mid_vol = (volatility >= np.percentile(volatility, 33)) & (volatility <= np.percentile(volatility, 67))
    high_vol = volatility > np.percentile(volatility, 67)

    print("[Gating Dynamics by Volatility Regime]")
    print(f"  Low:    E1={gating_weights[low_vol, 0].mean():.3f} | E2={gating_weights[low_vol, 1].mean():.3f} | E3={gating_weights[low_vol, 2].mean():.3f}")
    print(f"  Medium: E1={gating_weights[mid_vol, 0].mean():.3f} | E2={gating_weights[mid_vol, 1].mean():.3f} | E3={gating_weights[mid_vol, 2].mean():.3f}")
    print(f"  High:   E1={gating_weights[high_vol, 0].mean():.3f} | E2={gating_weights[high_vol, 1].mean():.3f} | E3={gating_weights[high_vol, 2].mean():.3f}")


def plot_gating_by_regime(gating_weights, targets_original, save_path='gating_dynamics_by_regime.png'):
    """畫出不同波動區制下的 Gating 權重"""

    low_vol = targets_original < np.percentile(targets_original, 33)
    mid_vol = (targets_original >= np.percentile(targets_original, 33)) & \
              (targets_original <= np.percentile(targets_original, 67))
    high_vol = targets_original > np.percentile(targets_original, 67)

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

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
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

    for i, (e1, e2, e3) in enumerate(zip(expert1_means, expert2_means, expert3_means)):
        axes[0].text(i, e1/2, f'{e1:.1%}', ha='center', va='center',
                     fontweight='bold', color='white', fontsize=10)
        axes[0].text(i, e1 + e2/2, f'{e2:.1%}', ha='center', va='center',
                     fontweight='bold', color='white', fontsize=10)
        axes[0].text(i, e1 + e2 + e3/2, f'{e3:.1%}', ha='center', va='center',
                     fontweight='bold', color='white', fontsize=10)

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
    plt.close()


def plot_wfv_results(results_df, save_path='../results/wfv_summary.png'):
    """視覺化 Walk-Forward Validation 結果"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    folds = results_df['fold']

    # RMSE 趨勢
    axes[0, 0].plot(folds, results_df['rmse'], marker='o', linewidth=2, markersize=8)
    axes[0, 0].axhline(results_df['rmse'].mean(), color='r', linestyle='--',
                       label=f"Mean: {results_df['rmse'].mean():.4f}%")
    axes[0, 0].set_xlabel('Fold', fontsize=12)
    axes[0, 0].set_ylabel('RMSE (%)', fontsize=12)
    axes[0, 0].set_title('RMSE Across Folds', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # MAE 趨勢
    axes[0, 1].plot(folds, results_df['mae'], marker='s', linewidth=2,
                    markersize=8, color='orange')
    axes[0, 1].axhline(results_df['mae'].mean(), color='r', linestyle='--',
                       label=f"Mean: {results_df['mae'].mean():.4f}%")
    axes[0, 1].set_xlabel('Fold', fontsize=12)
    axes[0, 1].set_ylabel('MAE (%)', fontsize=12)
    axes[0, 1].set_title('MAE Across Folds', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # R² 趨勢
    axes[0, 2].plot(folds, results_df['r2'], marker='^', linewidth=2,
                    markersize=8, color='green')
    axes[0, 2].axhline(results_df['r2'].mean(), color='r', linestyle='--',
                       label=f"Mean: {results_df['r2'].mean():.4f}")
    axes[0, 2].set_xlabel('Fold', fontsize=12)
    axes[0, 2].set_ylabel('R²', fontsize=12)
    axes[0, 2].set_title('R² Across Folds', fontsize=14, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # Direction Accuracy
    axes[1, 0].plot(folds, results_df['direction_acc']*100, marker='d',
                    linewidth=2, markersize=8, color='purple')
    axes[1, 0].axhline(results_df['direction_acc'].mean()*100, color='r',
                       linestyle='--', label=f"Mean: {results_df['direction_acc'].mean()*100:.2f}%")
    axes[1, 0].set_xlabel('Fold', fontsize=12)
    axes[1, 0].set_ylabel('Direction Accuracy (%)', fontsize=12)
    axes[1, 0].set_title('Direction Accuracy Across Folds', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Gating Weights
    axes[1, 1].plot(folds, results_df['expert1_weight'], marker='o',
                    linewidth=2, label='Expert 1 (Trend)', color='green')
    axes[1, 1].plot(folds, results_df['expert2_weight'], marker='s',
                    linewidth=2, label='Expert 2 (Cyclic)', color='blue')
    axes[1, 1].plot(folds, results_df['expert3_weight'], marker='^',
                    linewidth=2, label='Expert 3 (High-Freq)', color='orange')
    axes[1, 1].set_xlabel('Fold', fontsize=12)
    axes[1, 1].set_ylabel('Average Gating Weight', fontsize=12)
    axes[1, 1].set_title('Gating Weights Across Folds', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    # Box plot for RMSE
    bp = axes[1, 2].boxplot([results_df['rmse']], labels=['RMSE'],
                            patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[1, 2].set_ylabel('RMSE (%)', fontsize=12)
    axes[1, 2].set_title('RMSE Distribution', fontsize=14, fontweight='bold')
    axes[1, 2].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_wfv_predictions(all_predictions, save_path='../results/wfv_predictions.png'):
    """視覺化所有 fold 的預測結果"""

    fig, axes = plt.subplots(len(all_predictions), 1,
                             figsize=(16, 4*len(all_predictions)))

    if len(all_predictions) == 1:
        axes = [axes]

    for i, pred_data in enumerate(all_predictions):
        fold = pred_data['fold']
        preds = pred_data['predictions']
        targets = pred_data['targets']

        axes[i].plot(targets, label='Actual', color='blue', linewidth=1.5, alpha=0.7)
        axes[i].plot(preds, label='Predicted', color='red', linewidth=1.5, alpha=0.7)
        axes[i].set_xlabel('Time Step', fontsize=11)
        axes[i].set_ylabel('Volatility (%)', fontsize=11)
        axes[i].set_title(f'Fold {fold} Predictions', fontsize=13, fontweight='bold')
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==================== Main Execution ====================
if __name__ == "__main__":
    os.makedirs('../results', exist_ok=True)

    print("[Loading Data]")
    df = pd.read_csv("../dataset/USD_TWD.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  {len(df)} days loaded")

    # ==================== 配置區 ====================
    COMPARE_WAVELETS = True
    DEFAULT_WAVELET = 'haar'
    WALK_FORWARD_VALIDATION = False

    wavelets_to_test = ['haar', 'db2', 'db4', 'db6', 'sym4', 'sym8', 'coif1'] if COMPARE_WAVELETS else [DEFAULT_WAVELET]
    print(f"[Mode] {'Wavelet Comparison' if COMPARE_WAVELETS else 'Single Wavelet'}")

    results_comparison = []
    all_training_histories = {}
    all_test_results = {}

    for wavelet in wavelets_to_test:
        print(f"\n[Testing Wavelet: {wavelet}]")

        # Prepare data with centering (80/20 split)
        train_loader, test_loader, scalers, components, energies = prepare_modwt_data(
            df,
            vol_window=7,
            lookback=30,
            forecast_horizon=1,
            wavelet=wavelet,  # ← 使用當前小波
            level=4,
            train_ratio=0.80,
            use_robust_scaler=False
        )

        # Train model
        trained_model, training_history, best_epoch = train_modwt_moe(
            train_loader,
            test_loader,
            num_epochs=50,
            lr=0.001,
            device=DEVICE
        )

        # Evaluate on test set
        test_metrics, test_preds, test_targets, test_expert_preds, test_gating_weights, test_attention_weights = evaluate(
            trained_model, test_loader, DEVICE
        )

        # Inverse transform
        target_scaler = scalers['target']
        volatility_mean = scalers['volatility_mean']

        test_preds_centered = target_scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
        test_targets_centered = target_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()

        test_preds_original = test_preds_centered + volatility_mean
        test_targets_original = test_targets_centered + volatility_mean

        rmse_original = np.sqrt(mean_squared_error(test_targets_original, test_preds_original))
        mae_original = mean_absolute_error(test_targets_original, test_preds_original)
        r2_original = r2_score(test_targets_original, test_preds_original)

        # 保存結果
        results_comparison.append({
            'wavelet': wavelet,
            'rmse': rmse_original,
            'mae': mae_original,
            'r2': r2_original,
            'direction_acc': test_metrics['direction_acc'],
            'best_epoch': best_epoch
        })

        # 保存詳細結果（用於後續視覺化）
        all_training_histories[wavelet] = training_history
        all_test_results[wavelet] = {
            'preds': test_preds_original,
            'targets': test_targets_original,
            'gating_weights': test_gating_weights,
            'expert_preds': test_expert_preds,
            'attention_weights': test_attention_weights,
            'scalers': scalers,
            'energies': energies,
            'metrics': test_metrics
        }

        print(f"  RMSE: {rmse_original:.4f}% | MAE: {mae_original:.4f}% | R²: {r2_original:.4f}")

    # ==================== 結果比較與選擇 ====================
    comparison_df = pd.DataFrame(results_comparison)
    comparison_df = comparison_df.sort_values('rmse')

    if COMPARE_WAVELETS:
        print("\n[Wavelet Comparison Results]")

        # 找出最佳小波
        best_wavelet = comparison_df.iloc[0]['wavelet']
        best_rmse = comparison_df.iloc[0]['rmse']
        best_mae = comparison_df.iloc[0]['mae']
        best_r2 = comparison_df.iloc[0]['r2']

        print(f"  Best: {best_wavelet} (RMSE: {best_rmse:.4f}% | MAE: {best_mae:.4f}% | R²: {best_r2:.4f})")

        # 保存比較結果
        comparison_df.to_csv('../results/wavelet_comparison.csv', index=False)

        # 視覺化比較
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # RMSE 比較
        axes[0, 0].bar(comparison_df['wavelet'], comparison_df['rmse'], color='steelblue', alpha=0.8)
        axes[0, 0].set_ylabel('RMSE (%)', fontsize=12)
        axes[0, 0].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # MAE 比較
        axes[0, 1].bar(comparison_df['wavelet'], comparison_df['mae'], color='coral', alpha=0.8)
        axes[0, 1].set_ylabel('MAE (%)', fontsize=12)
        axes[0, 1].set_title('MAE Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)

        # R² 比較
        axes[1, 0].bar(comparison_df['wavelet'], comparison_df['r2'], color='lightgreen', alpha=0.8)
        axes[1, 0].set_ylabel('R²', fontsize=12)
        axes[1, 0].set_title('R² Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Direction Accuracy 比較
        axes[1, 1].bar(comparison_df['wavelet'], comparison_df['direction_acc']*100, color='plum', alpha=0.8)
        axes[1, 1].set_ylabel('Direction Accuracy (%)', fontsize=12)
        axes[1, 1].set_title('Direction Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('../results/wavelet_comparison_charts.png', dpi=300, bbox_inches='tight')
        plt.close()

    else:
        best_wavelet = DEFAULT_WAVELET

    # ==================== 使用最佳小波的結果進行詳細分析 ====================
    print(f"\n[Detailed Analysis - Wavelet: {best_wavelet}]")

    # 取得最佳小波的結果
    best_results = all_test_results[best_wavelet]
    test_preds_original = best_results['preds']
    test_targets_original = best_results['targets']
    test_gating_weights = best_results['gating_weights']
    test_expert_preds = best_results['expert_preds']
    test_attention_weights = best_results['attention_weights']
    scalers = best_results['scalers']
    energies = best_results['energies']
    test_metrics = best_results['metrics']
    training_history = all_training_histories[best_wavelet]

    # 重新計算 test_preds 和 test_targets (scaled版本，用於存檔)
    target_scaler = scalers['target']
    volatility_mean = scalers['volatility_mean']
    test_preds = target_scaler.transform((test_preds_original - volatility_mean).reshape(-1, 1)).flatten()
    test_targets = target_scaler.transform((test_targets_original - volatility_mean).reshape(-1, 1)).flatten()

    # 計算指標
    rmse_original = np.sqrt(mean_squared_error(test_targets_original, test_preds_original))
    mae_original = mean_absolute_error(test_targets_original, test_preds_original)
    r2_original = r2_score(test_targets_original, test_preds_original)

    print(f"[Test Set Performance]")
    print(f"  RMSE: {rmse_original:.4f}% | MAE: {mae_original:.4f}% | R²: {r2_original:.4f} | Dir Acc: {test_metrics['direction_acc']*100:.2f}%")

    # Visualizations & Results (使用最佳小波)
    plot_training_history(training_history, f'../results/training_history_{best_wavelet}.png')
    plot_predictions(test_targets_original, test_preds_original, f'../results/test_predictions_{best_wavelet}.png')
    plot_gating_weights(test_gating_weights, f'../results/test_gating_weights_{best_wavelet}.png')
    plot_attention_maps(test_attention_weights, f'../results/test_attention_maps_{best_wavelet}.png')

    test_results_df = save_results_to_csv(
        test_targets, test_preds, test_gating_weights,
        test_expert_preds, scalers, f'../results/test_results_{best_wavelet}.csv'
    )

    # Analysis (使用最佳小波)
    print("[Gating Dynamics]")
    analyze_gating_dynamics(test_gating_weights, test_targets_original)
    plot_gating_by_regime(test_gating_weights, test_targets_original,
                          f'../results/test_gating_dynamics_by_regime_{best_wavelet}.png')

    print(f"\n[Summary]")
    print(f"  Wavelet: {best_wavelet} | RMSE: {rmse_original:.4f}% | MAE: {mae_original:.4f}% | R²: {r2_original:.4f}")
    print(f"  Results saved to ../results/")

    # # ==================== Walk-Forward Validation ====================
    if WALK_FORWARD_VALIDATION:
        results_df, all_predictions, all_models = walk_forward_validation(
            df,
            vol_window=7,
            lookback=30,
            forecast_horizon=1,
            wavelet='db4',
            level=4,
            train_window=4000,      # 每次用 4000 天訓練
            test_window=500,        # 預測 500 天
            step_size=500,          # 每次滾動 500 天
            num_epochs=50,          # 每個 fold 訓練 50 epoch (節省時間)
            batch_size=32,
            lr=0.001,
            use_robust_scaler=False,
            device=DEVICE
        )

        # 保存結果
        results_df.to_csv('../results/wfv_results.csv', index=False)

        # 視覺化
        plot_wfv_results(results_df, '../results/wfv_summary.png')
        plot_wfv_predictions(all_predictions, '../results/wfv_predictions.png')

        # 保存所有預測結果
        for i, pred_data in enumerate(all_predictions):
            fold = pred_data['fold']
            pred_df = pd.DataFrame({
                'Fold': fold,
                'True_Volatility': pred_data['targets'],
                'Predicted_Volatility': pred_data['predictions'],
                'Expert1_Weight': pred_data['gating_weights'][:, 0],
                'Expert2_Weight': pred_data['gating_weights'][:, 1],
                'Expert3_Weight': pred_data['gating_weights'][:, 2],
            })
            pred_df.to_csv(f'../results/wfv_fold{fold}_predictions.csv', index=False)

        # 合併Fold 2-5的殘差（排除warm-up期）
        residuals_stable = np.concatenate([
            all_predictions[1]['targets'] - all_predictions[1]['predictions'],  # Fold 2
            all_predictions[2]['targets'] - all_predictions[2]['predictions'],  # Fold 3
            all_predictions[3]['targets'] - all_predictions[3]['predictions'],  # Fold 4
            all_predictions[4]['targets'] - all_predictions[4]['predictions'],  # Fold 5
        ])

        # 1. Normality test
        stat_shapiro, p_shapiro = stats.shapiro(residuals_stable)
        is_normal = "YES" if p_shapiro > 0.05 else "NO"

        # 2. Autocorrelation test
        lb_test = acorr_ljungbox(residuals_stable, lags=[10, 20], return_df=True)
        is_independent = "YES" if lb_test['lb_pvalue'].min() > 0.05 else "NO"

        print(f"  Mean: {residuals_stable.mean():.6f}% | Std: {residuals_stable.std():.6f}%")
        print(f"  Normal: {is_normal} | Independent: {is_independent}")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Time series
        axes[0, 0].plot(residuals_stable, alpha=0.7, linewidth=1)
        axes[0, 0].axhline(0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Residuals Over Time (Fold 2-5)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Residual (%)')
        axes[0, 0].grid(alpha=0.3)

        # Distribution
        axes[0, 1].hist(residuals_stable, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 1].axvline(residuals_stable.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean={residuals_stable.mean():.4f}%')
        axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Residual (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3, axis='y')

        # ACF
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals_stable, lags=40, ax=axes[1, 0])
        axes[1, 0].set_title('ACF - Autocorrelation Check', fontsize=12, fontweight='bold')

        # Q-Q plot
        stats.probplot(residuals_stable, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot - Normality Check', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig('../results/residual_diagnostics_stable.png', dpi=300, bbox_inches='tight')
        plt.close()