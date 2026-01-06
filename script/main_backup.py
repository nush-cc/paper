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
# from scipy import stats
# from statsmodels.stats.diagnostic import acorr_ljungbox

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
# SEED = 5592

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
                components[f'cD{i + 1}'] = mra[i]
            components[f'cA{self.level}_trend'] = mra[-1]
        else:
            components = {}
            for i in range(self.level):
                components[f'cD{i + 1}'] = w[i]
            components[f'cA{self.level}_trend'] = w[-1]

        self.components_names = list(components.keys())
        return components

    def get_component_energies(self, components):
        """Calculate energy percentage for each component"""
        total_energy = sum(np.sum(comp ** 2) for comp in components.values())
        energies = {}
        for name, comp in components.items():
            energy_pct = np.sum(comp ** 2) / total_energy * 100
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
        修正後的 Attention：讓最後一個時間點 (Query) 去關注過去所有時間點 (Keys)
        """
        # gru_out: [batch, seq_len, hidden_size] -> 這是 Keys
        # h_n:     [num_layers, batch, hidden_size] -> 這是最後的狀態
        gru_out, h_n = self.gru(x)

        # 1. 定義 Query (取最後一層的最後一個時間點狀態)
        # h_n[-1] 形狀是 [batch, hidden_size]
        # 我們需要把它擴充成 [batch, seq_len, hidden_size] 以便跟序列做運算，或者直接用廣播
        query_vector = h_n[-1]

        # 2. 計算 Attention Score (Bahdanau Style)
        # Score = V * tanh(Wq * Query + Wk * Keys)

        # [batch, 1, attn_size]
        query_proj = self.attention_query(query_vector).unsqueeze(1)

        # [batch, seq_len, attn_size]
        key_proj = self.attention_key(gru_out)

        # 兩者相加 (PyTorch 會自動廣播 Query 到每一個 Time Step)
        # 意義：比較 "今天狀態" 與 "過去每一天狀態"
        attention_logits = torch.tanh(query_proj + key_proj)

        # [batch, seq_len, 1]
        attention_logits = self.attention_score(attention_logits)

        # 3. Softmax (歸一化)
        attention_weights = torch.softmax(attention_logits.squeeze(-1), dim=1)

        # 4. 加權總和
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
class HybridDirectionalLoss(nn.Module):
    """
        Differentiable Directional Loss using Tanh approximation.
        Solves the 'Zero Gradient' problem of torch.sign().
        """

    def __init__(self, delta=1.0, direction_weight=0.5, penalty_scale=10.0, sharpness=5.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.direction_weight = direction_weight
        self.penalty_scale = penalty_scale
        self.sharpness = sharpness  # 控制 tanh 有多陡峭 (越像 sign)

    def forward(self, pred, target, prev_value):
        # 1. Magnitude Loss
        loss_val = self.huber(pred, target)

        # 2. Difference Calculation
        true_diff = target - prev_value
        pred_diff = pred - prev_value

        # 3. Soft Direction Calculation (The Fix!)
        # 使用 tanh 模擬 sign，保留梯度
        # sharpness 越大，越接近 sign；越小越平滑
        true_sign_soft = torch.tanh(true_diff * self.sharpness)
        pred_sign_soft = torch.tanh(pred_diff * self.sharpness)

        # 4. Correlation Penalty
        # 我們希望 pred_diff 和 true_diff 的方向一致
        # 即：Maximize (pred_sign * true_sign) -> Minimize -(pred_sign * true_sign)
        # 範圍：-1 (完全一致) 到 +1 (完全相反)
        # 我們將其移位：1 - (product)，範圍變 0 (一致) 到 2 (相反)

        direction_loss = 1 - (pred_sign_soft * true_sign_soft)

        # 這裡不需要再乘 abs(true_diff)，因為 tanh 已經包含了 magnitude 的梯度引導
        # 但我們可以加權，讓大波動的錯誤懲罰更重
        weighted_direction_loss = torch.mean(direction_loss * torch.abs(true_diff) * self.penalty_scale)

        return (1 - self.direction_weight) * loss_val + self.direction_weight * weighted_direction_loss


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
class AdvancedResidualMODWTMoE(nn.Module):
    def __init__(self):
        super().__init__()

        # ====================================================
        # Branch A: LSTM Baseline
        # [CHANGE] input_size=3 (Vol, RSI, Return)
        # ====================================================
        self.base_branch = RawLSTM(input_size=3, hidden_size=64, num_layers=2, dropout=0.2)

        # Branch B: Experts (不變，還是吃 Wavelets)
        self.expert1 = TrendExpert(input_size=1)
        self.expert2 = CyclicExpert(input_size=2)
        self.expert3 = HighFreqExpert(input_size=2)

        # ====================================================
        # Gating Network
        # [CHANGE] Input size increased
        # Last Wavelets(5) + Mean Wavelets(5) + Raw Last(3) = 13
        # ====================================================
        self.gating = GatingNetwork(input_size=13, hidden_size=128, num_experts=3, dropout=0.1)

        # [CHANGE] Meta-Gate Input size = 13
        self.meta_gate = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, raw_input, expert1_input, expert2_input, expert3_input):
        # 1. Anchor: 這裡要注意！
        # raw_input 是 [batch, seq, 3]，我們預測的目標是 Volatility (第 0 個 feature)
        # 所以 prev_value 只要取第 0 個 channel
        prev_value = raw_input[:, -1, 0:1]  # [batch, 1]

        # 2. Base Branch: LSTM 吃所有特徵 (Vol, RSI, Ret) 來預測 Delta
        base_delta = self.base_branch(raw_input)

        # 3. Experts (不變)
        pred1, attn1 = self.expert1(expert1_input)
        pred2, attn2 = self.expert2(expert2_input)
        pred3, attn3 = self.expert3(expert3_input)

        attention_weights = {'expert1': attn1, 'expert2': attn2, 'expert3': attn3}

        # 4. Context Preparation
        e1_last = expert1_input[:, -1, :]
        e2_last = expert2_input[:, -1, :]
        e3_last = expert3_input[:, -1, :]
        last_wavelets = torch.cat([e1_last, e2_last, e3_last], dim=1)

        e1_mean = torch.mean(expert1_input, dim=1)
        e2_mean = torch.mean(expert2_input, dim=1)
        e3_mean = torch.mean(expert3_input, dim=1)
        mean_wavelets = torch.cat([e1_mean, e2_mean, e3_mean], dim=1)

        # [CHANGE] Raw data last timestep (包含 Vol, RSI, Ret)
        raw_last = raw_input[:, -1, :]  # [batch, 3]

        # Combine Context: 5 + 5 + 3 = 13
        context_input = torch.cat([last_wavelets, mean_wavelets, raw_last], dim=1)

        # 5. Gating & Meta-Gating
        expert_weights = self.gating(context_input)
        expert_preds_stack = torch.stack([pred1, pred2, pred3], dim=2)
        moe_delta = torch.sum(expert_preds_stack * expert_weights.unsqueeze(1), dim=2)

        alpha = self.meta_gate(context_input)

        # 6. Final Prediction
        total_delta = base_delta + (alpha * moe_delta)
        output = prev_value + total_delta

        return output, total_delta, moe_delta, expert_weights, expert_preds_stack.squeeze(1), attention_weights, alpha


# ==================== Data Preparation ====================
def prepare_modwt_data(df, vol_window=7, lookback=30, forecast_horizon=1,
                       wavelet='db4', level=4,
                       train_ratio=0.80, batch_size=32,
                       use_robust_scaler=False, buffer_size=200, run_mode="NORMAL"):
    """Prepare MODWT data WITH RSI & RETURNS Features"""

    print("[Data Preparation] MODWT + RSI + Returns (Feature Enhanced)")

    # Step 1: Calculate volatility & Basic Features
    df = df.copy()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['log_return'].rolling(vol_window).std() * np.sqrt(252) * 100

    # --- [NEW] Calculate RSI (14-day) ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)  # Fill NaN with neutral 50

    # Drop NaN from Volatility calc
    df = df.dropna().reset_index(drop=True)

    # Extract Raw Features
    volatility = df['Volatility'].values
    rsi = df['RSI'].values
    returns = df['log_return'].values

    print(f"  Volatility mean: {volatility.mean():.4f}%")

    # Step 2: Split into train/test
    total_len = len(volatility)
    train_split_idx = int(total_len * train_ratio)

    # Split all features
    train_vol = volatility[:train_split_idx]
    test_vol = volatility[train_split_idx:]

    train_rsi = rsi[:train_split_idx]
    test_rsi = rsi[train_split_idx:]

    train_ret = returns[:train_split_idx]
    test_ret = returns[train_split_idx:]

    # Step 3: Center and Scale (Each feature independently)
    # Volatility
    train_mean = train_vol.mean()
    scaler_vol = StandardScaler()
    train_vol_scaled = scaler_vol.fit_transform((train_vol - train_mean).reshape(-1, 1)).flatten()
    test_vol_scaled = scaler_vol.transform((test_vol - train_mean).reshape(-1, 1)).flatten()

    # RSI (Already 0-100, but scaling helps)
    scaler_rsi = StandardScaler()
    train_rsi_scaled = scaler_rsi.fit_transform(train_rsi.reshape(-1, 1)).flatten()
    test_rsi_scaled = scaler_rsi.transform(test_rsi.reshape(-1, 1)).flatten()

    # Returns
    scaler_ret = StandardScaler()
    train_ret_scaled = scaler_ret.fit_transform(train_ret.reshape(-1, 1)).flatten()
    test_ret_scaled = scaler_ret.transform(test_ret.reshape(-1, 1)).flatten()

    # Step 4: Walk-forward rolling decomposition
    # Note: Decomposition is ONLY done on Volatility (Experts specialize in Volatility structure)
    # But Raw Input will have Volatility + RSI + Returns

    decomposer = MODWTDecomposer(wavelet=wavelet, level=level)

    def generate_wf_features_enhanced(vol, rsi, ret, decomposer, lookback, buffer_size):
        num_samples = len(vol) - lookback - 1

        expert1_data = np.zeros((num_samples, lookback, 1))
        expert2_data = np.zeros((num_samples, lookback, 2))
        expert3_data = np.zeros((num_samples, lookback, 2))

        # [CHANGE] Raw data now has 3 channels
        raw_data = np.zeros((num_samples, lookback, 3))

        for t in range(lookback, lookback + num_samples):
            buffer_start = max(0, t - buffer_size)
            buffer_end = t + 1
            buffer_vol = vol[buffer_start:buffer_end]

            # Decompose Volatility only
            try:
                components = decomposer.decompose(buffer_vol, use_mra=True)
            except Exception as e:
                components = {f'cD{i + 1}': np.zeros(len(buffer_vol)) for i in range(level)}
                components[f'cA{level}_trend'] = np.zeros(len(buffer_vol))

            # Fill sequence
            seq_start = t - lookback + 1
            for seq_idx, time_pos in enumerate(range(seq_start, t + 1)):
                buffer_time_idx = time_pos - buffer_start
                if buffer_time_idx < 0 or buffer_time_idx >= len(buffer_vol): continue

                # Wavelet Features (Same as before)
                expert1_data[t - lookback, seq_idx, 0] = components[f'cA{level}_trend'][buffer_time_idx]
                expert2_data[t - lookback, seq_idx, 0] = components['cD4'][buffer_time_idx]
                expert2_data[t - lookback, seq_idx, 1] = components['cD3'][buffer_time_idx]
                expert3_data[t - lookback, seq_idx, 0] = components['cD2'][buffer_time_idx]
                expert3_data[t - lookback, seq_idx, 1] = components['cD1'][buffer_time_idx]

                # [CHANGE] Raw Features: Volatility, RSI, Returns
                raw_data[t - lookback, seq_idx, 0] = vol[time_pos]
                raw_data[t - lookback, seq_idx, 1] = rsi[time_pos]
                raw_data[t - lookback, seq_idx, 2] = ret[time_pos]

        return expert1_data, expert2_data, expert3_data, raw_data

    # Generate
    print(f"  Generating features with RSI & Returns...")
    train_e1, train_e2, train_e3, train_raw = generate_wf_features_enhanced(
        train_vol_scaled, train_rsi_scaled, train_ret_scaled, decomposer, lookback, buffer_size
    )

    test_e1, test_e2, test_e3, test_raw = generate_wf_features_enhanced(
        test_vol_scaled, test_rsi_scaled, test_ret_scaled, decomposer, lookback, buffer_size
    )

    # Targets (Still Volatility)
    train_targets = train_vol_scaled[lookback + 1:lookback + 1 + len(train_e1)].reshape(-1, 1)
    test_targets = test_vol_scaled[lookback + 1:lookback + 1 + len(test_e1)].reshape(-1, 1)

    # =========================================================
    # [SANITY CHECK MODE]
    # 這裡才是正確的打亂方式：保留 X 的結構，但隨機打亂 y。
    # 如果模型還能預測準，那就是見鬼了。
    # =========================================================
    DO_SANITY_CHECK = (run_mode == "SANITY_TARGET_SHUFFLE")

    if DO_SANITY_CHECK:
        print("\nWARNING: SANITY CHECK ENABLED - SHUFFLING TARGETS ONLY")
        np.random.shuffle(train_targets)
        np.random.shuffle(test_targets)
        print("  Targets have been shuffled independently of features.")
    # =========================================================

    # [SANITY CHECK MODE 2: INPUT DESTRUCTION]
    DO_INPUT_SHUFFLE = (run_mode == "SANITY_INPUT_SHUFFLE")

    if DO_INPUT_SHUFFLE:
        print("\nWARNING: INPUT SHUFFLE CHECK - DESTROYING FEATURES")

        # 方法 A: 隨機打亂 (保留分佈，破壞順序) -> 測試是否依賴時間結構
        # 注意：要對 train_e1, e2, e3 和 raw 全部打亂
        idx = np.random.permutation(len(train_e1))
        train_e1 = train_e1[idx]
        train_e2 = train_e2[idx]
        train_e3 = train_e3[idx]
        train_raw = train_raw[idx]

        idx_test = np.random.permutation(len(test_e1))
        test_e1 = test_e1[idx_test]
        test_e2 = test_e2[idx_test]
        test_e3 = test_e3[idx_test]
        test_raw = test_raw[idx_test]

        # Target 保持原樣！不動！
        # train_targets = train_targets 

        print("  Features have been destroyed. Targets remain real.")

    # Datasets
    train_dataset = MODWTVolatilityDataset(train_e1, train_e2, train_e3, train_targets, raw_input=train_raw)
    test_dataset = MODWTVolatilityDataset(test_e1, test_e2, test_e3, test_targets, raw_input=test_raw)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    scalers = {'target': scaler_vol, 'volatility_mean': train_mean}

    # Calculate energies (dummy for return signature)
    energies = {}

    run_mode = run_mode
    if DO_SANITY_CHECK:
        run_mode = "SANITY_TARGET_SHUFFLE"
    elif DO_INPUT_SHUFFLE:
        run_mode = "SANITY_INPUT_SHUFFLE"

    return train_loader, test_loader, scalers, train_mean, energies, run_mode


# ==================== Two-Stage Curriculum Training ====================
def train_hybrid_moe_curriculum(train_loader, test_loader, num_epochs=100,
                                lr=0.001, device=DEVICE):
    """
    Updated Training Loop for AdvancedResidualMODWTMoE (Fixed Metrics Logging)
    """

    print("[Training] Advanced Residual MODWT-MoE (Curriculum Learning)")

    model = AdvancedResidualMODWTMoE().to(device)

    # 使用新的混合損失函數
    criterion = HybridDirectionalLoss(direction_weight=0.8, penalty_scale=40.0, sharpness=10.0)

    # [FIX] 補齊所有需要的 history keys
    history = {
        'train_loss': [], 'test_loss': [],
        'train_rmse': [], 'train_mae': [],
        'epochs': [], 'alpha_mean': [],
        'base_pred_mean': [], 'moe_pred_mean': [],  # 補上這兩個，繪圖函數需要
        'branch_weight': [],  # 雖然改用動態權重，但為了相容繪圖函數保留此key (存成 alpha mean)
        'stage': []
    }

    best_test_loss = float('inf')
    best_model_state = model.state_dict().copy()
    best_epoch = 1

    stage1_epochs = num_epochs // 2
    stage2_epochs = num_epochs - stage1_epochs

    # ==================== STAGE 1: LSTM-Only Training ====================
    print(f"\n[Stage 1] Training LSTM Baseline Only (Epochs 1-{stage1_epochs})")
    model.expert1.requires_grad_(False)
    model.expert2.requires_grad_(False)
    model.expert3.requires_grad_(False)
    model.gating.requires_grad_(False)
    model.meta_gate.requires_grad_(False)

    optimizer = torch.optim.Adam(model.base_branch.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(stage1_epochs):
        model.train()
        train_losses = []

        # [FIX] 用來收集整個 epoch 的預測值以計算 RMSE/MAE
        epoch_preds = []
        epoch_targets = []
        epoch_base_preds = []

        for batch in train_loader:
            raw_input = batch['raw_input'].to(device)
            e1 = batch['expert1'].to(device)
            e2 = batch['expert2'].to(device)
            e3 = batch['expert3'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()

            # Forward (雖然有回傳 total_delta, 但 Stage 1 我們手動計算)
            output, total_delta, moe_delta, weights, expert_preds, attn, alpha = model(
                raw_input, e1, e2, e3
            )

            # Stage 1: Manual LSTM-only prediction
            prev_value = raw_input[:, -1, 0:1]

            base_delta = model.base_branch(raw_input)
            stage1_pred = prev_value + base_delta

            loss = criterion(stage1_pred, target, prev_value)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.base_branch.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

            # [FIX] 收集數據
            epoch_preds.append(stage1_pred.detach().cpu().numpy())
            epoch_targets.append(target.cpu().numpy())
            epoch_base_preds.append(stage1_pred.detach().cpu().numpy())

        # [FIX] 計算並儲存 Metrics
        avg_train_loss = np.mean(train_losses)

        # Concatenate arrays
        all_preds = np.concatenate(epoch_preds, axis=0)
        all_targets = np.concatenate(epoch_targets, axis=0)
        all_base = np.concatenate(epoch_base_preds, axis=0)

        train_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        train_mae = mean_absolute_error(all_targets, all_preds)

        # Test Loop
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in test_loader:
                raw_input = batch['raw_input'].to(device)
                target = batch['target'].to(device)
                prev_value = raw_input[:, -1, 0:1]

                base_delta = model.base_branch(raw_input)
                stage1_pred = prev_value + base_delta

                loss = criterion(stage1_pred, target, prev_value)
                test_losses.append(loss.item())

        avg_test_loss = np.mean(test_losses)
        scheduler.step(avg_test_loss)

        # [FIX] Append History
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['train_rmse'].append(train_rmse)  # 這是之前報錯的地方
        history['train_mae'].append(train_mae)
        history['epochs'].append(epoch + 1)
        history['stage'].append(1)
        history['alpha_mean'].append(0.0)
        history['branch_weight'].append(0.0)  # For compatibility

        # Stage 1: Base is everything, MoE is 0
        history['base_pred_mean'].append(np.mean(all_base))
        history['moe_pred_mean'].append(0.0)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{stage1_epochs} | Loss: {avg_train_loss:.6f} | RMSE: {train_rmse:.4f}")

    # ==================== STAGE 2: Joint Training ====================
    print(f"\n[Stage 2] Joint Training (Unfreezing All) (Epochs {stage1_epochs + 1}-{num_epochs})")

    model.expert1.requires_grad_(True)
    model.expert2.requires_grad_(True)
    model.expert3.requires_grad_(True)
    model.gating.requires_grad_(True)
    model.meta_gate.requires_grad_(True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(stage2_epochs):
        model.train()
        train_losses = []
        alpha_values = []

        # [FIX] 收集數據
        epoch_preds = []
        epoch_targets = []
        epoch_base_preds = []
        epoch_moe_preds = []

        for batch in train_loader:
            raw_input = batch['raw_input'].to(device)
            e1 = batch['expert1'].to(device)
            e2 = batch['expert2'].to(device)
            e3 = batch['expert3'].to(device)
            target = batch['target'].to(device)
            prev_value = raw_input[:, -1, 0:1]

            optimizer.zero_grad()

            # Forward
            output, total_delta, moe_delta, weights, expert_preds, attn, alpha = model(
                raw_input, e1, e2, e3
            )

            loss = criterion(output, target, prev_value)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            alpha_values.append(alpha.mean().item())

            # [FIX] 收集數據
            epoch_preds.append(output.detach().cpu().numpy())
            epoch_targets.append(target.cpu().numpy())

            # Reconstruct base part: prev_value + base_delta
            # total_delta = base_delta + alpha*moe_delta
            # base_pred = output - alpha*moe_delta
            moe_contribution = (alpha * moe_delta)
            base_prediction = output - moe_contribution

            epoch_base_preds.append(base_prediction.detach().cpu().numpy())
            epoch_moe_preds.append(moe_contribution.detach().cpu().numpy())

        # [FIX] 計算指標
        avg_train_loss = np.mean(train_losses)
        avg_alpha = np.mean(alpha_values)

        all_preds = np.concatenate(epoch_preds, axis=0)
        all_targets = np.concatenate(epoch_targets, axis=0)
        all_base = np.concatenate(epoch_base_preds, axis=0)
        all_moe = np.concatenate(epoch_moe_preds, axis=0)

        train_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        train_mae = mean_absolute_error(all_targets, all_preds)

        # Test Loop
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in test_loader:
                raw_input = batch['raw_input'].to(device)
                e1 = batch['expert1'].to(device)
                e2 = batch['expert2'].to(device)
                e3 = batch['expert3'].to(device)
                target = batch['target'].to(device)
                prev_value = raw_input[:, -1, 0:1]

                output, _, _, _, _, _, _ = model(raw_input, e1, e2, e3)
                loss = criterion(output, target, prev_value)
                test_losses.append(loss.item())

        avg_test_loss = np.mean(test_losses)
        scheduler.step(avg_test_loss)

        current_epoch = stage1_epochs + epoch + 1
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['train_rmse'].append(train_rmse)  # [FIX]
        history['train_mae'].append(train_mae)  # [FIX]
        history['epochs'].append(current_epoch)
        history['stage'].append(2)
        history['alpha_mean'].append(avg_alpha)
        history['branch_weight'].append(avg_alpha)  # For compatibility

        history['base_pred_mean'].append(np.mean(all_base))  # [FIX]
        history['moe_pred_mean'].append(np.mean(all_moe))  # [FIX]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {current_epoch:3d}/{num_epochs} | Loss: {avg_train_loss:.6f} | RMSE: {train_rmse:.4f} | Alpha: {avg_alpha:.3f}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_epoch = current_epoch
            best_model_state = model.state_dict().copy()

    print(f"\nTraining complete. Best model from epoch {best_epoch} (Test Loss: {best_test_loss:.6f})")
    model.load_state_dict(best_model_state)

    return model, history, best_epoch


def evaluate(model, data_loader, device):
    """
    Updated Evaluate function with Naive Baseline Comparison
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_base_preds = []
    all_moe_deltas = []
    all_expert_preds = []
    all_gating_weights = []
    all_alphas = []
    all_naive_preds = []  # [新增] 用來存 Naive (Prev Value)
    all_attention_weights = {'expert1': [], 'expert2': [], 'expert3': []}

    with torch.no_grad():
        for batch in data_loader:
            # 確保你的 Dataset 回傳的 raw_input 包含 prev_value
            raw_input = batch['raw_input'].to(device)
            e1 = batch['expert1'].to(device)
            e2 = batch['expert2'].to(device)
            e3 = batch['expert3'].to(device)
            target = batch['target'].to(device)

            # 1. 取得 Naive Prediction (即 t-1 時刻的值)
            # 這是我們用來比較的基準：假設明天波動率 = 今天波動率
            prev_value = raw_input[:, -1, 0:1]

            # [新增] 收集 Naive 預測
            all_naive_preds.append(prev_value.cpu().numpy())

            # 2. 模型預測 (注意：如果你還沒改模型 output 數量，這裡維持 7 個變數)
            # 如果你已經加上了 auxiliary head，記得這裡要改成接收 8 個變數
            output, total_delta, moe_delta, weights, expert_preds, attention_weights, alpha = model(
                raw_input, e1, e2, e3
            )

            # Reconstruct "LSTM Only" prediction
            base_delta = total_delta - (alpha * moe_delta)
            base_pred_val = prev_value + base_delta

            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_base_preds.append(base_pred_val.cpu().numpy())
            all_moe_deltas.append(moe_delta.cpu().numpy())
            all_expert_preds.append(expert_preds.cpu().numpy())
            all_gating_weights.append(weights.cpu().numpy())
            all_alphas.append(alpha.cpu().numpy())

            all_attention_weights['expert1'].append(attention_weights['expert1'].cpu().numpy())
            all_attention_weights['expert2'].append(attention_weights['expert2'].cpu().numpy())
            all_attention_weights['expert3'].append(attention_weights['expert3'].cpu().numpy())

    # Concatenate
    predictions = np.concatenate(all_preds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()
    base_predictions = np.concatenate(all_base_preds, axis=0).flatten()
    moe_deltas = np.concatenate(all_moe_deltas, axis=0).flatten()
    expert_preds_all = np.concatenate(all_expert_preds, axis=0)
    gating_weights = np.concatenate(all_gating_weights, axis=0)
    alphas = np.concatenate(all_alphas, axis=0).flatten()

    # [新增] Naive 數組
    naive_predictions = np.concatenate(all_naive_preds, axis=0).flatten()

    attention_weights_concat = {
        'expert1': np.concatenate(all_attention_weights['expert1'], axis=0),
        'expert2': np.concatenate(all_attention_weights['expert2'], axis=0),
        'expert3': np.concatenate(all_attention_weights['expert3'], axis=0)
    }

    # ==========================================
    # Metrics & Naive Comparison
    # ==========================================

    # 1. Model Metrics
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    # Model Direction Accuracy
    # 實際變化方向 (t -> t+1)
    diff_true = targets - naive_predictions
    # 預測變化方向 (t -> t+1)
    diff_pred = predictions - naive_predictions

    # 排除變化量極小的雜訊 (Optional, 這裡先用標準計算)
    direction_acc = np.mean(np.sign(diff_true) == np.sign(diff_pred))

    # 2. Naive Metrics (Baseline)
    naive_rmse = np.sqrt(mean_squared_error(targets, naive_predictions))
    naive_mae = mean_absolute_error(targets, naive_predictions)

    # 3. Naive Direction Accuracy (Lag Test)
    # 測試邏輯：如果我們單純把實際曲線往後移一格 (Lag 1)，它的方向準確率是多少？
    # 這是檢測 "Auto-Correlation" (自相關)
    # 我們比較 [t vs t+1] 的真實變化 和 [t-1 vs t] 的真實變化
    # 注意：這裡我們要對齊時間軸，所以會少一個樣本

    true_dir_current = np.sign(targets[1:] - targets[:-1])
    naive_dir_lagged = np.sign(naive_predictions[1:] - naive_predictions[:-1])

    naive_direction_acc = np.mean(true_dir_current == naive_dir_lagged)

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_acc': direction_acc,
        'naive_rmse': naive_rmse,  # [新增]
        'naive_mae': naive_mae,  # [新增]
        'naive_direction_acc': naive_direction_acc  # [新增]
    }

    return (metrics, predictions, targets, base_predictions, moe_deltas,
            expert_preds_all, gating_weights, attention_weights_concat, alphas)


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
        if stages[i] != stages[i - 1]:
            stage_transition = epochs[i - 1]
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


def plot_gating_weights(gating_weights, test_targets_original, save_path='gating_weights.png'):
    """
    根據波動率regime繪製gating weights（如參考圖片所示）
    
    改進點：
    - 左圖：堆疊長條圖顯示各regime下的expert權重分布
    - 右圖：折線圖展示expert權重隨regime變化的動態
    - 自動計算低/中/高波動率閾值（33%和67%分位數）
    - 在長條圖內顯示百分比標籤
    
    Args:
        gating_weights: numpy array of shape (num_samples, 3)
                       [Expert 1 Weight, Expert 2 Weight, Expert 3 Weight]
        test_targets_original: 實際波動率值（用於regime分類）
        save_path: 圖片儲存路徑
    """

    if isinstance(gating_weights, torch.Tensor):
        gating_weights = gating_weights.cpu().detach().numpy()

    # 根據百分位數定義波動率regimes
    vol_33 = np.percentile(test_targets_original, 33)
    vol_67 = np.percentile(test_targets_original, 67)

    # 分類到不同regimes
    low_mask = test_targets_original <= vol_33
    med_mask = (test_targets_original > vol_33) & (test_targets_original <= vol_67)
    high_mask = test_targets_original > vol_67

    # 計算各regime的平均權重
    regimes = ['Low\nVolatility', 'Medium\nVolatility', 'High\nVolatility']
    expert1_means = [
        gating_weights[low_mask, 0].mean(),
        gating_weights[med_mask, 0].mean(),
        gating_weights[high_mask, 0].mean()
    ]
    expert2_means = [
        gating_weights[low_mask, 1].mean(),
        gating_weights[med_mask, 1].mean(),
        gating_weights[high_mask, 1].mean()
    ]
    expert3_means = [
        gating_weights[low_mask, 2].mean(),
        gating_weights[med_mask, 2].mean(),
        gating_weights[high_mask, 2].mean()
    ]

    # 創建2個子圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # === 左圖：堆疊長條圖 ===
    x_pos = np.arange(len(regimes))
    width = 0.6

    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']  # 綠、藍、橘

    ax1.bar(x_pos, expert1_means, width, label='Expert 1 (Trend)',
            color=colors[0], alpha=0.85, edgecolor='white', linewidth=1.5)
    ax1.bar(x_pos, expert2_means, width, bottom=expert1_means,
            label='Expert 2 (Cyclic)', color=colors[1], alpha=0.85,
            edgecolor='white', linewidth=1.5)

    bottoms = np.array(expert1_means) + np.array(expert2_means)
    ax1.bar(x_pos, expert3_means, width, bottom=bottoms,
            label='Expert 3 (High-Freq)', color=colors[2], alpha=0.85,
            edgecolor='white', linewidth=1.5)

    # 在長條圖內添加百分比標籤
    for i in range(len(regimes)):
        # Expert 1
        if expert1_means[i] > 0.08:
            ax1.text(i, expert1_means[i] / 2, f'{expert1_means[i] * 100:.1f}%',
                     ha='center', va='center', fontweight='bold',
                     fontsize=11, color='white')

        # Expert 2
        if expert2_means[i] > 0.08:
            ax1.text(i, expert1_means[i] + expert2_means[i] / 2,
                     f'{expert2_means[i] * 100:.1f}%',
                     ha='center', va='center', fontweight='bold',
                     fontsize=11, color='white')

        # Expert 3
        if expert3_means[i] > 0.08:
            ax1.text(i, expert1_means[i] + expert2_means[i] + expert3_means[i] / 2,
                     f'{expert3_means[i] * 100:.1f}%',
                     ha='center', va='center', fontweight='bold',
                     fontsize=11, color='white')

    ax1.set_ylabel('Gating Weight', fontsize=13, fontweight='bold')
    ax1.set_title('Gating Weights by Volatility Regime (Stacked)',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(regimes, fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # === 右圖：折線圖展示動態變化 ===
    x_pos_line = np.arange(len(regimes))

    ax2.plot(x_pos_line, expert1_means, marker='o', markersize=12,
             linewidth=3, label='Expert 1 (Trend)', color=colors[0], alpha=0.85)
    ax2.plot(x_pos_line, expert2_means, marker='s', markersize=12,
             linewidth=3, label='Expert 2 (Cyclic)', color=colors[1], alpha=0.85)
    ax2.plot(x_pos_line, expert3_means, marker='^', markersize=12,
             linewidth=3, label='Expert 3 (High-Freq)', color=colors[2], alpha=0.85)

    ax2.set_ylabel('Gating Weight', fontsize=13, fontweight='bold')
    ax2.set_title('Gating Weight Dynamics Across Regimes',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos_line)
    ax2.set_xticklabels(regimes, fontsize=12)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc='best', frameon=True, framealpha=0.95, fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Gating weights plot saved to {save_path}")
    print(f"  Regime thresholds: Low ≤ {vol_33:.2f}% < Medium ≤ {vol_67:.2f}% < High")


def plot_attention_maps(attention_weights, save_path='attention_maps.png'):
    """
    視覺化時間注意力機制 - 更清晰的版本
    
    改進點：
    - 添加標準差陰影區域（confidence interval）
    - 用紅色虛線標記"最近時間點"
    - 添加marker讓趨勢更明顯
    - 統一配色方案（與gating weights一致）
    - 更清楚的軸標籤和說明
    
    Args:
        attention_weights: Dict with keys 'expert1', 'expert2', 'expert3'
                          每個都是 [num_samples, seq_len] array
        save_path: 圖片儲存路徑
    """

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 提取attention weights
    attn1 = attention_weights['expert1']  # [num_samples, 30]
    attn2 = attention_weights['expert2']
    attn3 = attention_weights['expert3']

    # 計算所有樣本的平均attention
    mean_attn1 = attn1.mean(axis=0)
    mean_attn2 = attn2.mean(axis=0)
    mean_attn3 = attn3.mean(axis=0)

    # 計算標準差（用於confidence intervals）
    std_attn1 = attn1.std(axis=0)
    std_attn2 = attn2.std(axis=0)
    std_attn3 = attn3.std(axis=0)

    timesteps = np.arange(len(mean_attn1))

    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']  # 與expert顏色一致

    # ==================== 子圖1: Expert 1 (Trend) ====================
    axes[0, 0].plot(timesteps, mean_attn1, color=colors[0], linewidth=2.5,
                    label='Mean Attention', marker='o', markersize=4, markevery=5)
    axes[0, 0].fill_between(timesteps, mean_attn1 - std_attn1, mean_attn1 + std_attn1,
                            color=colors[0], alpha=0.25, label='±1 Std Dev')
    axes[0, 0].axvline(len(mean_attn1) - 1, color='red', linestyle='--',
                       linewidth=1.5, alpha=0.5, label='Most Recent')
    axes[0, 0].set_xlabel('Lookback Days (0 = t-30, 29 = t-1)', fontsize=11)
    axes[0, 0].set_ylabel('Attention Weight', fontsize=11)
    axes[0, 0].set_title('Expert 1 (Trend) - Temporal Attention',
                         fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3, linestyle='--')
    axes[0, 0].legend(fontsize=10, loc='best')
    axes[0, 0].set_xlim([-0.5, len(mean_attn1) - 0.5])

    # ==================== 子圖2: Expert 2 (Cyclic) ====================
    axes[0, 1].plot(timesteps, mean_attn2, color=colors[1], linewidth=2.5,
                    label='Mean Attention', marker='s', markersize=4, markevery=5)
    axes[0, 1].fill_between(timesteps, mean_attn2 - std_attn2, mean_attn2 + std_attn2,
                            color=colors[1], alpha=0.25, label='±1 Std Dev')
    axes[0, 1].axvline(len(mean_attn2) - 1, color='red', linestyle='--',
                       linewidth=1.5, alpha=0.5, label='Most Recent')
    axes[0, 1].set_xlabel('Lookback Days (0 = t-30, 29 = t-1)', fontsize=11)
    axes[0, 1].set_ylabel('Attention Weight', fontsize=11)
    axes[0, 1].set_title('Expert 2 (Cyclic) - Temporal Attention',
                         fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, linestyle='--')
    axes[0, 1].legend(fontsize=10, loc='best')
    axes[0, 1].set_xlim([-0.5, len(mean_attn2) - 0.5])

    # ==================== 子圖3: Expert 3 (High-Freq) ====================
    axes[1, 0].plot(timesteps, mean_attn3, color=colors[2], linewidth=2.5,
                    label='Mean Attention', marker='^', markersize=4, markevery=5)
    axes[1, 0].fill_between(timesteps, mean_attn3 - std_attn3, mean_attn3 + std_attn3,
                            color=colors[2], alpha=0.25, label='±1 Std Dev')
    axes[1, 0].axvline(len(mean_attn3) - 1, color='red', linestyle='--',
                       linewidth=1.5, alpha=0.5, label='Most Recent')
    axes[1, 0].set_xlabel('Lookback Days (0 = t-30, 29 = t-1)', fontsize=11)
    axes[1, 0].set_ylabel('Attention Weight', fontsize=11)
    axes[1, 0].set_title('Expert 3 (High-Freq) - Temporal Attention',
                         fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, linestyle='--')
    axes[1, 0].legend(fontsize=10, loc='best')
    axes[1, 0].set_xlim([-0.5, len(mean_attn3) - 0.5])

    # ==================== 子圖4: 所有Expert的綜合比較 ====================
    axes[1, 1].plot(timesteps, mean_attn1, color=colors[0], linewidth=2.5,
                    label='Expert 1 (Trend)', alpha=0.85, marker='o', markersize=3, markevery=5)
    axes[1, 1].plot(timesteps, mean_attn2, color=colors[1], linewidth=2.5,
                    label='Expert 2 (Cyclic)', alpha=0.85, marker='s', markersize=3, markevery=5)
    axes[1, 1].plot(timesteps, mean_attn3, color=colors[2], linewidth=2.5,
                    label='Expert 3 (High-Freq)', alpha=0.85, marker='^', markersize=3, markevery=5)
    axes[1, 1].axvline(len(mean_attn1) - 1, color='red', linestyle='--',
                       linewidth=1.5, alpha=0.5, label='Most Recent')
    axes[1, 1].set_xlabel('Lookback Days (0 = t-30, 29 = t-1)', fontsize=11)
    axes[1, 1].set_ylabel('Attention Weight', fontsize=11)
    axes[1, 1].set_title('All Experts - Temporal Focus Comparison',
                         fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, linestyle='--')
    axes[1, 1].legend(fontsize=10, loc='best')
    axes[1, 1].set_xlim([-0.5, len(mean_attn1) - 0.5])

    # 添加總標題
    fig.suptitle('Temporal Attention Mechanisms: Expert Focus Across Lookback Window',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Attention maps saved to {save_path}")


def plot_alpha_distribution(alphas, save_path='../results/alpha_dist.png'):
    """
    繪製 Meta-Gate Alpha 值的直方圖分佈
    用來觀察模型是傾向於依賴 Expert (Alpha -> 1) 還是 Base (Alpha -> 0)
    或者是動態調整 (分佈廣)
    """
    # 確保 alphas 是 numpy array
    if isinstance(alphas, torch.Tensor):
        alphas = alphas.cpu().detach().numpy()

    plt.figure(figsize=(10, 6))

    # 繪製直方圖
    n, bins, patches = plt.hist(alphas, bins=50, color='#9467bd', alpha=0.7,
                                edgecolor='black', linewidth=0.8, label='Alpha Frequency')

    # 繪製平均線
    mean_val = alphas.mean()
    plt.axvline(mean_val, color='#d62728', linestyle='--', linewidth=2,
                label=f'Mean: {mean_val:.4f}')

    # 裝飾圖表
    plt.title('Distribution of Meta-Gate Alpha Values\n(0 = LSTM Base Only, 1 = Full Expert Residual)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Alpha Value', fontsize=12)
    plt.ylabel('Frequency (Number of Samples)', fontsize=12)
    plt.legend(loc='upper right', frameon=True, fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.xlim(0, 1)  # Alpha 範圍被 Sigmoid 限制在 0-1 之間

    # 加入文字註解
    plt.text(0.02, plt.ylim()[1] * 0.95, 'Mostly Base ←', fontsize=10, color='gray', ha='left')
    plt.text(0.98, plt.ylim()[1] * 0.95, '→ Mostly Expert', fontsize=10, color='gray', ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Alpha distribution plot saved to {save_path}")


# ==================== Main ====================
if __name__ == "__main__":
    os.makedirs('../results', exist_ok=True)

    RUN_MODE = "NORMAL"  # 可選：NORMAL, SANITY_INPUT_SHUFFLE, SANITY_TARGET_SHUFFLE

    print("\n" + "=" * 80)
    print("[MODWT-MoE Volatility Forecasting - Production Model]")
    print("=" * 80)

    # 1. Load data
    print("\n[Loading Data]")
    df = pd.read_csv("../dataset/USD_TWD.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  Loaded {len(df)} days")

    # 2. Prepare data
    print("\n[Data Preparation]")
    train_loader, test_loader, scalers, volatility_mean, energies, run_mode = prepare_modwt_data(
        df,
        vol_window=7,
        lookback=30,
        forecast_horizon=1,
        wavelet='db4',
        level=4,
        train_ratio=0.80,
        batch_size=32,
        use_robust_scaler=False,
        buffer_size=200,
        run_mode=RUN_MODE
    )

    # 3. Train
    print("\n[Training]")
    trained_model, training_history, best_epoch = train_hybrid_moe_curriculum(
        train_loader,
        test_loader,
        num_epochs=100,
        lr=0.001,
        device=DEVICE
    )

    # =========================================================================
    # [Evaluation & Smart Reporting Logic]
    # =========================================================================

    # 4. Get Raw Predictions
    (test_metrics, test_preds, test_targets, test_base_preds, test_moe_deltas,
     test_expert_preds, test_gating_weights, test_attention_weights, test_alphas) = evaluate(
        trained_model, test_loader, DEVICE
    )

    # 5. Inverse Transform (還原回真實波動率數值)
    target_scaler = scalers['target']

    test_preds_centered = target_scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
    test_targets_centered = target_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()
    test_base_preds_centered = target_scaler.inverse_transform(test_base_preds.reshape(-1, 1)).flatten()

    std_dev = np.sqrt(target_scaler.var_[0])
    test_moe_deltas_original = test_moe_deltas * std_dev

    test_preds_original = test_preds_centered + volatility_mean
    test_targets_original = test_targets_centered + volatility_mean
    test_base_preds_original = test_base_preds_centered + volatility_mean

    # 6. Calculate Real-World Metrics
    # R2 & RMSE
    r2_original = r2_score(test_targets_original, test_preds_original)
    rmse_original = np.sqrt(mean_squared_error(test_targets_original, test_preds_original))
    mae_original = mean_absolute_error(test_targets_original, test_preds_original)

    # LSTM Base Metrics
    rmse_base = np.sqrt(mean_squared_error(test_targets_original, test_base_preds_original))

    # True Direction Accuracy (基於還原後的真實走勢)
    # 這是最公正的指標，不受 Shuffle 影響定義
    dir_true = np.sign(np.diff(test_targets_original))
    dir_pred = np.sign(np.diff(test_preds_original))
    true_direction_acc = np.mean(dir_true == dir_pred)

    # =========================================================================
    # [CONDITIONAL PRINTING] 根據模式決定顯示什麼報告
    # =========================================================================
    print("\n" + "=" * 80)

    if run_mode == "SANITY_INPUT_SHUFFLE":
        print(f"SANITY CHECK REPORT: INPUT DESTRUCTION MODE (X-Shuffle)")
        print("=" * 80)
        print("   TEST GOAL: Verify model fails when input features are destroyed.")
        print("   (We want to prove the model isn't just memorizing the output trend.)")
        print("-" * 60)

        # 判斷 R2 (應該要是負的)
        print(f"   1. R² Score:            {r2_original:.4f}  ", end="")

        # 判斷 Direction (應該要在 50% 附近)
        print(f"   2. Direction Accuracy:  {true_direction_acc * 100:.2f}%   \n", end="")

        print("-" * 60)
        print("   NOTE: Ignore internal metrics inside evaluate().")
        print("   The metrics above are calculated against the REAL target trend.")

    elif run_mode == "SANITY_TARGET_SHUFFLE":
        print(f"SANITY CHECK REPORT: TARGET SHUFFLE MODE (Y-Shuffle)")
        print("=" * 80)
        print("   TEST GOAL: Verify model fails when answers are randomized.")
        print("   (We want to prove there is no label leakage.)")
        print("-" * 60)

        print(f"   R² Score: {r2_original:.4f} \n", end="")

    else:
        # NORMAL MODE (正常訓練時才顯示這份詳細報告)
        print(f"[PRODUCTION RESULTS] Advanced Residual MODWT-MoE")
        print("=" * 80)

        print(f"\n1. Overall Performance (Hybrid):")
        print(f"   RMSE:               {rmse_original:.4f}%")
        print(f"   MAE:                {mae_original:.4f}%")
        print(f"   R²:                 {r2_original:.4f}")
        print(f"   Direction Accuracy: {true_direction_acc * 100:.2f}%")

        print(f"\n2. Ablation Analysis (Contribution):")
        print(f"   LSTM Base RMSE:     {rmse_base:.4f}%")
        print(f"   Hybrid Improv.:     {(rmse_base - rmse_original):.4f}%")

        print(f"\n3. Meta-Gate Statistics:")
        print(f"   Mean Alpha:         {test_alphas.mean():.4f}")
        print(f"   (High Alpha = Experts Active, Low Alpha = Base Active)")

        # 只有在正常模式下才畫圖
        print("\n[Visualizations]")
        plot_training_curves(training_history, '../results/training_history.png')
        plot_predictions(test_targets_original, test_preds_original, test_base_preds_original,
                         '../results/predictions.png')
        plot_gating_weights(test_gating_weights, test_targets_original, save_path='../results/gating_weights.png')
        plot_attention_maps(test_attention_weights, save_path='../results/attention_maps.png')
        plot_alpha_distribution(test_alphas, save_path='../results/alpha_dist.png')

    print("=" * 80 + "\n")
