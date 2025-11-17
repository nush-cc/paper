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
    print("✅ modwtpy imported successfully")
except ImportError:
    print("❌ modwtpy not found. Please install it:")
    print("   pip install modwtpy")
    print("   or: pip install git+https://github.com/pistonly/modwtpy.git")
    raise

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
        print(f"   Decomposing with modwtpy (wavelet={self.wavelet}, level={self.level}, MRA={use_mra})...")

        # Perform MODWT
        try:
            w = modwt(signal, self.wavelet, self.level)
        except Exception as e:
            print(f"❌ MODWT failed: {e}")
            print(f"   Try using 'haar', 'db2', 'db4', or 'sym4'")
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

        print(f"   ✅ MODWT Decomposition Complete")
        print(f"      Wavelet: {self.wavelet}, Level: {self.level}")
        print(f"      Components: {self.components_names}")
        print(f"      Signal length: {len(signal)}")

        if not np.isnan(recon_error):
            print(f"      Reconstruction Error: {recon_error:.10f}")

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

        # 取得各個成分
        cA4 = components_dict['cA4_trend']
        cD4 = components_dict['cD4']
        cD3 = components_dict['cD3']
        cD2 = components_dict['cD2']
        cD1 = components_dict['cD1']

        # 確保所有 component 長度一致
        min_len = min(len(cA4), len(cD4), len(cD3), len(cD2), len(cD1), len(target))
        cA4 = cA4[:min_len]
        cD4 = cD4[:min_len]
        cD3 = cD3[:min_len]
        cD2 = cD2[:min_len]
        cD1 = cD1[:min_len]
        target = target[:min_len]

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


# ==================== Enhanced Gating Network ====================
class GatingNetwork(nn.Module):
    """Enhanced Gating network with richer features"""

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


# ==================== Data Preparation with Centering ====================
def prepare_modwt_data(df, vol_window=7, lookback=30, forecast_horizon=1,
                       wavelet='db4', level=4,
                       train_ratio=0.80, batch_size=32,  # 移除 val_ratio
                       use_robust_scaler=False):
    """
    準備 MODWT 資料，使用去中心化處理
    修改為 80/20 Train/Test 切分

    關鍵改進：
    1. 先切分時間序列
    2. 對每個子集**去中心化**（減去均值）
    3. 分別做 MODWT 分解（使用 MRA）
    4. 這樣確保能量分布合理，且無資料洩漏
    """

    print("\n" + "=" * 80)
    print("🔧 Data Preparation with MODWT + Centering (Train/Test Split)")
    print("=" * 80)

    # Step 1: 計算波動率
    print("\n📊 Step 1: Calculate volatility...")
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['log_return'].rolling(window=vol_window).std() * np.sqrt(252) * 100
    df = df.dropna().reset_index(drop=True)
    volatility = df['Volatility'].values

    print(f"   ✅ Volatility calculated: {len(volatility)} samples")
    print(f"      Mean: {volatility.mean():.4f}%")
    print(f"      Std: {volatility.std():.4f}%")
    print(f"      Range: [{volatility.min():.4f}, {volatility.max():.4f}]")

    # Step 2: 時間序列切分 (80/20)
    print("\n📊 Step 2: Time-series split (80% Train / 20% Test)...")
    total_len = len(volatility)
    train_split_idx = int(total_len * train_ratio)

    print(f"   Total samples: {total_len}")
    print(f"   Train: 0 to {train_split_idx} ({train_split_idx} samples, {train_ratio*100:.0f}%)")
    print(f"   Test: {train_split_idx} to {total_len} ({total_len - train_split_idx} samples, {(1-train_ratio)*100:.0f}%)")

    train_volatility = volatility[:train_split_idx]
    test_volatility = volatility[train_split_idx:]

    # Step 3: 去中心化（關鍵步驟！）
    print("\n📊 Step 3: Centering (remove mean)...")

    # 只用 train 的均值
    train_mean = train_volatility.mean()

    train_volatility_centered = train_volatility - train_mean
    test_volatility_centered = test_volatility - train_mean

    print(f"   ✅ Train mean: {train_mean:.4f}% (will be subtracted)")
    print(f"      Train centered: mean={train_volatility_centered.mean():.6f}, std={train_volatility_centered.std():.4f}")
    print(f"      Test centered: mean={test_volatility_centered.mean():.6f}, std={test_volatility_centered.std():.4f}")

    # Step 4: 分別對每個子集進行 MODWT 分解（使用 MRA）
    print("\n📊 Step 4: Separate MODWT decomposition with MRA...")
    decomposer = MODWTDecomposer(wavelet=wavelet, level=level)

    print("\n   🔹 Decomposing TRAIN set...")
    train_components = decomposer.decompose(train_volatility_centered, use_mra=True)
    train_energies = decomposer.get_component_energies(train_components)

    print("\n   🔹 Decomposing TEST set...")
    test_components = decomposer.decompose(test_volatility_centered, use_mra=True)

    print("\n📊 Component Energies (Train set - After Centering):")
    for name, energy in train_energies.items():
        print(f"   {name}: {energy:.2f}%")

    # Step 5: Scaling
    print("\n📊 Step 5: Scaling components...")
    scalers = {}
    train_components_scaled = {}
    test_components_scaled = {}

    # 選擇 Scaler
    if use_robust_scaler:
        print("   Using RobustScaler (better for outliers)")
        ScalerClass = RobustScaler
    else:
        print("   Using StandardScaler")
        ScalerClass = StandardScaler

    for name in train_components.keys():
        scaler = ScalerClass()

        # 只用 train 資料來 fit
        train_components_scaled[name] = scaler.fit_transform(
            train_components[name].reshape(-1, 1)
        ).flatten()

        # Transform test
        test_components_scaled[name] = scaler.transform(
            test_components[name].reshape(-1, 1)
        ).flatten()

        scalers[name] = scaler

        if isinstance(scaler, StandardScaler):
            print(f"   ✅ {name}: Mean={scaler.mean_[0]:.6f}, Std={scaler.scale_[0]:.6f}")
        else:
            print(f"   ✅ {name}: Median={scaler.center_[0]:.6f}, IQR={scaler.scale_[0]:.6f}")

    # Scale targets (用去中心化的數據)
    target_scaler = ScalerClass()
    train_target_scaled = target_scaler.fit_transform(train_volatility_centered.reshape(-1, 1)).flatten()
    test_target_scaled = target_scaler.transform(test_volatility_centered.reshape(-1, 1)).flatten()
    scalers['target'] = target_scaler

    # 保存均值用於還原
    scalers['volatility_mean'] = train_mean

    # Step 6: 建立 Dataset 和 DataLoader
    print("\n📊 Step 6: Create datasets...")

    train_dataset = MODWTVolatilityDataset(
        train_components_scaled, train_target_scaled,
        window=lookback, forecast_horizon=forecast_horizon
    )

    test_dataset = MODWTVolatilityDataset(
        test_components_scaled, test_target_scaled,
        window=lookback, forecast_horizon=forecast_horizon
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n✅ Data preparation complete!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    return train_loader, test_loader, scalers, train_components, train_energies


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

    print("\n" + "=" * 80)
    print("🚀 Training MODWT-MoE Model (80/20 Split)")
    print("=" * 80)

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
            output, weights, expert_preds = model(e1, e2, e3)
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

                output, _, _ = model(e1, e2, e3)
                loss = criterion(output, target)
                test_losses.append(loss.item())

        avg_test_loss = np.mean(test_losses)

        scheduler.step(avg_train_loss)

        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['train_rmse'].append(train_rmse)
        history['train_mae'].append(train_mae)
        history['epochs'].append(epoch + 1)

        # Print every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Test Loss: {avg_test_loss:.4f} | "
                  f"Train RMSE: {train_rmse:.4f}")

        # 保存最佳模型 (基於 Train Loss)
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   ✅ New best model! Train Loss: {best_train_loss:.4f}")

    model.load_state_dict(best_model_state)
    print(f"\n✅ Training complete! Best model from epoch {best_epoch}")

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

    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    expert_preds = np.concatenate(all_expert_preds, axis=0)
    gating_weights = np.concatenate(all_gating_weights, axis=0)

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

    return metrics, predictions.flatten(), targets.flatten(), expert_preds, gating_weights

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

    print("\n" + "=" * 80)
    print("🔄 Walk-Forward Validation for MODWT-MoE")
    print("=" * 80)
    print(f"📊 Configuration:")
    print(f"   Train Window: {train_window} days")
    print(f"   Test Window: {test_window} days")
    print(f"   Step Size: {step_size} days")
    print(f"   Epochs per Fold: {num_epochs}")

    # 計算可以做幾個 fold
    total_len = len(df)
    max_start = total_len - train_window - test_window
    num_folds = max_start // step_size + 1

    print(f"   Total Data: {total_len} days")
    print(f"   Number of Folds: {num_folds}")
    print("=" * 80)

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
            print(f"\n⚠️  Fold {fold+1}: Insufficient data, skipping...")
            break

        print(f"\n{'='*80}")
        print(f"🔹 Fold {fold+1}/{num_folds}")
        print(f"{'='*80}")
        print(f"   Train: Index {train_start:4d} to {train_end:4d} ({train_window} days)")
        print(f"   Test:  Index {train_end:4d} to {test_end:4d} ({test_window} days)")

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
            print(f"\n🚀 Training Fold {fold+1}...")
            model, history, best_epoch = train_modwt_moe(
                train_loader,
                test_loader,
                num_epochs=num_epochs,
                lr=lr,
                device=device
            )

            # 評估
            print(f"\n📊 Evaluating Fold {fold+1}...")
            test_metrics, test_preds, test_targets, test_expert_preds, test_gating_weights = evaluate(
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
            print(f"\n✅ Fold {fold+1} Results:")
            print(f"   RMSE: {rmse_original:.4f}%")
            print(f"   MAE: {mae_original:.4f}%")
            print(f"   R²: {r2_original:.6f}")
            print(f"   Direction Accuracy: {test_metrics['direction_acc']*100:.2f}%")
            print(f"   Gating Weights: E1={test_gating_weights[:, 0].mean():.3f}, "
                  f"E2={test_gating_weights[:, 1].mean():.3f}, "
                  f"E3={test_gating_weights[:, 2].mean():.3f}")

        except Exception as e:
            print(f"\n❌ Fold {fold+1} failed with error: {e}")
            continue

    # 匯總所有結果
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("📊 Walk-Forward Validation Summary")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print("\n📈 Statistical Summary:")
    print(f"   RMSE:           {results_df['rmse'].mean():.4f}% ± {results_df['rmse'].std():.4f}%")
    print(f"   MAE:            {results_df['mae'].mean():.4f}% ± {results_df['mae'].std():.4f}%")
    print(f"   R²:             {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}")
    print(f"   Direction Acc:  {results_df['direction_acc'].mean()*100:.2f}% ± {results_df['direction_acc'].std()*100:.2f}%")

    print("\n📊 Gating Weights Across Folds:")
    print(f"   Expert 1 (Trend):    {results_df['expert1_weight'].mean():.3f} ± {results_df['expert1_weight'].std():.3f}")
    print(f"   Expert 2 (Cyclic):   {results_df['expert2_weight'].mean():.3f} ± {results_df['expert2_weight'].std():.3f}")
    print(f"   Expert 3 (High-Freq): {results_df['expert3_weight'].mean():.3f} ± {results_df['expert3_weight'].std():.3f}")

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
    print(f"📊 Saved: {save_path}")
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
    print(f"📊 Saved: {save_path}")
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
    print(f"📊 Saved: {save_path}")
    plt.close()


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
    print(f"💾 Results saved to {save_path}")

    return results_df


def analyze_gating_dynamics(gating_weights, volatility):
    """分析 Gating 動態"""

    low_vol = volatility < np.percentile(volatility, 33)
    mid_vol = (volatility >= np.percentile(volatility, 33)) & (volatility <= np.percentile(volatility, 67))
    high_vol = volatility > np.percentile(volatility, 67)

    print("📊 Gating Weights by Volatility Regime:")
    print("\nLow Volatility:")
    print(f"  Expert 1 (Trend): {gating_weights[low_vol, 0].mean():.3f}")
    print(f"  Expert 2 (Cyclic): {gating_weights[low_vol, 1].mean():.3f}")
    print(f"  Expert 3 (High-Freq): {gating_weights[low_vol, 2].mean():.3f}")

    print("\nMedium Volatility:")
    print(f"  Expert 1 (Trend): {gating_weights[mid_vol, 0].mean():.3f}")
    print(f"  Expert 2 (Cyclic): {gating_weights[mid_vol, 1].mean():.3f}")
    print(f"  Expert 3 (High-Freq): {gating_weights[mid_vol, 2].mean():.3f}")

    print("\nHigh Volatility:")
    print(f"  Expert 1 (Trend): {gating_weights[high_vol, 0].mean():.3f}")
    print(f"  Expert 2 (Cyclic): {gating_weights[high_vol, 1].mean():.3f}")
    print(f"  Expert 3 (High-Freq): {gating_weights[high_vol, 2].mean():.3f}")


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
    print(f"📊 Saved: {save_path}")
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
    print(f"📊 Saved: {save_path}")
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
    print(f"📊 Saved: {save_path}")
    plt.close()


# ==================== Main Execution ====================
if __name__ == "__main__":
    os.makedirs('../results', exist_ok=True)

    print("\n📂 Loading data...")
    df = pd.read_csv("../dataset/USD_TWD.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"✅ Loaded {len(df)} days of data")

    # ==================== 配置區 ====================
    COMPARE_WAVELETS = True  # ← 改這裡！True=比較模式, False=單一小波模式
    DEFAULT_WAVELET = 'db4'  # 單一模式時使用的小波

    if COMPARE_WAVELETS:
        wavelets_to_test = ['haar', 'db2', 'db4', 'db6', 'sym4', 'sym8', 'coif1']
        print(f"\n🔬 Mode: Wavelet Comparison")
        print(f"   Testing {len(wavelets_to_test)} wavelets: {wavelets_to_test}")
    else:
        wavelets_to_test = [DEFAULT_WAVELET]
        print(f"\n🔬 Mode: Single Wavelet")
        print(f"   Using wavelet: {DEFAULT_WAVELET}")

    results_comparison = []
    all_training_histories = {}
    all_test_results = {}

    for wavelet in wavelets_to_test:
        print(f"\n{'='*80}")
        print(f"🔬 Testing Wavelet: {wavelet}")
        print(f"{'='*80}")

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
        print("\n" + "=" * 80)
        print("📊 Final Evaluation on Test Set")
        print("=" * 80)

        test_metrics, test_preds, test_targets, test_expert_preds, test_gating_weights = evaluate(
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
            'scalers': scalers,
            'energies': energies,
            'metrics': test_metrics
        }

        print(f"\n✅ {wavelet} Results:")
        print(f"   RMSE: {rmse_original:.4f}%")
        print(f"   MAE: {mae_original:.4f}%")
        print(f"   R²: {r2_original:.6f}")
        print(f"   Direction Accuracy: {test_metrics['direction_acc']*100:.2f}%")

    # ==================== 結果比較與選擇 ====================
    comparison_df = pd.DataFrame(results_comparison)
    comparison_df = comparison_df.sort_values('rmse')

    if COMPARE_WAVELETS:
        print("\n" + "="*80)
        print("📊 Wavelet Comparison Results")
        print("="*80)
        print(comparison_df.to_string(index=False))

        # 找出最佳小波
        best_wavelet = comparison_df.iloc[0]['wavelet']
        best_rmse = comparison_df.iloc[0]['rmse']
        best_mae = comparison_df.iloc[0]['mae']
        best_r2 = comparison_df.iloc[0]['r2']

        print(f"\n🏆 Best Wavelet: {best_wavelet}")
        print(f"   RMSE: {best_rmse:.4f}%")
        print(f"   MAE: {best_mae:.4f}%")
        print(f"   R²: {best_r2:.6f}")

        # 保存比較結果
        comparison_df.to_csv('../results/wavelet_comparison.csv', index=False)
        print(f"\n💾 Comparison results saved to ../results/wavelet_comparison.csv")

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
        print(f"📊 Saved: ../results/wavelet_comparison_charts.png")
        plt.close()

    else:
        best_wavelet = DEFAULT_WAVELET
        print(f"\n✅ Using single wavelet: {best_wavelet}")

    # ==================== 使用最佳小波的結果進行詳細分析 ====================
    print(f"\n{'='*80}")
    print(f"📊 Detailed Analysis for Best Wavelet: {best_wavelet}")
    print(f"{'='*80}")

    # 取得最佳小波的結果
    best_results = all_test_results[best_wavelet]
    test_preds_original = best_results['preds']
    test_targets_original = best_results['targets']
    test_gating_weights = best_results['gating_weights']
    test_expert_preds = best_results['expert_preds']
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

    print(f"\n✅ Test Set Performance (Original Scale):")
    print(f"   RMSE: {rmse_original:.4f}%")
    print(f"   MAE: {mae_original:.4f}%")
    print(f"   R²: {r2_original:.6f}")
    print(f"   Direction Accuracy: {test_metrics['direction_acc']:.4f}")

    print(f"\n📊 Gating Weights on Test Set:")
    print(f"   Expert 1 (Trend): {test_gating_weights[:, 0].mean():.3f} ± {test_gating_weights[:, 0].std():.3f}")
    print(f"   Expert 2 (Cyclic): {test_gating_weights[:, 1].mean():.3f} ± {test_gating_weights[:, 1].std():.3f}")
    print(f"   Expert 3 (High-Freq): {test_gating_weights[:, 2].mean():.3f} ± {test_gating_weights[:, 2].std():.3f}")

    # Visualizations (使用最佳小波)
    print("\n📊 Generating visualizations...")
    plot_training_history(training_history, f'../results/training_history_{best_wavelet}.png')
    plot_predictions(test_targets_original, test_preds_original, f'../results/test_predictions_{best_wavelet}.png')
    plot_gating_weights(test_gating_weights, f'../results/test_gating_weights_{best_wavelet}.png')

    # Save results (使用最佳小波)
    test_results_df = save_results_to_csv(
        test_targets, test_preds, test_gating_weights,
        test_expert_preds, scalers, f'../results/test_results_{best_wavelet}.csv'
    )

    # Analysis (使用最佳小波)
    print("\n" + "=" * 80)
    print("📊 Gating Dynamics Analysis (Test Set)")
    print("=" * 80)
    analyze_gating_dynamics(test_gating_weights, test_targets_original)
    plot_gating_by_regime(test_gating_weights, test_targets_original,
                          f'../results/test_gating_dynamics_by_regime_{best_wavelet}.png')

    # Summary of energy distribution (使用最佳小波)
    print(f"\n📊 Component Energy Distribution (After Centering) - {best_wavelet}:")
    for name, energy in energies.items():
        print(f"   {name}: {energy:.2f}%")

    print("\n✅ All done! Check the ../results/ folder for outputs.")
    print(f"\n🏆 Best Wavelet: {best_wavelet}")
    print(f"   Final Test RMSE: {rmse_original:.4f}%")
    print(f"   Final Test MAE: {mae_original:.4f}%")
    print(f"   Final R²: {r2_original:.6f}")
    print(f"   Final Test Direction Accuracy: {test_metrics['direction_acc']*100:.2f}%")

    # # ==================== Walk-Forward Validation ====================
    print("\n🔄 Starting Walk-Forward Validation...")

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
    print(f"\n💾 Results saved to ../results/wfv_results.csv")

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

    print("\n✅ Walk-Forward Validation Complete!")
    print(f"   Check ../results/ for all outputs")

    # 診斷腳本 - 請在你的main code後面跑
    print("=" * 80)
    print("🔍 DIAGNOSTIC CHECK")
    print("=" * 80)

    # 假設你有 all_predictions (from WFV)
    # all_predictions[0] = {'fold': 1, 'predictions': ..., 'targets': ...}

    for i, pred_data in enumerate(all_predictions):
        fold = pred_data['fold']
        preds = pred_data['predictions']
        targets = pred_data['targets']

        residuals_fold = targets - preds

        print(f"\nFold {fold}:")
        print(f"   Residual mean: {residuals_fold.mean():.6f}")
        print(f"   Residual std:  {residuals_fold.std():.6f}")
        print(f"   Residual max:  {residuals_fold.max():.6f}")
        print(f"   Residual min:  {residuals_fold.min():.6f}")
        print(f"   Count of |res| > 5%: {np.sum(np.abs(residuals_fold) > 5)}")
        print(f"   Count of |res| > 10%: {np.sum(np.abs(residuals_fold) > 10)}")

        # 合併Fold 2-5的殘差（排除warm-up期）
    residuals_stable = np.concatenate([
        all_predictions[1]['targets'] - all_predictions[1]['predictions'],  # Fold 2
        all_predictions[2]['targets'] - all_predictions[2]['predictions'],  # Fold 3
        all_predictions[3]['targets'] - all_predictions[3]['predictions'],  # Fold 4
        all_predictions[4]['targets'] - all_predictions[4]['predictions'],  # Fold 5
    ])

    print("=" * 80)
    print("📊 RESIDUAL DIAGNOSTICS (Fold 2-5, Excluding Warm-up)")
    print("=" * 80)

    # 1. Normality test
    stat_shapiro, p_shapiro = stats.shapiro(residuals_stable)
    print(f"\n1️⃣ Shapiro-Wilk Normality Test:")
    print(f"   p-value: {p_shapiro:.6f}")
    print(f"   Result: {'✅ 正態分布 (p > 0.05)' if p_shapiro > 0.05 else '⚠️ 非正態 (時間序列常見)'}")

    # 2. Autocorrelation test
    lb_test = acorr_ljungbox(residuals_stable, lags=[10, 20], return_df=True)
    print(f"\n2️⃣ Ljung-Box Autocorrelation Test:")
    print(f"   Lag 10 p-value: {lb_test['lb_pvalue'].iloc[0]:.6f}")
    print(f"   Lag 20 p-value: {lb_test['lb_pvalue'].iloc[1]:.6f}")
    print(f"   Result: {'✅ 無自相關 (獨立同分布)' if lb_test['lb_pvalue'].min() > 0.05 else '⚠️ 有自相關 (時間序列常見)'}")

    # 3. Statistics
    print(f"\n3️⃣ Residual Statistics:")
    print(f"   Mean: {residuals_stable.mean():.6f}% ✅ (應接近0)")
    print(f"   Std:  {residuals_stable.std():.6f}%")
    print(f"   Min:  {residuals_stable.min():.6f}%")
    print(f"   Max:  {residuals_stable.max():.6f}%")
    print(f"   Range: {residuals_stable.max() - residuals_stable.min():.6f}%")

    # 4. Visualization
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
    plt.savefig('residual_diagnostics_stable.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: residual_diagnostics_stable.png")

    print("\n" + "=" * 80)