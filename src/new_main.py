"""
New Main: End-to-End Frequency-Aware Residual MoE (E2E-FAR-MoE)
================================================================
Academic Novelty:
1. Replaces static MODWT with Learnable Decomposition (inspired by Autoformer, NeurIPS 2021).
2. Joint optimization of decomposition and forecasting (Task-Aware).
3. Cross-Domain Gating Mechanism with Two-Stage Curriculum Learning.

Run: python new_main.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import random


warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent

dataset_path = BASE_DIR / "dataset" / "USD_TWD.csv"
save_path = BASE_DIR / "results"

if not dataset_path.exists():
    raise FileNotFoundError(f"找不到檔案，確認路徑: {dataset_path}")

# ==================== Configuration ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SEED = random.randint(0, 10000)  # Fixed seed for reproducibility
SEED = 5827

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print(f"[Setup] Device: {DEVICE} | Seed: {SEED}")


# ==================== 1. Learnable Decomposition Layers (The SOTA Part) ====================
class MovingAvg(nn.Module):

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding to keep sequence length unchanged
        # x shape: [Batch, Seq_Len, Channels]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        # Permute for AvgPool1d: [Batch, Channels, Seq_Len]
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


# ==================== 2. Expert & Base Networks ====================
class SimpleGRUExpert(nn.Module):
    """
    沒有 Attention 的單純 GRU 專家
    直接使用最後一個時間點的 Hidden State 進行預測
    """

    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

        # 移除 Attention 相關層，保留最後的預測層
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        # gru_out: [batch, seq, hidden] -> 包含每個時間點的輸出
        # h_n: [num_layers, batch, hidden] -> 包含最後時間點的狀態
        gru_out, h_n = self.gru(x)

        # 取最後一層的最後一個時間點狀態 (Last Hidden State)
        # h_n[-1] 形狀為 [batch, hidden]
        last_hidden = h_n[-1]

        context = self.dropout(last_hidden)

        # 回傳預測值，以及 None (因為沒有 Attention Weight 了)
        return self.fc(context), None


class RawLSTM(nn.Module):
    """Base Branch: Standard LSTM"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        return self.fc(self.dropout(h_n[-1]))


# ==================== 3. The Main Architecture (E2E-FAR-MoE) ====================
class AdvancedLearnableMoE(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super().__init__()

        # --- A. Learnable Decomposition (The Innovation) ---
        # Kernel 25 ~= 1 month of trading days (extracts long-term trend)
        self.decomp = SeriesDecomp(kernel_size=15)

        # --- B. Base Branch (Anchor) ---
        self.base_branch = RawLSTM(input_size, hidden_size)

        # --- C. Experts ---
        # Expert 1: Focuses on Trend (Low Freq)
        self.expert_trend = SimpleGRUExpert(input_size, hidden_size=32)
        # Expert 2: Focuses on Seasonal/Residual (High Freq)
        self.expert_cyclic = SimpleGRUExpert(input_size, hidden_size=32)

        # --- D. Gating Network ---
        # Inputs: Last Step (3) + Trend Mean (3) + Seasonal Mean (3) = 9 features
        self.gating = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),  # 2 Experts
        )

        # --- E. Meta-Gate (Alpha) ---
        self.meta_gate = nn.Sequential(
            nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 1)
        )

        nn.init.constant_(self.meta_gate[-1].bias, 0.0)
        nn.init.xavier_uniform_(self.meta_gate[-1].weight, gain=0.01) # 權重也設小一點

    def forward(self, x):
        # x: [Batch, Seq, Features]

        # 1. End-to-End Decomposition
        seasonal_part, trend_part = self.decomp(x)

        # 2. Base Prediction
        base_delta = self.base_branch(x)

        # 3. Expert Predictions
        pred_trend, attn_trend = self.expert_trend(trend_part)
        pred_cyclic, attn_cyclic = self.expert_cyclic(seasonal_part)

        # 4. Context for Gating (Time-Domain Context)
        last_step = x[:, -1, :]
        trend_mean = trend_part.mean(dim=1)
        seasonal_mean = seasonal_part.mean(dim=1)
        context = torch.cat([last_step, trend_mean, seasonal_mean], dim=1)

        # 5. Gating Weights (With Temperature Sharpening)
        gating_logits = self.gating(context)  # [Batch, 2]

        # ★★★ 修改點 2: 設定溫度係數 (T) ★★★
        # T 越小，分佈越極端 (雙峰)。建議嘗試 0.1, 0.2, 0.5
        temperature = 0.1
        expert_weights = torch.softmax(gating_logits / temperature, dim=1)

        w_trend, w_cyclic = expert_weights[:, 0:1], expert_weights[:, 1:2]

        # 6. Residual Fusion
        moe_delta = (w_trend * pred_trend) + (w_cyclic * pred_cyclic)
        alpha_logit = self.meta_gate(context)
        alpha = torch.sigmoid(alpha_logit)

        # 7. Final Output (Anchor + Residual)
        # Prev Value is Volatility (Index 0 in feature list)
        prev_value = x[:, -1, 0:1]
        total_delta = base_delta + (alpha * moe_delta)
        output = prev_value + total_delta

        return (
            output,
            total_delta,
            moe_delta,
            expert_weights,
            {"trend": attn_trend, "cyclic": attn_cyclic},
            alpha,
        )


# ==================== 4. Loss Function ====================
class HybridDirectionalLoss(nn.Module):
    def __init__(self, direction_weight=0.5):
        super().__init__()
        self.huber = nn.HuberLoss(delta=1.0)
        self.dir_weight = direction_weight

    def forward(self, pred, target, prev_value):
        # 1. 數值精確度 (Magnitude)
        loss_val = self.huber(pred, target)

        # 2. 方向懲罰 (Direction Penalty)
        # 我們不關心波動大小，只關心：預測的 delta 和 真實的 delta 是否同號？
        true_delta = target - prev_value
        pred_delta = pred - prev_value

        # 使用 tanh 模擬 sign 函數，但保持可微分
        # 如果符號相同 (++) 或 (--)，乘積為正 -> tanh 為正
        # 如果符號相反 (+-) 或 (-+)，乘積為負 -> tanh 為負
        # check
        sign_agreement = torch.tanh(true_delta * 10) * torch.tanh(pred_delta * 10)

        # 目標是 maximize sign_agreement (讓它接近 1)
        # 所以 Loss 是 1 - sign_agreement
        # 結果範圍：0 (方向完全正確) ~ 2 (方向完全相反)
        dir_loss = torch.mean(1 - sign_agreement)

        # 3. 組合
        # 這裡的 dir_loss 不再受波動大小影響，即使波動很小，
        # 只要方向錯了，Loss 就是 2，這會給模型很大的修正梯度！
        return (1 - self.dir_weight) * loss_val + self.dir_weight * dir_loss


# ==================== 5. Data Preparation (No MODWT Needed!) ====================
class VolatilityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"raw_input": self.X[idx], "target": self.y[idx]}


def prepare_data(df, vol_window=7, lookback=30, mode="NORMAL"):
    print(f"[Data Prep] Mode: {mode} | Features: Volatility, RSI, LogReturn")

    # 1. Indicators
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = df["log_return"].rolling(vol_window).std() * np.sqrt(252) * 100

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    df = df.dropna().reset_index(drop=True)

    # 2. Select Features (Order: Volatility, RSI, Return)
    # Volatility is at index 0 (Target)
    features = df[["Volatility", "RSI", "log_return"]].values
    targets = df["Volatility"].values

    # 3. Train/Test Split (80/20)
    split_idx = int(len(features) * 0.8)
    train_feat, test_feat = features[:split_idx], features[split_idx:]

    # 4. Scaling
    scaler_vol = StandardScaler()
    # Fit only on train to avoid leakage
    train_feat_scaled = train_feat.copy()
    test_feat_scaled = test_feat.copy()

    # Scale each channel independently
    scalers = {}
    for i in range(3):
        s = StandardScaler()
        train_feat_scaled[:, i] = s.fit_transform(
            train_feat[:, i].reshape(-1, 1)
        ).flatten()
        test_feat_scaled[:, i] = s.transform(test_feat[:, i].reshape(-1, 1)).flatten()
        if i == 0:
            scalers["target"] = s  # Save Volatility scaler

    # 5. Sliding Window
    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i : i + lookback])
            # Target is the NEXT day's volatility (index 0)
            y.append(data[i + lookback, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_feat_scaled, lookback)
    X_test, y_test = create_sequences(test_feat_scaled, lookback)

    # --- SANITY CHECK LOGIC ---
    if mode == "SANITY_TARGET_SHUFFLE":
        np.random.shuffle(y_train)
        np.random.shuffle(y_test)
        print("⚠️ WARNING: Targets shuffled! (Expect R2 < 0)")

    elif mode == "SANITY_INPUT_SHUFFLE":
        # Shuffle inputs but keep targets real
        idx_train = np.random.permutation(len(X_train))
        X_train = X_train[idx_train]
        idx_test = np.random.permutation(len(X_test))
        X_test = X_test[idx_test]
        print("⚠️ WARNING: Inputs shuffled! (Expect R2 < 0, Dir Acc ~50%)")

    # 6. Dataloaders
    train_loader = DataLoader(
        VolatilityDataset(X_train, y_train), batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        VolatilityDataset(X_test, y_test), batch_size=32, shuffle=False
    )

    return train_loader, test_loader, scalers, df["Volatility"][:split_idx].mean(), mode


# ==================== 6. Training & Evaluation ====================
def train_model(train_loader, test_loader, num_epochs=100, lr=0.001):
    model = AdvancedLearnableMoE().to(DEVICE)
    criterion = HybridDirectionalLoss(direction_weight=0.4)

    # Stage 1: Freeze Experts
    print("\n[Stage 1] Pre-training Base Branch...")
    optimizer = torch.optim.Adam(model.base_branch.parameters(), lr=lr)

    for epoch in range(num_epochs // 2):
        model.train()
        losses = []
        for batch in train_loader:
            x, y = (
                batch["raw_input"].to(DEVICE),
                batch["target"].to(DEVICE).unsqueeze(-1),
            )
            optimizer.zero_grad()

            # =================================================
            # 【關鍵修正】
            # 1. 不呼叫 model(x)，因為那會跑去算 Expert 和 Gating
            #    我們直接單獨呼叫 base_branch，拿到純淨的 delta
            # =================================================
            base_delta = model.base_branch(x)

            # 2. 【尺度修正】手動還原成絕對數值 (Prediction = Prev + Delta)
            #    這樣才能跟 Target (y) 在同一個基準上比較
            prev_val = x[:, -1, 0:1]
            stage1_pred = prev_val + base_delta

            # 3. 計算 Loss
            #    這時候的 stage1_pred 是純粹由 LSTM 算出來的預測值
            loss = criterion(stage1_pred, y, prev_val)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1} | Loss: {np.mean(losses):.4f}")
    #
    # # [Stage 1.5] Expert Warm-up (關鍵新增)
    # print("\n[Stage 1.5] Warming up Experts (Alpha=1 forced)...")
    # # 凍結 Base
    # for param in model.base_branch.parameters():
    #     param.requires_grad = False
    #
    # # 訓練 Experts (Trend/Cyclic) 和 Gating
    # optimizer_exp = torch.optim.Adam([
    #     {'params': model.expert_trend.parameters()},
    #     {'params': model.expert_cyclic.parameters()},
    #     {'params': model.gating.parameters()}
    # ], lr=lr)
    #
    # for epoch in range(10): # 短暫預熱 10 個 Epoch
    #     model.train()
    #     for batch in train_loader:
    #         x, y = batch['raw_input'].to(DEVICE), batch['target'].to(DEVICE).unsqueeze(-1)
    #         optimizer_exp.zero_grad()
    #
    #         # 強制只訓練 Expert 部分，假設 Alpha=1
    #         # 我們需要手動呼叫 forward 的內部邏輯，或者在 model 加一個 warmup 模式
    #         # 這裡用一個簡化寫法：直接拿 moe_delta 算 Loss
    #
    #         seasonal_part, trend_part = model.decomp(x)
    #         pred_trend, _ = model.expert_trend(trend_part)
    #         pred_cyclic, _ = model.expert_cyclic(seasonal_part)
    #
    #         # Gating
    #         last_step = x[:, -1, :]
    #         trend_mean = trend_part.mean(dim=1)
    #         seasonal_mean = seasonal_part.mean(dim=1)
    #         context = torch.cat([last_step, trend_mean, seasonal_mean], dim=1)
    #         expert_weights = model.gating(context)
    #
    #         moe_delta = (expert_weights[:, 0:1] * pred_trend) + (expert_weights[:, 1:2] * pred_cyclic)
    #
    #         # 讓 Experts 試著去預測 "殘差" (Target - Base_Prediction)
    #         with torch.no_grad():
    #             base_delta = model.base_branch(x)
    #             prev_val = x[:, -1, 0:1]
    #             # 目標是：Expert 應該要把 Base 沒預測準的部分補起來
    #             # Residual Target = (True_Target - Prev) - Base_Delta
    #             residual_target = (y - prev_val) - base_delta
    #
    #         # 這裡只算 MSE 讓 Expert 快速收斂
    #         loss = nn.MSELoss()(moe_delta, residual_target)
    #         loss.backward()
    #         optimizer_exp.step()
    #     print(f"  Warmup Epoch {epoch+1}")
    #
    # # 解凍 Base，準備進入 Stage 2
    # for param in model.base_branch.parameters():
    #     param.requires_grad = True

    # Stage 2: Joint Training
    print("\n[Stage 2] Joint Training ...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.5)  # Lower LR

    history = {"loss": [], "alpha": []}

    for epoch in range(num_epochs // 2):
        model.train()
        epoch_loss = []
        epoch_alpha = []

        for batch in train_loader:
            x, y = (
                batch["raw_input"].to(DEVICE),
                batch["target"].to(DEVICE).unsqueeze(-1),
            )
            optimizer.zero_grad()
            out, _, _, _, _, alpha = model(x)
            prev_val = x[:, -1, 0:1]
            loss = criterion(out, y, prev_val)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss.append(loss.item())
            epoch_alpha.append(alpha.mean().item())

        avg_loss = np.mean(epoch_loss)
        history["loss"].append(avg_loss)
        history["alpha"].append(np.mean(epoch_alpha))

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch + 1 + num_epochs // 2} | Loss: {avg_loss:.4f} | Alpha: {np.mean(epoch_alpha):.3f}"
            )

    return model, history


@torch.no_grad()
def evaluate(model, loader, scaler, mean_vol):
    model.eval()
    preds, targets, alphas, expert_weights_list = [], [], [], []

    for batch in loader:
        x, y = batch["raw_input"].to(DEVICE), batch["target"].to(DEVICE)
        out, _, _, weights, _, alpha = model(x)
        preds.append(out.cpu().numpy())
        targets.append(y.cpu().numpy())
        alphas.append(alpha.cpu().numpy())
        expert_weights_list.append(weights.cpu().numpy()) # [Batch, 2]

    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(targets).flatten()
    alphas = np.concatenate(alphas).flatten()

    all_weights = np.concatenate(expert_weights_list, axis=0)

    # Inverse Transform
    preds_orig = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    targets_orig = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

    return preds_orig, targets_orig, alphas, all_weights

def analyze_high_volatility(targets, preds):
    """
    分析模型在「大波動」日子的表現 vs 整體表現
    targets, preds 都是原始數值 (Inverse Transformed)
    """
    # 1. 對齊數據 (計算 Delta 需要 t 與 t-1)
    true_delta = targets[1:] - targets[:-1]
    pred_delta = preds[1:] - targets[:-1]

    # 2. 計算方向是否正確
    dir_correct = (np.sign(true_delta) == np.sign(pred_delta))

    # 3. 定義「大行情」 (Top 20% 的波動幅度)
    magnitude = np.abs(true_delta)
    threshold = np.percentile(magnitude, 80)
    high_vol_mask = magnitude > threshold

    # 4. 計算準確率
    acc_overall = np.mean(dir_correct)
    acc_high_vol = np.mean(dir_correct[high_vol_mask])

    return acc_overall, acc_high_vol, threshold

def evaluate_naive(targets_orig):
    # Lag-1 Prediction: Pred[t] = Target[t-1]
    # We must align arrays: Target[1:] vs Target[:-1]
    y_true = targets_orig[1:]
    y_pred = targets_orig[:-1]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    dir_acc = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
    r2 = r2_score(y_true, y_pred)
    return rmse, dir_acc, r2


def plot_alpha_distribution(alphas, save_path=None):
    if save_path == None:
        raise ValueError("路徑為空")

    plt.figure(figsize=(10, 6))
    plt.hist(alphas, bins=50, color="#9467bd", alpha=0.7, edgecolor="black")
    plt.axvline(
        alphas.mean(), color="red", linestyle="--", label=f"Mean: {alphas.mean():.4f}"
    )
    plt.title("Distribution of Meta-Gate Alpha (Adaptability Check)")
    plt.xlabel("Alpha (0=Base, 1=Expert)")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Alpha plot saved: {save_path}")


def plot_learned_decomposition(model, test_loader, save_path=None):
    if save_path == None:
        raise ValueError("路徑為空")

    model.eval()

    # 取一個 batch 的資料
    batch = next(iter(test_loader))
    x = batch["raw_input"].to(DEVICE)  # [Batch, Seq, Feat]

    # 讓模型分解
    with torch.no_grad():
        seasonal, trend = model.decomp(x)

    # 轉成 numpy，只畫第一筆資料 (Sample 0)
    # x: [Batch, Seq, Feat], 我們只取 Volatility (Index 0)
    raw_seq = x[0, :, 0].cpu().numpy()
    trend_seq = trend[0, :, 0].cpu().numpy()
    seasonal_seq = seasonal[0, :, 0].cpu().numpy()

    # 繪圖
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(raw_seq, label="Raw Input (Volatility)", color="black")
    axes[0].set_title("Original Input Series")
    axes[0].legend()

    axes[1].plot(
        trend_seq, label="Trend (Low Freq)", color="green", linewidth=2
    )
    axes[1].set_title("Trend")
    axes[1].legend()

    axes[2].plot(
        seasonal_seq, label="Seasonal (Residual)", color="orange", linewidth=2
    )
    axes[2].set_title("Seasonal/Residual")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Decomposition plot saved: {save_path}")

def plot_expert_analysis(weights_data, save_path=None):
    if save_path is None:
        raise ValueError("路徑為空")

    # weights_data: shape [N, 2], column 0=Trend, column 1=Cyclic

    # 設定風格
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # --- 子圖 1: 時間序列堆疊圖 ---
    time_steps = range(len(weights_data))
    axes[0].stackplot(time_steps,
                      weights_data[:, 0], # Trend
                      weights_data[:, 1], # Cyclic
                      labels=['Trend Expert', 'Cyclic Expert'],
                      colors=['#1f77b4', '#ff7f0e'],
                      alpha=0.8)
    axes[0].set_title('Dynamic Expert Weight Allocation Over Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time Step (Test Set)')
    axes[0].set_ylabel('Gate Weight')
    axes[0].legend(loc='upper right')

    # ★★★ 修正這裡 ★★★
    # 原本錯誤: axes[0].set_xmargin(0, 0) 或 axes[0].set_margins(0, 0)
    # 正確寫法: 指定 x 軸留白為 0
    axes[0].margins(x=0)

    # --- 子圖 2: 權重密度分佈 (KDE) ---
    sns.kdeplot(data=weights_data[:, 0], ax=axes[1], color='#1f77b4', fill=True, label='Trend Expert', alpha=0.3, clip=(0,1))
    sns.kdeplot(data=weights_data[:, 1], ax=axes[1], color='#ff7f0e', fill=True, label='Cyclic Expert', alpha=0.3, clip=(0,1))

    axes[1].set_title('Distribution of Expert Weights (Specialization Check)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Weight Value (0 to 1)')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # 恢復 matplotlib 預設風格
    sns.reset_orig()
    print(f"✓ Expert weights plot saved: {save_path}")


def plot_metric_comparison(metrics_dict, save_path=None):
    """
    動態調整子圖數量，適用於繪製 RMSE 和 R2
    """
    if save_path is None: raise ValueError("路徑為空")

    metrics_names = list(metrics_dict.keys())
    num_metrics = len(metrics_names) # 自動計算有幾個指標

    # 設定畫布：動態寬度
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 6))

    # 如果只有一個指標，axes 不會是 list，需要轉一下以便迴圈通用 (雖然這裡我們預計會有 2 個)
    if num_metrics == 1: axes = [axes]

    color_model = '#2ca02c'  # 綠色
    color_naive = '#7f7f7f'  # 灰色

    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        model_val = metrics_dict[metric][0]
        naive_val = metrics_dict[metric][1]

        x_labels = ['Our Model', 'Naive']
        values = [model_val, naive_val]
        colors = [color_model, color_naive]

        bars = ax.bar(x_labels, values, color=colors, alpha=0.8, width=0.6)

        ax.set_title(metric, fontweight='bold', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}' if height < 10 else f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=12)

    fig.suptitle('Regression Metrics Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Regression metrics plot saved: {save_path}")

def plot_performance_comparison(overall_acc, high_vol_acc, naive_overall, naive_high_vol, save_path=None):
    if save_path is None: raise ValueError("路徑為空")

    categories = ['Overall Accuracy', 'High Volatility\nDirectional Acc.']
    model_scores = [overall_acc, high_vol_acc]
    naive_scores = [naive_overall, naive_high_vol]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    # 你的模型 (深紫色)
    rects1 = ax.bar(x - width/2, model_scores, width, label='Proposed Model', color='#845EC2', edgecolor='black', alpha=0.9, zorder=3)
    # Naive Baseline (灰色 + 斜線)
    rects2 = ax.bar(x + width/2, naive_scores, width, label='Naive Baseline', color='#B0A8B9', edgecolor='black', hatch='//', zorder=3)

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Performance Comparison: Model vs Naive Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc='upper left')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1%}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Performance comparison plot saved: {save_path}")

# ==================== 8. Leakage Zoom-In Check (Add this to the end) ====================
def plot_zoom_check(targets, preds, save_path=None):
    if save_path == None:
        raise ValueError("路徑為空")

    import matplotlib.pyplot as plt

    # 找出波動率最大的那天 (尖峰)
    peak_idx = np.argmax(targets)

    # 取前後 50 天
    start = max(0, int(peak_idx) - 50)
    end = min(len(targets), int(peak_idx) + 50)

    plt.figure(figsize=(12, 6))

    # 畫出真實值 (黑線)
    plt.plot(
        range(start, end),
        targets[start:end],
        label="Actual Volatility",
        color="black",
        linewidth=2,
        marker="o",
        markersize=4,
    )

    # 畫出預測值 (藍線)
    plt.plot(
        range(start, end),
        preds[start:end],
        label="E2E-MoE Prediction",
        color="blue",
        linewidth=2,
        linestyle="--",
        marker="x",
        markersize=4,
    )

    plt.axvline(float(peak_idx), color="red", linestyle=":", alpha=0.5, label="Peak Day")

    plt.title(
        f"Zoom-in Check (Day {start} to {end})\nCheck if Blue line lags behind Black line (Correct) or leads it (Leakage)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ★★★ 關鍵：存檔而不是顯示 ★★★
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Zoom-in check plot saved to: {save_path}")


# ==================== 7. Main Execution ====================
if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)

    # 1. Prepare Data
    df = pd.read_csv(dataset_path)
    train_loader, test_loader, scalers, mean_vol, mode = prepare_data(df, mode="NORMAL")

    # 2. Train
    model, history = train_model(train_loader, test_loader)

    # 3. Evaluate Model
    preds, targets, alphas, expert_weights = evaluate(model, test_loader, scalers["target"], mean_vol)

    # 4. Evaluate Naive Baseline (這裡會算出 naive_dir)
    naive_rmse, naive_dir, naive_r2 = evaluate_naive(targets)

    # 5. Metrics Calculation
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)

    # ★★★ 呼叫大行情分析 (取得 acc_overall) ★★★
    acc_overall, acc_high_vol, vol_threshold = analyze_high_volatility(targets, preds)

    # ★★★ 計算差異：我們的新指標 vs Naive 指標 ★★★
    improvement = acc_high_vol - acc_overall # 這是比較大行情 vs 整體

    # 這是比較 Model vs Naive
    win_naive = acc_overall > naive_dir

    # 7. 計算 Naive 在高波動下的表現 (為了畫圖比較)
    # 邏輯：Naive 預測方向 = 昨天的方向 (Momentum)
    # 我們必須對齊數據：
    # true_delta_all: 從 t=1 到 t=N 的真實變化
    true_delta_all = targets[1:] - targets[:-1]

    # 我們比較範圍：從 index 1 開始 (因為第 0 個數據沒有"昨天")
    # True Direction (今天): targets[t] - targets[t-1]
    true_dir_aligned = np.sign(true_delta_all[1:])

    # Naive Prediction (猜跟昨天一樣): targets[t-1] - targets[t-2]
    naive_dir_aligned = np.sign(true_delta_all[:-1])

    # 找出對應的高波動日子
    magnitude_aligned = np.abs(true_delta_all[1:])
    high_vol_mask_naive = magnitude_aligned > vol_threshold

    # 計算 Naive 在這些日子的準確率
    naive_correct_mask = (true_dir_aligned == naive_dir_aligned)
    if np.sum(high_vol_mask_naive) > 0:
        naive_high_vol_acc = np.mean(naive_correct_mask[high_vol_mask_naive])
    else:
        naive_high_vol_acc = 0.5 # 防止分母為 0

    improvement = acc_high_vol - acc_overall # 這是比較大行情 vs 整體

    # 6. Final Report
    print("\n" + "=" * 60)

    if "SANITY" in mode:
        print(f"⚠️ SANITY CHECK REPORT ({mode})")
        print(f"R2 Score: {r2:.4f} (Should be < 0)")
        print(f"Dir Acc:  {acc_overall * 100:.2f}% (Should be ~50%)")
    else:
        print("✅ FINAL PRODUCTION REPORT (E2E-FAR-MoE)")
        print("-" * 60)
        print(f"{'Metric':<15} | {'Our Model':<12} | {'Naive (Lag-1)':<12} | {'Status'}")
        print("-" * 60)

        # RMSE
        print(f"{'RMSE':<15} | {rmse:<12.4f} | {naive_rmse:<12.4f} | {'Comparable' if abs(rmse - naive_rmse) < 0.1 else 'Check'}")

        # R2
        print(f"{'R2 Score':<15} | {r2:<12.4f} | {naive_r2:<12.4f} | {'Good' if r2 > 0.8 else 'Low'}")

        # ★★★ [這裡補上] 方向準確率比較 ★★★
        # 使用 acc_overall (Model) 對決 naive_dir (Baseline)
        status = "WIN" if win_naive else "LOSE"
        print(f"{'Dir Accuracy':<15} | {acc_overall * 100:<12.2f}% | {naive_dir * 100:<12.2f}% | {status}")


        # Deep Dive (保持不變)
        print("-" * 60)
        print("[Direction Accuracy Deep Dive] 🎯")
        print(f"整體方向準確率 (Overall):       {acc_overall * 100:.2f}%")
        print(f"大行情準確率 (Top 20% Vol):     {acc_high_vol * 100:.2f}%  (Threshold > {vol_threshold:.4f})")

        # 這裡的 Improvement 是指「大行情有沒有比整體準」
        imp_status = "Positive" if improvement > 0 else "Negative"
        print(f"大行情提升幅度:                 {improvement * 100:+.2f}%  ({imp_status})")

        print("-" * 60)
        print(f"Mean Alpha: {alphas.mean():.4f}")

        # Plotting (保持不變)
        plot_alpha_distribution(alphas, save_path / "alpha_dist.png")
        plot_learned_decomposition(model, test_loader, save_path / "decomposition.png")
        plot_zoom_check(targets, preds, save_path / "zoom_check.png")
        plot_expert_analysis(expert_weights, save_path / "expert_analysis.png")

        # Prediction Plot
        plt.figure(figsize=(12, 6))
        plt.plot(targets, label="Actual Volatility", color="black", alpha=0.6)
        plt.plot(preds, label="E2E-MoE Prediction", color="blue", alpha=0.8)
        plt.title(f"Forecast vs Actual (Overall Dir Acc: {acc_overall * 100:.2f}%)")
        plt.legend()
        plt.savefig(save_path / "final_forecast.png")
        print("✓ Forecast plot saved.")

        # 1. 繪製數值回歸指標 (RMSE & R2)
        # 這裡不放 Dir Acc，因為會在下一張圖專門畫
        regression_metrics = {
            'RMSE (Lower is better)': (rmse, naive_rmse),
            'R2 Score (Higher is better)': (r2, naive_r2)
        }
        plot_metric_comparison(regression_metrics, save_path / "metric_comparison.png")

        # 2. 繪製方向準確度指標 (Overall vs High Volatility)
        # 這張圖專門負責講 "方向預測" 的故事
        plot_performance_comparison(
            overall_acc=acc_overall,
            high_vol_acc=acc_high_vol,
            naive_overall=naive_dir,
            naive_high_vol=naive_high_vol_acc,
            save_path=save_path / "performance_comparison.png"
        )

    print("=" * 60 + "\n")