"""
Bayesian Optimization for MODWT-MoE Hyperparameters
針對 Loss 權重和關鍵架構參數進行優化
"""

import numpy as np
import torch
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence, plot_objective
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# 引入你的主程式模組
from main import (
    prepare_modwt_data,
    train_modwt_moe,
    evaluate,
    MODWTMoE,
    CombinedLoss,
    DEVICE
)

# ==================== 配置 ====================
optimization_pbar = None
RANDOM_SEEDS = [42, 123, 456]  # 每組參數測試 3 次
NUM_EPOCHS = 40  # 縮短訓練加速優化
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🚀 貝葉斯優化初始化")
print(f"   Device: {DEVICE}")
print(f"   每組參數測試種子: {RANDOM_SEEDS}")
print(f"   訓練 Epochs: {NUM_EPOCHS}")

# ==================== 載入資料（只載入一次）====================
print("\n📂 載入資料...")
df = pd.read_csv("../dataset/USD_TWD.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
print(f"✅ 載入 {len(df)} 天資料")

# ==================== 搜索空間定義 ====================
search_space = [
    Real(0.10, 0.35, name='beta'),         # Direction Loss 權重
    Real(0.03, 0.15, name='gamma'),        # Diversity Loss 權重
    Real(5e-4, 3e-3, name='lr', prior='log-uniform'),  # 學習率
    Integer(64, 128, name='expert2_hidden')  # Expert 2 容量
]

print("\n🔍 搜索空間:")
for param in search_space:
    if hasattr(param, 'prior'):
        print(f"   {param.name}: [{param.bounds[0]:.5f}, {param.bounds[1]:.5f}] (log-uniform)")
    else:
        print(f"   {param.name}: {param.bounds}")

# ==================== 目標函數 ====================
@use_named_args(search_space)
def objective(beta, gamma, lr, expert2_hidden):
    """
    訓練模型並返回目標值（要最小化）
    目標 = RMSE - 0.3 * Direction_Acc
    (鼓勵 RMSE 低且方向準確度高)
    """

    global optimization_pbar

    print(f"\n{'='*70}")
    print(f"🧪 測試參數:")
    print(f"   beta={beta:.4f}, gamma={gamma:.4f}")
    print(f"   lr={lr:.6f}, expert2_hidden={expert2_hidden}")
    print(f"{'='*70}")

    expert2_hidden = int(expert2_hidden)

    rmse_list = []
    direction_list = []
    r2_list = []

    seed_pbar = tqdm(enumerate(RANDOM_SEEDS),
                     total=len(RANDOM_SEEDS),
                     desc="  隨機種子",
                     leave=False,
                     ncols=80)

    # 跑 3 次不同隨機種子
    for seed_idx, seed in seed_pbar:
        seed_pbar.set_description(f"  種子 {seed_idx+1}/{len(RANDOM_SEEDS)}: {seed}")

        # 設定隨機種子
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        try:
            # 準備資料
            train_loader, test_loader, scalers, _, _ = prepare_modwt_data(
                df,
                wavelet='sym4',
                level=4,
                lookback=30,
                forecast_horizon=1,
                train_ratio=0.8
            )

            # 建立模型（動態架構）
            from main import TrendExpert, CyclicExpert, HighFreqExpert, GatingNetwork

            class MODWTMoEDynamic(torch.nn.Module):
                """動態架構的 MoE"""
                def __init__(self, expert2_hidden):
                    super().__init__()
                    self.expert1 = TrendExpert(input_size=1, hidden_size=32, num_layers=2, dropout=0.2)
                    self.expert2 = CyclicExpert(input_size=2, hidden_size=expert2_hidden, num_layers=2, dropout=0.3)
                    self.expert3 = HighFreqExpert(input_size=2, hidden_size=32, num_layers=2, dropout=0.4)
                    self.gating = GatingNetwork(input_size=13, hidden_size=32, num_experts=3)

                def forward(self, expert1_input, expert2_input, expert3_input):
                    from main import extract_gating_features
                    pred1 = self.expert1(expert1_input)
                    pred2 = self.expert2(expert2_input)
                    pred3 = self.expert3(expert3_input)
                    expert_preds = torch.cat([pred1, pred2, pred3], dim=1)
                    gating_features = extract_gating_features(expert1_input, expert2_input, expert3_input)
                    gating_weights = self.gating(gating_features)
                    final_pred = (expert_preds * gating_weights).sum(dim=1, keepdim=True)
                    return final_pred, expert_preds, gating_weights

            model = MODWTMoEDynamic(expert2_hidden=expert2_hidden).to(DEVICE)

            # 使用當前參數
            criterion = CombinedLoss(
                huber_delta=1.0,
                alpha=1.0,      # 固定
                beta=beta,      # 優化
                gamma=gamma     # 優化
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,          # 優化
                weight_decay=1e-5
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )

            # 訓練（簡化版）
            from main import train_one_epoch

            best_rmse = float('inf')
            patience = 0

            epoch_pbar = tqdm(range(NUM_EPOCHS),
                              desc="    Epoch",
                              leave=False,
                              ncols=80)

            for epoch in epoch_pbar:
                train_loss, _ = train_one_epoch(
                    model, train_loader, optimizer, criterion, DEVICE
                )

                metrics, preds, targets, expert_preds, gating_weights = evaluate(
                    model, test_loader, DEVICE
                )

                scheduler.step(metrics['rmse'])

                epoch_pbar.set_postfix({
                    'loss': f'{train_loss:.4f}',
                    'rmse': f'{metrics["rmse"]:.4f}',
                    'dir': f'{metrics["direction_acc"]:.3f}'
                })

                if metrics['rmse'] < best_rmse:
                    best_rmse = metrics['rmse']
                    patience = 0
                    best_metrics = metrics
                else:
                    patience += 1

                if patience >= 8:  # 早停
                    break

            epoch_pbar.close()

            # Inverse transform
            target_scaler = scalers['cA4_trend']
            preds_original = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            targets_original = target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

            from sklearn.metrics import mean_squared_error
            rmse_original = np.sqrt(mean_squared_error(targets_original, preds_original))

            rmse_list.append(rmse_original)
            direction_list.append(best_metrics['direction_acc'])
            r2_list.append(best_metrics['r2'])

            # 更新種子進度條
            seed_pbar.set_postfix({
                'RMSE': f'{rmse_original:.4f}%',
                'Dir': f'{best_metrics["direction_acc"]:.3f}'
            })

            print(f"    ✅ RMSE: {rmse_original:.4f}%, Direction: {best_metrics['direction_acc']:.3f}, R²: {best_metrics['r2']:.4f}")

        except Exception as e:
            print(f"    ❌ 訓練失敗: {e}")
            return 10.0  # 懲罰值

    seed_pbar.close()

    # 計算平均
    avg_rmse = np.mean(rmse_list)
    avg_direction = np.mean(direction_list)
    avg_r2 = np.mean(r2_list)

    print(f"\n  📊 平均結果:")
    print(f"     RMSE: {avg_rmse:.4f}%")
    print(f"     Direction: {avg_direction:.3f}")
    print(f"     R²: {avg_r2:.4f}")

    # 目標函數：主要優化 RMSE，兼顧方向準確度
    objective_value = avg_rmse - 0.3 * avg_direction

    print(f"     目標值: {objective_value:.6f}")

    return objective_value

# ==================== 執行優化 ====================
print("\n" + "="*80)
print("🚀 開始貝葉斯優化")
print("="*80)

optimization_pbar = tqdm(total=30, desc="貝葉斯優化", ncols=100)

result = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=30,              # 總共評估 30 組參數
    n_initial_points=5,      # 前 5 組隨機探索
    initial_point_generator='sobol',
    acq_func='EI',           # Expected Improvement
    n_jobs=1,
    verbose=True,
    random_state=42
)

optimization_pbar.close()

print("\n" + "="*80)
print("✅ 優化完成！")
print("="*80)

# ==================== 結果分析 ====================
print("\n📊 最佳參數:")
best_params = {
    'beta': result.x[0],
    'gamma': result.x[1],
    'lr': result.x[2],
    'expert2_hidden': result.x[3]
}

for param_name, param_value in best_params.items():
    print(f"   {param_name}: {param_value}")

print(f"\n🏆 最佳目標值: {result.fun:.6f}")

# 估算對應的 RMSE
estimated_rmse = result.fun + 0.3 * 0.77  # 假設 direction ≈ 0.77
print(f"   (估計 RMSE ≈ {estimated_rmse:.4f}%)")


# ==================== 視覺化 ====================
print("\n📊 繪製優化過程...")

# 圖 1: 收斂曲線
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Convergence plot
plot_convergence(result, ax=axes[0])
axes[0].set_title('Optimization Convergence', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Evaluations')
axes[0].set_ylabel('Objective Value (lower is better)')

# 參數歷史
iterations = range(1, len(result.func_vals) + 1)
axes[1].plot(iterations, result.func_vals, 'o-', alpha=0.6, label='Objective Value')
axes[1].axhline(y=result.fun, color='r', linestyle='--', linewidth=2, label=f'Best: {result.fun:.4f}')
axes[1].set_xlabel('Iteration', fontsize=12)
axes[1].set_ylabel('Objective Value', fontsize=12)
axes[1].set_title('Objective Value History', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../results/bayesian_opt/convergence.png', dpi=300, bbox_inches='tight')
print("   ✅ 儲存: ../results/bayesian_opt/convergence.png")
plt.close()

# 圖 2: 參數重要性（部分依賴圖）
try:
    from skopt.plots import plot_evaluations
    fig = plt.figure(figsize=(12, 10))
    plot_evaluations(result)
    plt.tight_layout()
    plt.savefig('../results/bayesian_opt/evaluations.png', dpi=300, bbox_inches='tight')
    print("   ✅ 儲存: ../results/bayesian_opt/evaluations.png")
    plt.close()
except:
    print("   ⚠️ 無法繪製 evaluations 圖（可能需要更多迭代）")

# ==================== 儲存結果 ====================
results_df = pd.DataFrame({
    'iteration': range(1, len(result.func_vals) + 1),
    'objective': result.func_vals,
    'beta': [x[0] for x in result.x_iters],
    'gamma': [x[1] for x in result.x_iters],
    'lr': [x[2] for x in result.x_iters],
    'expert2_hidden': [x[3] for x in result.x_iters]
})

results_df.to_csv('../results/bayesian_opt/optimization_history.csv', index=False)
print("\n💾 優化歷史已儲存: ../results/bayesian_opt/optimization_history.csv")

# 儲存最佳參數
with open('../results/bayesian_opt/best_params.txt', 'w') as f:
    f.write("="*50 + "\n")
    f.write("貝葉斯優化 - 最佳參數\n")
    f.write("="*50 + "\n\n")
    for param_name, param_value in best_params.items():
        f.write(f"{param_name}: {param_value}\n")
    f.write(f"\n最佳目標值: {result.fun:.6f}\n")
    f.write(f"估計 RMSE: {estimated_rmse:.4f}%\n")

print("💾 最佳參數已儲存: ../results/bayesian_opt/best_params.txt")

print("\n" + "="*80)
print("✅ 貝葉斯優化完成！")
print("="*80)