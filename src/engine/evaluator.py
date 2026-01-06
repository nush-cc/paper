import torch
import numpy as np
from src.utils.metrics import get_metrics


def evaluate_model(model, test_loader, scaler, device, horizon):
    """
    通用評估函式 (Unified Evaluator)

    自動適應兩種模式：
    1. 詳細比較模式 (For Main): 當模型擁有 CNN 增強模組時，顯示 Base vs Enhanced 的差異。
    2. 簡潔模式 (For Ablation): 當模型只有基礎結構 (如 w/o CNN) 時，只顯示最終結果。
    """
    model.eval()

    preds_base = []
    preds_final = []
    targets_all = []
    last_knowns_all = []

    # --- 1. 自動偵測模型結構 (決定是否為混合模型) ---
    has_cnn = hasattr(model, 'cnn_trend') and model.cnn_trend is not None
    has_decomp = hasattr(model, 'decomp') and model.decomp is not None

    # 提取權重 (如果有 Gate 參數的話)
    trend_w = 0.0
    seas_w = 0.0
    if hasattr(model, 'trend_gate') and model.trend_gate is not None:
        trend_w = torch.tanh(model.trend_gate).item()
    if hasattr(model, 'seasonal_gate') and model.seasonal_gate is not None:
        seas_w = torch.tanh(model.seasonal_gate).item()

    # --- 2. 推論迴圈 ---
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch['raw_input'].to(device), batch['target'].to(device)
            B = x.shape[0]

            # A. 序列分解 (兼容 w/o Decomposition 實驗)
            if has_decomp:
                seasonal_part, trend_part = model.decomp(x)
            else:
                # 若無分解層，視原始輸入為 Trend，Seasonal 為 0
                seasonal_part = torch.zeros_like(x)
                trend_part = x

                # B. Linear Backbone (Base Model)
            trend_out_linear = model.linear_trend(trend_part.reshape(B, -1))
            seasonal_out_linear = model.linear_seasonal(seasonal_part.reshape(B, -1))

            last_val = x[:, -1, 0:1]
            pred_base_val = last_val + trend_out_linear + seasonal_out_linear

            # C. CNN Booster (兼容 w/o CNN 實驗)
            if has_cnn:
                trend_out_cnn = model.cnn_trend(trend_part)
                seasonal_out_cnn = model.cnn_seasonal(seasonal_part)
                correction = (trend_w * trend_out_cnn) + (seas_w * seasonal_out_cnn)
                pred_final_val = pred_base_val + correction
            else:
                # 若無 CNN，最終預測等於 Base 預測
                pred_final_val = pred_base_val

            # D. 收集數據
            preds_base.append(pred_base_val.cpu().numpy())
            preds_final.append(pred_final_val.cpu().numpy())
            targets_all.append(y.cpu().numpy())
            last_knowns_all.append(last_val.cpu().numpy())

    # --- 3. 反正規化 (Inverse Transform) ---
    def inv_seq(data_list):
        data_arr = np.vstack(data_list)
        return scaler.inverse_transform(data_arr.reshape(-1, 1)).reshape(data_arr.shape)

    def inv_anchor(data_list):
        return scaler.inverse_transform(np.vstack(data_list))

    y_true = inv_seq(targets_all)
    y_last = inv_anchor(last_knowns_all)
    y_base = inv_seq(preds_base)
    y_final = inv_seq(preds_final)

    # --- 4. 計算指標 ---
    # 這裡計算兩組：Base 和 Final (如果是純 Linear 模型，兩者數值會一樣)
    metrics_base = get_metrics(y_true, y_base, y_last)
    metrics_final = get_metrics(y_true, y_final, y_last)

    # 解包 Final Metrics (用於回傳)
    r2_f, rmse_f, acc_f, step_f, h_acc_f = metrics_final

    # --- 5. 智慧列印報告 (Smart Logging) ---
    print("\n" + "=" * 95)

    if has_cnn:
        # === 模式 A: 詳細比較 (適用於 Main Notebook) ===
        r2_b, rmse_b, acc_b, _, h_acc_b = metrics_base

        print(f" FINAL MODEL EVALUATION (Horizon={horizon}): Hybrid Architecture Analysis")
        print(f" (Gating Weights -> Trend: {trend_w:.4f} | Seasonal: {seas_w:.4f})")
        print("=" * 95)
        print(f"{'Metric':<20} | {'Linear Base':<15} | {'Enhanced':<15} | {'Improvement':<15}")
        print("-" * 95)

        # R2
        diff_r2 = r2_f - r2_b
        mark_r2 = '✅' if diff_r2 > 0.0001 else ('❌' if diff_r2 < -0.0001 else '➖')
        print(f"{'R2 Score':<20} | {r2_b:<15.4f} | {r2_f:<15.4f} | {diff_r2:+.4f} {mark_r2}")

        # RMSE
        diff_rmse = rmse_f - rmse_b
        mark_rmse = '✅' if diff_rmse < -0.0001 else ('❌' if diff_rmse > 0.0001 else '➖')
        print(f"{'RMSE':<20} | {rmse_b:<15.4f} | {rmse_f:<15.4f} | {diff_rmse:+.4f} {mark_rmse}")

        # Acc
        diff_acc = acc_f - acc_b
        mark_acc = '✅' if diff_acc > 0.0001 else ('❌' if diff_acc < -0.0001 else '➖')
        print(f"{'Avg Accuracy':<20} | {acc_b:<15.4f} | {acc_f:<15.4f} | {diff_acc:+.4f} {mark_acc}")

    else:
        # === 模式 B: 簡潔報告 (適用於 Ablation / Baseline) ===
        # 不印比較表，只印最終分數，看起來更清爽
        model_type = "Baseline (No-CNN)" if not has_cnn else "Ablation Variant"
        if not has_decomp: model_type = "No-Decomposition Variant"

        print(f" MODEL EVALUATION (Horizon={horizon}): {model_type}")
        print("=" * 95)
        print(f"{'Metric':<25} | {'Score':<15}")
        print("-" * 95)
        print(f"{'R2 Score':<25} | {r2_f:.4f}")
        print(f"{'RMSE':<25} | {rmse_f:.4f}")
        print(f"{'Avg Accuracy':<25} | {acc_f:.4f}")
        print(f"{'High Volatility Acc':<25} | {h_acc_f:.4f}")

    print("=" * 95)

    # --- 6. 回傳結果 (Return Dict) ---
    # 這對於消融實驗記錄數據非常重要
    return {
        "MSE": rmse_f ** 2,
        "RMSE": rmse_f,
        "MAE": 0.0,  # 如果你需要 MAE，請在 get_metrics 裡補算
        "R2": r2_f,
        "Acc": acc_f,
        "High_Vol_Acc": h_acc_f
    }
