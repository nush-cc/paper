import torch
import numpy as np

from src.utils.metrics import get_metrics


def evaluate_model(model, test_loader, scaler, device, horizon):
    """
    執行推論、反正規化、計算指標並列印詳細比較報告。
    """
    model.eval()

    preds_dlinear = []
    preds_hybrid = []
    targets_all = []
    last_knowns_all = []

    # --- 1. 提取權重 (若模型有 gate 則提取，否則為 0) ---
    trend_w = 0.0
    seas_w = 0.0
    # 檢查屬性是否存在且不為 None (兼容有無 CNN 的情況)
    if hasattr(model, 'trend_gate') and model.trend_gate is not None:
        trend_w = torch.tanh(model.trend_gate).item()
    if hasattr(model, 'seasonal_gate') and model.seasonal_gate is not None:
        seas_w = torch.tanh(model.seasonal_gate).item()

    # --- 2. 推論迴圈 ---
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch['raw_input'].to(device), batch['target'].to(device)
            B = x.shape[0]

            # A. 手動執行 Forward Pass (為了拆解 Base vs Hybrid)
            seasonal_part, trend_part = model.decomp(x)

            # B. Linear 部分 (Base Model)
            trend_out_linear = model.linear_trend(trend_part.reshape(B, -1))
            seasonal_out_linear = model.linear_seasonal(seasonal_part.reshape(B, -1))

            last_val = x[:, -1, 0:1]

            # C. 計算 Base Prediction
            pred_base = last_val + trend_out_linear + seasonal_out_linear

            # D. 計算 Hybrid Prediction (如果有 CNN)
            # 檢查是否有 cnn_trend 屬性且不為 None
            if hasattr(model, 'cnn_trend') and model.cnn_trend is not None:
                trend_out_cnn = model.cnn_trend(trend_part)
                seasonal_out_cnn = model.cnn_seasonal(seasonal_part)
                correction = (trend_w * trend_out_cnn) + (seas_w * seasonal_out_cnn)
                pred_full = pred_base + correction
            else:
                pred_full = pred_base

            # E. 儲存結果
            preds_dlinear.append(pred_base.cpu().numpy())
            preds_hybrid.append(pred_full.cpu().numpy())
            targets_all.append(y.cpu().numpy())
            last_knowns_all.append(last_val.cpu().numpy())

    # --- 3. 反正規化 (Inverse Transform) ---
    def inv_seq(data_list):
        data_arr = np.vstack(data_list)
        N, H = data_arr.shape
        data_flat = data_arr.reshape(-1, 1)
        data_inv = scaler.inverse_transform(data_flat)
        return data_inv.reshape(N, H)

    def inv_anchor(data_list):
        data_arr = np.vstack(data_list)
        return scaler.inverse_transform(data_arr)

    y_true = inv_seq(targets_all)
    y_last = inv_anchor(last_knowns_all)
    y_dlinear = inv_seq(preds_dlinear)
    y_hybrid = inv_seq(preds_hybrid)

    # --- 4. 計算指標 (Calculate Metrics) ---
    r2_base, rmse_base, acc_base, step_base, h_acc_base = get_metrics(y_true, y_dlinear, y_last)
    r2_full, rmse_full, acc_full, step_full, h_acc_full = get_metrics(y_true, y_hybrid, y_last)

    # --- 5. 列印報告 (Print Report) ---
    print("\n" + "=" * 95)
    print(f" FINAL SEQUENCE FORECAST (Horizon={horizon}): Detailed Breakdown")
    print(f"   (Weights -> Trend: {trend_w:.4f} | Seasonal: {seas_w:.4f})")
    print("=" * 95)
    print(f"{'Metric':<20} | {'Pure DLinear':<15} | {'Enhanced':<15} | {'Improvement':<15}")
    print("-" * 95)

    # (1) R2 Score
    diff_r2 = r2_full - r2_base
    mark_r2 = '✅' if diff_r2 > 0.0001 else ('❌' if diff_r2 < -0.0001 else '➖')
    print(f"{'R2 Score (Overall)':<20} | {r2_base:<15.4f} | {r2_full:<15.4f} | {diff_r2:+.4f} {mark_r2}")

    # (2) RMSE
    diff_rmse = rmse_full - rmse_base
    mark_rmse = '✅' if diff_rmse < -0.0001 else ('❌' if diff_rmse > 0.0001 else '➖')
    print(f"{'RMSE (Overall)':<20} | {rmse_base:<15.4f} | {rmse_full:<15.4f} | {diff_rmse:+.4f} {mark_rmse}")

    print("-" * 95)
    print("Direction Accuracy (Win Rate)")

    # (3) Average Accuracy
    diff_acc = acc_full - acc_base
    mark_acc = '✅' if diff_acc > 0.0001 else ('❌' if diff_acc < -0.0001 else '➖')
    print(f"{'  Average (All Days)':<20} | {acc_base:<15.4f} | {acc_full:<15.4f} | {diff_acc:+.4f} {mark_acc}")

    # (4) Day-wise Breakdown
    for i in range(horizon):
        d_base = step_base[i]
        d_full = step_full[i]
        diff = d_full - d_base
        day_label = f"  Day {i + 1} (T+{i + 1})"
        mark_day = '✅' if diff > 0.0001 else ('❌' if diff < -0.0001 else '➖')
        print(f"{day_label:<20} | {d_base:<15.4f} | {d_full:<15.4f} | {diff:+.4f} {mark_day}")

    print("-" * 95)

    # (5) High Volatility Accuracy
    diff_h_acc = h_acc_full - h_acc_base
    mark_h = '✅' if diff_h_acc > 0.0001 else ('❌' if diff_h_acc < -0.0001 else '➖')
    print(f"{'High Volatility Acc':<20} | {h_acc_base:<15.4f} | {h_acc_full:<15.4f} | {diff_h_acc:+.4f} {mark_h}")

    print("=" * 95)
