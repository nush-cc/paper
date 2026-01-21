import numpy as np
import torch

import src.utils.metrics as metrics


def evaluate_model(model, test_loader, device, horizon, scaler=None):
    model.eval()

    preds_base = []
    preds_final = []
    targets_all = []
    last_knowns_all = []

    has_cnn = getattr(model, 'use_seasonal_cnn', False)

    with torch.no_grad():
        for batch in test_loader:
            x = batch['raw_input'].to(device)
            y = batch['target'].to(device)

            pred_final, pred_base, _ = model(x)
            last_val = x[:, -1, 0:1]

            preds_final.append(pred_final.cpu().numpy())
            preds_base.append(pred_base.cpu().numpy())
            targets_all.append(y.cpu().numpy())
            last_knowns_all.append(last_val.cpu().numpy())

    y_true = np.vstack(targets_all)
    y_final = np.vstack(preds_final)
    y_base = np.vstack(preds_base)
    y_last = np.vstack(last_knowns_all)

    if scaler is not None:
        vol_std = scaler.scale_[0]
        vol_mean = scaler.mean_[0]
        
        y_true = y_true * vol_std + vol_mean
        y_final = y_final * vol_std + vol_mean
        y_base = y_base * vol_std + vol_mean
        y_last = y_last * vol_std + vol_mean

    # --- 計算指標 ---
    metrics_base = metrics.get_metrics(y_true, y_base, y_last)
    metrics_final = metrics.get_metrics(y_true, y_final, y_last)
    
    # 新增：計算兩者的 Sharpe Ratio
    sharpe_b = metrics.get_sharpe_ratio(y_true, y_base, y_last)
    sharpe_f = metrics.get_sharpe_ratio(y_true, y_final, y_last)

    r2_f, rmse_f, acc_f, step_acc_f, h_acc_f = metrics_final
    r2_b, rmse_b, acc_b, step_acc_b, h_acc_b = metrics_base

    # --- 輸出報表 ---
    print("\n" + "=" * 95)

    if has_cnn:
        print(f" FINAL MODEL EVALUATION (Horizon={horizon}): Ablation Study")
        print("=" * 95)
        print(f"{'Metric':<20} | {'Linear Base':<15} | {'Base + CNN':<15} | {'Improvement':<15}")
        print("-" * 95)

        # RMSE
        print(f"{'RMSE':<20} | {rmse_b:<15.4f} | {rmse_f:<15.4f} | {rmse_f - rmse_b:+.4f}")

        # R2 Score
        print(f"{'R2 Score':<20} | {r2_b:<15.4f} | {r2_f:<15.4f} | {r2_f - r2_b:+.4f}")

        # Accuracy
        print(f"{'Avg Accuracy':<20} | {acc_b:<15.4f} | {acc_f:<15.4f} | {acc_f - acc_b:+.4f}")
        print(f"{'High Vol Accuracy':<20} | {h_acc_b:<15.4f} | {h_acc_f:<15.4f} | {h_acc_f - h_acc_b:+.4f}")

        # 新增：Sharpe Ratio 報表輸出
        print(f"{'Sharpe Ratio':<20} | {sharpe_b:<15.4f} | {sharpe_f:<15.4f} | {sharpe_f - sharpe_b:+.4f}")

    else:
        print(f" MODEL EVALUATION (Horizon={horizon}): Pure Linear Model")
        print("-" * 60)
        print(f" RMSE: {rmse_f:.4f} | R2: {r2_f:.4f} | Sharpe: {sharpe_f:.4f}")

    print("=" * 95)

    return {
        "RMSE_Final": rmse_f,
        "RMSE_Base": rmse_b,
        "R2_Final": r2_f,
        "Avg_Acc": acc_f,
        "High_Vol_Acc": h_acc_f,
        "Sharpe_Final": sharpe_f,  # 新增回傳
        "Sharpe_Base": sharpe_b    # 新增回傳
    }
