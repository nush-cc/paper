import torch
import numpy as np

import src.utils.metrics as metrics


def evaluate_model(model, test_loader, device, horizon):
    model.eval()

    preds_base = []
    preds_final = []
    targets_all = []
    last_knowns_all = []

    # 判斷是否為混合模型 (用於顯示不同報表)
    has_cnn = getattr(model, 'use_cnn', False)

    with torch.no_grad():
        for batch in test_loader:
            x = batch['raw_input'].to(device)
            y = batch['target'].to(device)  # 真實數值

            # 直接從模型取得兩組預測值
            pred_final, pred_base, _ = model(x)
            last_val = x[:, -1, 0:1]

            preds_final.append(pred_final.cpu().numpy())
            preds_base.append(pred_base.cpu().numpy())
            targets_all.append(y.cpu().numpy())
            last_knowns_all.append(last_val.cpu().numpy())

    # 轉成 numpy array
    y_true = np.vstack(targets_all)
    y_final = np.vstack(preds_final)
    y_base = np.vstack(preds_base)
    y_last = np.vstack(last_knowns_all)

    # 計算最後一筆數據作為 Naive 比較基準 (Optional)
    # 這裡假設你的 test_loader 有保留最後一筆 input，如果沒有可以傳入
    # 這裡先省略，專注於 Base vs Enhanced

    # --- 計算指標 ---
    metrics_base = metrics.get_metrics(y_true, y_base, y_last)
    metrics_final = metrics.get_metrics(y_true, y_final, y_last)

    r2_f, rmse_f, acc_f, step_acc_f, h_acc_f = metrics_final
    r2_b, rmse_b, acc_b, step_acc_b, _ = metrics_base

    # --- 輸出報表 ---
    print("\n" + "=" * 95)

    if has_cnn:
        print(f" FINAL MODEL EVALUATION (Horizon={horizon}): Ablation Study (RevIN Enabled)")
        print("=" * 95)
        print(f"{'Metric':<20} | {'Linear Base':<15} | {'Base + CNN':<15} | {'Improvement':<15}")
        print("-" * 95)

        # RMSE (越低越好)
        diff_rmse = rmse_f - rmse_b
        # 如果 Final 比 Base 低，代表有改善 (負數是好的)
        print(f"{'RMSE':<20} | {rmse_b:<15.4f} | {rmse_f:<15.4f} | {diff_rmse:+.4f}")

        # R2 (越高越好)
        diff_r2 = r2_f - r2_b
        print(f"{'R2 Score':<20} | {r2_b:<15.4f} | {r2_f:<15.4f} | {diff_r2:+.4f}")

        # Accuracy
        diff_acc = acc_f - acc_b
        print(f"{'Avg Accuracy':<20} | {acc_b:<15.4f} | {acc_f:<15.4f} | {diff_acc:+.4f}")

        diff_high_acc = h_acc_f - metrics_base[4]
        print(f"{'High Vol Accuracy':<20} | {metrics_base[4]:<15.4f} | {h_acc_f:<15.4f} | {diff_high_acc:+.4f}")

    else:
        print(f" MODEL EVALUATION (Horizon={horizon}): Pure Linear Model")
        print("-" * 60)
        print(f" RMSE: {rmse_f:.4f} | R2: {r2_f:.4f}")

    print("=" * 95)

    return {
        "RMSE_Final": rmse_f,
        "RMSE_Base": rmse_b,
        "R2_Final": r2_f,
        "Avg_Acc": acc_f,
        "High_Vol_Acc": h_acc_f
    }
