import numpy as np
import torch

import src.utils.metrics as metrics


def evaluate_model(model, test_loader, device, horizon, scaler=None):
    model.eval()

    preds_base = []
    preds_final = []
    targets_all = []    # 真實波動率
    returns_all = []    # 真實報酬率
    last_knowns_all = [] # 最後一筆已知波動率
    
    has_cnn = getattr(model, 'use_seasonal_cnn', False)

    with torch.no_grad():
        for batch in test_loader:
            x = batch['raw_input'].to(device)
            y = batch['target'].to(device)
            r = batch['return'].to(device) # 取得真實報酬率

            pred_final, pred_base, _ = model(x)
            
            # 取得最後一筆輸入的波動率 (用於計算 Acc 方向)
            last_val = x[:, -1, 0:1] 

            # 我們只取 Horizon 的第一步 (t+1) 進行評估
            preds_final.append(pred_final[:, 0:1].cpu().numpy())
            preds_base.append(pred_base[:, 0:1].cpu().numpy())
            targets_all.append(y[:, 0:1].cpu().numpy())
            last_knowns_all.append(last_val.cpu().numpy())
            returns_all.append(r[:, 0].cpu().numpy()) # Returns

    # 轉成 numpy array
    y_true_vol = np.vstack(targets_all)      # [N, 1]
    y_pred_final_vol = np.vstack(preds_final)# [N, 1]
    y_pred_base_vol = np.vstack(preds_base)  # [N, 1]
    y_last_vol = np.vstack(last_knowns_all)  # [N, 1]
    y_returns = np.concatenate(returns_all)  # [N]

    # 反標準化 (還原成真實波動率數值)
    if scaler is not None:
        vol_std = scaler.scale_[0]
        vol_mean = scaler.mean_[0]
        
        y_true_vol = y_true_vol * vol_std + vol_mean
        y_pred_final_vol = y_pred_final_vol * vol_std + vol_mean
        y_pred_base_vol = y_pred_base_vol * vol_std + vol_mean
        y_last_vol = y_last_vol * vol_std + vol_mean

    # === [部分 1] 統計指標評估 (RMSE, R2, Accuracy) ===
    # 這裡呼叫你原本的 get_metrics 來確保這些指標都在
    
    # 1. Final Model Metrics
    metrics_final = metrics.get_metrics(y_true_vol, y_pred_final_vol, y_last_vol)
    r2_f, rmse_f, acc_f, step_acc_f, h_acc_f = metrics_final

    # 2. Base Model Metrics
    metrics_base = metrics.get_metrics(y_true_vol, y_pred_base_vol, y_last_vol)
    r2_b, rmse_b, acc_b, step_acc_b, h_acc_b = metrics_base


    # === [部分 2] 金融回測指標 (Sharpe, Returns) ===
    # 策略：波動率擇時 (Risk Control)
    # 門檻：使用 Final Model 預測值的 80 分位數 (避開高風險)
    pred_vol_1d = y_pred_final_vol.flatten()
    threshold = np.percentile(pred_vol_1d, 80)
    
    # 訊號：預測波動率 < 門檻 => 持有 (1.0), 否則 => 空手 (0.0)
    signals = (pred_vol_1d < threshold).astype(float)
    
    # 策略損益 = 訊號 * 真實報酬
    strategy_rets = signals * y_returns
    
    # 計算金融指標
    fin_metrics = metrics.calculate_financial_metrics(strategy_rets, y_returns)
    
    # --- 輸出綜合報表 ---
    print("\n" + "=" * 95)
    print(" FINAL MODEL EVALUATION (Horizon=1) | Ablation & Backtest")
    print("=" * 95)
    
    # 1. Ablation Study Table (統計指標)
    if has_cnn:
        print(f"{'Metric':<20} | {'Linear Base':<15} | {'Base + CNN':<15} | {'Improvement':<15}")
        print("-" * 95)
        print(f"{'RMSE':<20} | {rmse_b:<15.4f} | {rmse_f:<15.4f} | {rmse_f - rmse_b:+.4f}")
        print(f"{'R2 Score':<20} | {r2_b:<15.4f} | {r2_f:<15.4f} | {r2_f - r2_b:+.4f}")
        print(f"{'Avg Accuracy':<20} | {acc_b:<15.4f} | {acc_f:<15.4f} | {acc_f - acc_b:+.4f}")
        print(f"{'High Vol Acc':<20} | {h_acc_b:<15.4f} | {h_acc_f:<15.4f} | {h_acc_f - h_acc_b:+.4f}")
    else:
        print(f" RMSE: {rmse_f:.4f} | R2: {r2_f:.4f} | Acc: {acc_f:.4f}")

    print("-" * 95)
    
    # 2. Financial Backtest Table (策略回測)
    print(f" [Risk Control Strategy] (Threshold: {threshold:.2f}%)")
    print(f"{'Metric':<25} | {'Buy & Hold':<15} | {'Your Strategy':<15} | {'Diff':<15}")
    print("-" * 95)
    
    # Sharpe Ratio
    diff_sharpe = fin_metrics['Strategy_Sharpe'] - fin_metrics['Benchmark_Sharpe']
    print(f"{'Sharpe Ratio':<25} | {fin_metrics['Benchmark_Sharpe']:<15.4f} | {fin_metrics['Strategy_Sharpe']:<15.4f} | {diff_sharpe:+.4f}")
    
    # Cumulative Return
    diff_cum = fin_metrics['Strategy_CumRet'] - fin_metrics['Benchmark_CumRet']
    print(f"{'Cumulative Return':<25} | {fin_metrics['Benchmark_CumRet']*100:<14.2f}% | {fin_metrics['Strategy_CumRet']*100:<14.2f}% | {diff_cum*100:+.2f}%")
    
    # Max Drawdown
    print(f"{'Max Drawdown':<25} | {'--':<15} | {fin_metrics['Max_Drawdown']*100:<14.2f}% |")

    print("=" * 95)

    return {
        # 原本的統計指標回傳 (保留)
        "RMSE_Final": rmse_f,
        "RMSE_Base": rmse_b,
        "R2_Final": r2_f,
        "Avg_Acc": acc_f,
        "High_Vol_Acc": h_acc_f,
        
        # 新增的金融指標回傳
        "Sharpe_Strategy": fin_metrics['Strategy_Sharpe'],
        "Sharpe_Benchmark": fin_metrics['Benchmark_Sharpe'],
        "Cum_Ret_Strategy": fin_metrics['Strategy_CumRet']
    }