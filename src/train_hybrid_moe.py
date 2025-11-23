#!/usr/bin/env python
"""
Training script for HybridMODWTMoE model.

Usage:
    python train_hybrid_moe.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, 'src')

from main import (
    prepare_modwt_data, HybridMODWTMoE, HuberLoss, evaluate,
    plot_training_history, plot_predictions, plot_gating_weights,
    plot_attention_maps, save_results_to_csv
)

# ==================== Configuration ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print(f"[Setup] Device: {DEVICE}")


# ==================== Training Function for HybridMODWTMoE ====================
def train_hybrid_moe(train_loader, test_loader, num_epochs=100,
                     lr=0.001, delta=1.0, device=DEVICE):
    """
    Train HybridMODWTMoE model using Two-Stage Curriculum Learning.

    Stage 1 (First 50% of Epochs): Train LSTM baseline only, freeze MoE parameters
    Stage 2 (Last 50% of Epochs): Unfreeze all parameters, joint training

    Args:
        train_loader: Training DataLoader with 'raw_input', 'expert1/2/3', 'target'
        test_loader: Test DataLoader
        num_epochs: Number of training epochs
        lr: Learning rate
        delta: Huber loss delta parameter
        device: Device to train on

    Returns:
        model: Trained model
        history: Training history dict
        best_epoch: Best epoch number
    """

    print("[Training] HybridMODWTMoE Model with Two-Stage Curriculum Learning")

    model = HybridMODWTMoE().to(device)
    criterion = HuberLoss(delta=1.0)
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_rmse': [],
        'train_mae': [],
        'epochs': [],
        'base_pred_mean': [],
        'moe_pred_mean': [],
        'branch_weight': [],
        'stage': []
    }

    best_train_loss = float('inf')
    best_epoch = 1
    best_model_state = model.state_dict().copy()

    stage1_epochs = num_epochs // 2
    stage2_epochs = num_epochs - stage1_epochs

    # ==================== STAGE 1: LSTM Baseline Only ====================
    print(f"\n[Stage 1] Training LSTM Baseline Only (Epochs 1-{stage1_epochs})")
    print("  Freezing MoE parameters: expert1, expert2, expert3, gating, branch_weight")

    # Freeze MoE parameters
    model.expert1.requires_grad_(False)
    model.expert2.requires_grad_(False)
    model.expert3.requires_grad_(False)
    model.gating.requires_grad_(False)
    model.branch_weight.requires_grad_(False)

    # Create optimizer with LSTM parameters only
    lstm_params = model.lstm_branch.parameters()
    optimizer = torch.optim.Adam(lstm_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    for epoch in range(stage1_epochs):
        # ==================== Training Phase ====================
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        base_preds = []
        moe_preds = []

        for batch in train_loader:
            # Extract inputs
            raw_input = batch['raw_input'].to(device)
            e1 = batch['expert1'].to(device)
            e2 = batch['expert2'].to(device)
            e3 = batch['expert3'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()

            # Forward pass through HybridMODWTMoE
            output, base_pred, moe_pred, weights, expert_preds, _ = model(
                raw_input, e1, e2, e3
            )

            # Stage 1 Loss: Only LSTM baseline
            
            # [核心修改] Auxiliary Loss Strategy
            # 1. 計算最終輸出的 Loss
            main_loss = criterion(output, target)

            # 2. 單獨計算 LSTM Branch 的 Loss (強迫它學好)
            aux_loss = criterion(base_pred, target)

            # 3. 總 Loss = Main + 0.5 * Aux
            loss = main_loss + 0.5 * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.lstm_branch.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            train_losses.append(loss.item())
            train_preds.append(base_pred.detach().cpu().numpy())  # Use base_pred for Stage 1
            train_targets.append(target.cpu().numpy())
            base_preds.append(base_pred.detach().cpu().numpy())
            moe_preds.append(moe_pred.detach().cpu().numpy())

        # Compute training metrics
        avg_train_loss = np.mean(train_losses)
        train_preds_all = np.concatenate(train_preds, axis=0)
        train_targets_all = np.concatenate(train_targets, axis=0)
        base_preds_all = np.concatenate(base_preds, axis=0)
        moe_preds_all = np.concatenate(moe_preds, axis=0)

        train_rmse = np.sqrt(mean_squared_error(train_targets_all, train_preds_all))
        train_mae = mean_absolute_error(train_targets_all, train_preds_all)

        # ==================== Test Phase (Monitoring Only) ====================
        model.eval()
        test_losses = []

        with torch.no_grad():
            for batch in test_loader:
                raw_input = batch['raw_input'].to(device)
                e1 = batch['expert1'].to(device)
                e2 = batch['expert2'].to(device)
                e3 = batch['expert3'].to(device)
                target = batch['target'].to(device)

                output, base_pred, _, _, _, _ = model(raw_input, e1, e2, e3)
                # Stage 1: Evaluate LSTM baseline
                loss = criterion(base_pred, target)
                test_losses.append(loss.item())

        avg_test_loss = np.mean(test_losses)

        scheduler.step(avg_train_loss)

        # ==================== Track History ====================
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['train_rmse'].append(train_rmse)
        history['train_mae'].append(train_mae)
        history['epochs'].append(epoch + 1)
        history['base_pred_mean'].append(base_preds_all.mean())
        history['moe_pred_mean'].append(moe_preds_all.mean())
        history['branch_weight'].append(model.branch_weight.item())
        history['stage'].append(1)

        # ==================== Print Progress ====================
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{stage1_epochs} | Loss: {avg_train_loss:.4f} | RMSE: {train_rmse:.4f} | "
                  f"Base: {base_preds_all.mean():.4f} | MoE: {moe_preds_all.mean():.4f}")

        # ==================== Save Best Model ====================
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()

    # ==================== STAGE 2: Joint Training ====================
    print(f"\n[Stage 2] Joint Training - MoE Refinement (Epochs {stage1_epochs+1}-{num_epochs})")
    print("  Unfreezing all parameters for refinement training")

    # Unfreeze all parameters
    model.expert1.requires_grad_(True)
    model.expert2.requires_grad_(True)
    model.expert3.requires_grad_(True)
    model.gating.requires_grad_(True)
    model.branch_weight.requires_grad_(True)

    # Create new optimizer with all parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    for epoch in range(stage2_epochs):
        # ==================== Training Phase ====================
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        base_preds = []
        moe_preds = []

        for batch in train_loader:
            # Extract inputs
            raw_input = batch['raw_input'].to(device)
            e1 = batch['expert1'].to(device)
            e2 = batch['expert2'].to(device)
            e3 = batch['expert3'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()

            # Forward pass through HybridMODWTMoE
            output, base_pred, moe_pred, weights, expert_preds, _ = model(
                raw_input, e1, e2, e3
            )

            # Stage 2 Loss: Full hybrid model
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            train_losses.append(loss.item())
            train_preds.append(output.detach().cpu().numpy())  # Use full output for Stage 2
            train_targets.append(target.cpu().numpy())
            base_preds.append(base_pred.detach().cpu().numpy())
            moe_preds.append(moe_pred.detach().cpu().numpy())

        # Compute training metrics
        avg_train_loss = np.mean(train_losses)
        train_preds_all = np.concatenate(train_preds, axis=0)
        train_targets_all = np.concatenate(train_targets, axis=0)
        base_preds_all = np.concatenate(base_preds, axis=0)
        moe_preds_all = np.concatenate(moe_preds, axis=0)

        train_rmse = np.sqrt(mean_squared_error(train_targets_all, train_preds_all))
        train_mae = mean_absolute_error(train_targets_all, train_preds_all)

        # ==================== Test Phase (Monitoring Only) ====================
        model.eval()
        test_losses = []

        with torch.no_grad():
            for batch in test_loader:
                raw_input = batch['raw_input'].to(device)
                e1 = batch['expert1'].to(device)
                e2 = batch['expert2'].to(device)
                e3 = batch['expert3'].to(device)
                target = batch['target'].to(device)

                output, _, _, _, _, _ = model(raw_input, e1, e2, e3)
                # Stage 2: Evaluate full hybrid model
                loss = criterion(output, target)
                test_losses.append(loss.item())

        avg_test_loss = np.mean(test_losses)

        scheduler.step(avg_train_loss)

        # ==================== Track History ====================
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['train_rmse'].append(train_rmse)
        history['train_mae'].append(train_mae)
        history['epochs'].append(stage1_epochs + epoch + 1)
        history['base_pred_mean'].append(base_preds_all.mean())
        history['moe_pred_mean'].append(moe_preds_all.mean())
        history['branch_weight'].append(model.branch_weight.item())
        history['stage'].append(2)

        # ==================== Print Progress ====================
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {stage1_epochs + epoch+1:3d}/{num_epochs} | Loss: {avg_train_loss:.4f} | RMSE: {train_rmse:.4f} | "
                  f"Base: {base_preds_all.mean():.4f} | MoE: {moe_preds_all.mean():.4f}")

        # ==================== Save Best Model ====================
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = stage1_epochs + epoch + 1
            best_model_state = model.state_dict().copy()

    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nTraining complete. Best model from epoch {best_epoch} loaded.")

    return model, history, best_epoch


# ==================== Evaluation for Hybrid Model ====================
def evaluate_hybrid(model, data_loader, device):
    """
    Evaluate HybridMODWTMoE model and return both branches' predictions.

    Returns:
        metrics: Dict with RMSE, MAE, R2, direction_acc
        predictions: Final output predictions
        targets: True targets
        base_predictions: LSTM baseline predictions
        moe_predictions: MoE expert predictions
        expert_preds: Individual expert predictions
        gating_weights: Expert gating weights
        attention_weights: Attention maps from experts
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_base_preds = []
    all_moe_preds = []
    all_expert_preds = []
    all_gating_weights = []
    all_attention_weights = {'expert1': [], 'expert2': [], 'expert3': []}

    with torch.no_grad():
        for batch in data_loader:
            raw_input = batch['raw_input'].to(device)
            e1 = batch['expert1'].to(device)
            e2 = batch['expert2'].to(device)
            e3 = batch['expert3'].to(device)
            target = batch['target'].to(device)

            output, base_pred, moe_pred, weights, expert_preds, attention_weights = model(
                raw_input, e1, e2, e3
            )

            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_base_preds.append(base_pred.cpu().numpy())
            all_moe_preds.append(moe_pred.cpu().numpy())
            all_expert_preds.append(expert_preds.cpu().numpy())
            all_gating_weights.append(weights.cpu().numpy())

            all_attention_weights['expert1'].append(attention_weights['expert1'].cpu().numpy())
            all_attention_weights['expert2'].append(attention_weights['expert2'].cpu().numpy())
            all_attention_weights['expert3'].append(attention_weights['expert3'].cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_preds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()
    base_predictions = np.concatenate(all_base_preds, axis=0).flatten()
    moe_predictions = np.concatenate(all_moe_preds, axis=0).flatten()
    expert_preds_all = np.concatenate(all_expert_preds, axis=0)
    gating_weights = np.concatenate(all_gating_weights, axis=0)

    attention_weights_concat = {
        'expert1': np.concatenate(all_attention_weights['expert1'], axis=0),
        'expert2': np.concatenate(all_attention_weights['expert2'], axis=0),
        'expert3': np.concatenate(all_attention_weights['expert3'], axis=0)
    }

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    direction_true = np.sign(np.diff(targets))
    direction_pred = np.sign(np.diff(predictions))
    direction_acc = np.mean(direction_true == direction_pred)

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_acc': direction_acc
    }

    return (metrics, predictions, targets, base_predictions, moe_predictions,
            expert_preds_all, gating_weights, attention_weights_concat)


# ==================== Main Execution ====================
if __name__ == "__main__":
    os.makedirs('../results', exist_ok=True)

    print("[Loading Data]")
    df = pd.read_csv("../dataset/USD_TWD.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  {len(df)} days loaded\n")

    # ==================== Prepare Data ====================
    print("[Data Preparation]")
    train_loader, test_loader, scalers, _, energies = prepare_modwt_data(
        df,
        vol_window=7,
        lookback=30,
        forecast_horizon=1,
        wavelet='db2',
        level=4,
        train_ratio=0.80,
        batch_size=32,
        use_robust_scaler=False,
        buffer_size=200
    )

    # ==================== Train Hybrid Model ====================
    print()
    trained_model, training_history, best_epoch = train_hybrid_moe(
        train_loader,
        test_loader,
        num_epochs=80,
        lr=0.001,
        device=DEVICE
    )

    # ==================== Evaluate on Test Set ====================
    print("\n[Evaluation]")
    (test_metrics, test_preds, test_targets, test_base_preds, test_moe_preds,
     test_expert_preds, test_gating_weights, test_attention_weights) = evaluate_hybrid(
        trained_model, test_loader, DEVICE
    )

    # Inverse transform predictions
    target_scaler = scalers['target']
    volatility_mean = scalers['volatility_mean']

    test_preds_centered = target_scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
    test_targets_centered = target_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()
    test_base_preds_centered = target_scaler.inverse_transform(test_base_preds.reshape(-1, 1)).flatten()
    test_moe_preds_centered = target_scaler.inverse_transform(test_moe_preds.reshape(-1, 1)).flatten()

    test_preds_original = test_preds_centered + volatility_mean
    test_targets_original = test_targets_centered + volatility_mean
    test_base_preds_original = test_base_preds_centered + volatility_mean
    test_moe_preds_original = test_moe_preds_centered + volatility_mean

    # Compute metrics in original scale
    rmse_original = np.sqrt(mean_squared_error(test_targets_original, test_preds_original))
    mae_original = mean_absolute_error(test_targets_original, test_preds_original)
    r2_original = r2_score(test_targets_original, test_preds_original)
    rmse_base = np.sqrt(mean_squared_error(test_targets_original, test_base_preds_original))
    rmse_moe = np.sqrt(mean_squared_error(test_targets_original, test_moe_preds_original))

    print(f"[Test Set Performance]")
    print(f"  Combined:      RMSE: {rmse_original:.4f}% | MAE: {mae_original:.4f}% | R²: {r2_original:.4f}")
    print(f"  LSTM Baseline: RMSE: {rmse_base:.4f}%")
    print(f"  MoE Refinement: RMSE: {rmse_moe:.4f}%")
    print(f"  Direction Accuracy: {test_metrics['direction_acc']*100:.2f}%")

    # ==================== Visualizations ====================
    print("\n[Visualizations]")

    # Training history with branch analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    epochs = training_history['epochs']
    stages = training_history['stage']

    # Find stage transition point
    stage_transition = None
    for i in range(1, len(stages)):
        if stages[i] != stages[i-1]:
            stage_transition = epochs[i-1]
            break

    # Loss
    axes[0, 0].plot(epochs, training_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, training_history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    if stage_transition is not None:
        axes[0, 0].axvline(stage_transition, color='green', linestyle='--', linewidth=2,
                          label=f'Stage Transition (Epoch {stage_transition})', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss Over Time (Two-Stage Curriculum)', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(alpha=0.3)

    # RMSE
    axes[0, 1].plot(epochs, training_history['train_rmse'], 'g-', linewidth=2)
    if stage_transition is not None:
        axes[0, 1].axvline(stage_transition, color='green', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('RMSE (%)', fontsize=12)
    axes[0, 1].set_title('Training RMSE Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # Branch contributions
    axes[1, 0].plot(epochs, training_history['base_pred_mean'], 'b-', label='LSTM Base Mean', linewidth=2)
    axes[1, 0].plot(epochs, training_history['moe_pred_mean'], 'orange', label='MoE Refinement Mean', linewidth=2)
    axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    if stage_transition is not None:
        axes[1, 0].axvline(stage_transition, color='green', linestyle='--', linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Mean Prediction', fontsize=12)
    axes[1, 0].set_title('Branch Contribution Over Time', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(alpha=0.3)

    # Branch weight
    axes[1, 1].plot(epochs, training_history['branch_weight'], 'purple', linewidth=2)
    if stage_transition is not None:
        axes[1, 1].axvline(stage_transition, color='green', linestyle='--', linewidth=2,
                          label='Stage Transition', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Branch Weight', fontsize=12)
    axes[1, 1].set_title('Learned Weight for MoE Branch (Starts at 0.0)', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/hybrid_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: ../results/hybrid_training_history.png")

    # Predictions comparison
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    time_idx = np.arange(len(test_targets_original))

    # Combined predictions
    axes[0].plot(time_idx, test_targets_original, 'k-', alpha=0.6, linewidth=1.5, label='Actual')
    axes[0].plot(time_idx, test_preds_original, 'b-', alpha=0.7, linewidth=1.5, label='Hybrid Prediction')
    axes[0].fill_between(time_idx, test_targets_original, test_preds_original, alpha=0.2)
    axes[0].set_ylabel('Volatility (%)', fontsize=12)
    axes[0].set_title('Hybrid Model: Actual vs Predicted Volatility', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Branch contributions
    axes[1].plot(time_idx, test_base_preds_original - test_targets_original, 'b-', alpha=0.7, linewidth=1.5, label='LSTM Baseline Error')
    axes[1].plot(time_idx, test_preds_original - test_targets_original, 'orange', alpha=0.7, linewidth=1.5, label='Hybrid Model Error')
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_ylabel('Prediction Error (%)', fontsize=12)
    axes[1].set_title('Error Comparison: LSTM vs Hybrid', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/hybrid_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: ../results/hybrid_predictions.png")

    # Attention maps
    plot_attention_maps(test_attention_weights, '../results/hybrid_attention_maps.png')

    # Gating weights
    plot_gating_weights(test_gating_weights, '../results/hybrid_gating_weights.png')

    # Save results
    test_results_df = save_results_to_csv(
        test_targets, test_preds, test_gating_weights,
        test_expert_preds, scalers, '../results/hybrid_test_results.csv'
    )

    # ==================== Summary ====================
    print(f"\n[Summary]")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Test RMSE: {rmse_original:.4f}%")
    print(f"  Test MAE: {mae_original:.4f}%")
    print(f"  Test R²: {r2_original:.4f}")
    print(f"  Direction Accuracy: {test_metrics['direction_acc']*100:.2f}%")
    print(f"  Improvement over LSTM: {(rmse_base - rmse_original):.4f}% ({(rmse_base - rmse_original)/rmse_base*100:.1f}%)")
    print(f"\n  Results saved to ../results/")
