import torch
import numpy as np

from src.models.network import EnhancedDLinear
from src.utils.metrics import HybridDirectionalLoss


def train_v11(
    train_loader,
    val_loader,
    test_loader,
    device,
    horizon,
    seq_len=60,
    num_epochs=120,
    lr=0.001,
    trendCNNExpert_KernelSize=3,
    seasonalCNNExpert_KernelSize=7,
    seriesDecomposition_KernelSize=15,
    model_hyperparams=None,
):
    if model_hyperparams is None:
        model_hyperparams = {}

    model = EnhancedDLinear(
        seq_len=seq_len,
        pred_len=horizon,
        input_channels=2,
        seriesDecomposition_KernelSize=seriesDecomposition_KernelSize,
        trendCNNExpert_KernelSize=trendCNNExpert_KernelSize,
        seasonalCNNExpert_KernelSize=seasonalCNNExpert_KernelSize,
        **model_hyperparams,
    ).to(device)

    criterion = HybridDirectionalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    best_val_loss = float("inf")
    patience = 10  # 容忍 10 個 Epoch 不進步
    counter = 0

    print("\n[Training] Enhanced DLinear...")
    for epoch in range(num_epochs):
        # === 1. Training ===
        model.train()
        losses = []
        trend_gate_values = []
        seas_gate_values = []

        for batch in train_loader:
            x, y = batch["raw_input"].to(device), batch["target"].to(device)
            optimizer.zero_grad()
            prev_val = x[:, -1, 0:1]

            # 接收回傳的權重 (out, base, weights)
            out, _, gate_weights = model(x)

            loss = criterion(out, y, prev_val)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # 紀錄 Gate 權重 (如果有回傳)
            if gate_weights is not None:
                trend_gate_values.append(gate_weights[0].item())
                seas_gate_values.append(gate_weights[1].item())

        # === 2. Validation ===
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["raw_input"].to(device), batch["target"].to(device)
                prev_val = x[:, -1, 0:1]
                out, _, _ = model(x)
                loss = criterion(out, y, prev_val)
                val_losses.append(loss.item())

        train_loss = np.mean(losses)
        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)

        # 計算平均權重 (若無 CNN 則為 0)
        t_w = np.mean(trend_gate_values) if trend_gate_values else 0.0
        s_w = np.mean(seas_gate_values) if seas_gate_values else 0.0

        # === 3. Logging & Early Stopping ===
        # 印出當前狀態 (包含權重資訊)
        # if (epoch + 1) % 10 == 0 or epoch == 0:
        #     print(f"Epoch {epoch + 1:3d} | Train: {train_loss:.4f} | Val: {avg_val_loss:.4f} | Trend W: {t_w:.3f} | Seas W: {s_w:.3f}")

        print(
            f"Epoch {epoch + 1:3d} | Train: {train_loss:.4f} | Val: {avg_val_loss:.4f} | Trend W: {t_w:.3f} | Seas W: {s_w:.3f}"
        )

        # Checkpoint logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            # 儲存最佳模型參數
            torch.save(model.state_dict(), "best_model.pth")
            # print("  >>> New Best Model Saved!") # 可選擇是否要在每個變好時都印
        else:
            counter += 1
            if counter >= patience:
                print(
                    f"[Early Stopping] No improvement for {patience} epochs. Stopped at epoch {epoch + 1}."
                )
                break

    # === 4. Restore Best Model ===
    print(
        f"\n[Training Completed] Loading best model (Val Loss: {best_val_loss:.4f})..."
    )
    model.load_state_dict(torch.load("best_model.pth"))

    return model
