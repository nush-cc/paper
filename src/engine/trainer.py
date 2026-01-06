import torch
import numpy as np

from src.models.network import EnhancedDLinear
from src.utils.metrics import HybridDirectionalLoss


def train_v11(train_loader, test_loader, device, horizon, num_epochs=120, lr=0.001,
              cnnExpert_KernelSize=5, seriesDecomposition_KernelSize=15,
              model_hyperparams=None
              ):
    if model_hyperparams is None:
        model_hyperparams = {}

    model = EnhancedDLinear(seq_len=30, pred_len=horizon, input_channels=2,
                            seriesDecomposition_KernelSize=seriesDecomposition_KernelSize,
                            cnnExpert_KernelSize=cnnExpert_KernelSize,
                            **model_hyperparams
                            ).to(device)
    criterion = HybridDirectionalLoss(direction_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print("\n[Training] Enhanced DLinear...")
    for epoch in range(num_epochs):
        model.train()
        losses = []
        for batch in train_loader:
            x, y = batch['raw_input'].to(device), batch['target'].to(device)
            optimizer.zero_grad()
            prev_val = x[:, -1, 0:1]
            out, _, _ = model(x)
            loss = criterion(out, y, prev_val)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            # === [修正開始] 安全地讀取權重 ===
            # 先預設為 0 (或是 N/A)
            tr_w_val = 0.0
            se_w_val = 0.0

            # 檢查是否存在且不為 None
            # 注意：有些版本的 Pytorch Parameter 可以直接用 if checking，但最保險是檢查 None
            if hasattr(model, 'trend_gate') and model.trend_gate is not None:
                tr_w_val = torch.tanh(model.trend_gate).item()

            if hasattr(model, 'seasonal_gate') and model.seasonal_gate is not None:
                se_w_val = torch.tanh(model.seasonal_gate).item()
            # === [修正結束] ===

            print(
                f"  Epoch {epoch + 1} | Loss: {np.mean(losses):.4f} | Trend W: {tr_w_val:.3f} | Seas W: {se_w_val:.3f}")

    return model
