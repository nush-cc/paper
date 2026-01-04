import torch
import numpy as np

from src.models.network import EnhancedDLinear
from src.utils.metrics import HybridDirectionalLoss


def train_v11(train_loader, test_loader, device, horizon, num_epochs=120, lr=0.001):
    model = EnhancedDLinear(seq_len=30, pred_len=horizon, input_channels=2).to(device)
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
            tr_w = torch.tanh(model.trend_gate).item()
            se_w = torch.tanh(model.seasonal_gate).item()
            print(f"  Epoch {epoch + 1} | Loss: {np.mean(losses):.4f} | Trend W: {tr_w:.3f} | Seas W: {se_w:.3f}")

    return model
