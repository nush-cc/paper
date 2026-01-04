import torch
import torch.nn as nn

from src.models.layers import SeriesDecomp, CNNExpert


class EnhancedDLinear(nn.Module):
    def __init__(self, seq_len=30, pred_len=1, input_channels=2):
        super().__init__()

        # 1. Decomposition
        self.decomp = SeriesDecomp(kernel_size=5, input_channels=input_channels)

        # 2. Linear Backbone (DLinear)
        self.linear_trend = nn.Linear(seq_len * input_channels, pred_len)
        self.linear_seasonal = nn.Linear(seq_len * input_channels, pred_len)

        # 3. CNN Booster
        self.cnn_trend = CNNExpert(seq_len, pred_len, input_channels, hidden_dim=32)
        self.cnn_seasonal = CNNExpert(seq_len, pred_len, input_channels, hidden_dim=32)

        # 4. Static Weights (Learnable Scalars)
        # 給予一點點初始偏置，讓它容易學到負值 (阻尼)
        self.trend_gate = nn.Parameter(torch.tensor(-0.01))
        self.seasonal_gate = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        seasonal_part, trend_part = self.decomp(x)
        B, S, C = x.shape

        trend_flat = trend_part.reshape(B, -1)
        seasonal_flat = seasonal_part.reshape(B, -1)

        # Linear Parts
        trend_out_linear = self.linear_trend(trend_flat)
        seasonal_out_linear = self.linear_seasonal(seasonal_flat)

        # CNN Parts
        trend_out_cnn = self.cnn_trend(trend_part)
        seasonal_out_cnn = self.cnn_seasonal(seasonal_part)

        # Fusion
        trend_final = trend_out_linear + (torch.tanh(self.trend_gate) * trend_out_cnn)
        seasonal_final = seasonal_out_linear + (torch.tanh(self.seasonal_gate) * seasonal_out_cnn)

        output = x[:, -1, 0:1] + trend_final + seasonal_final

        # Return weights for logging
        return output, None, torch.stack([self.trend_gate, self.seasonal_gate])
