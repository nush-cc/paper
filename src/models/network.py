import torch
import torch.nn as nn
# 記得引入你的 SeriesDecomp 和 CNNExpert，假設它們在 layers.py
# 如果是在同一個檔案就不用 import
from src.models.layers import SeriesDecomp, CNNExpert


class EnhancedDLinear(nn.Module):
    def __init__(self, seq_len, pred_len, input_channels,
                 use_cnn=True,  # <--- 這些是 init 的參數
                 use_decomp=True,
                 hidden_dim=32):
        super().__init__()

        # 1. 把參數存起來 (這樣 forward 才讀得到)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_cnn = use_cnn
        self.use_decomp = use_decomp

        # ==========================================
        #  區域 A: 建立層 (Building Layers)
        #  這裡只負責 "定義" 模型有什麼零件
        # ==========================================

        # --- 分解層 ---
        if self.use_decomp:
            self.decomp = SeriesDecomp(kernel_size=15, input_channels=input_channels)
        else:
            self.decomp = None

        # --- Linear Backbone (一定會有) ---
        self.linear_trend = nn.Linear(seq_len * input_channels, pred_len)
        self.linear_seasonal = nn.Linear(seq_len * input_channels, pred_len)

        # --- CNN Booster (根據開關決定是否建立) ---
        if self.use_cnn:
            self.cnn_trend = CNNExpert(seq_len, pred_len, input_channels, hidden_dim)
            self.cnn_seasonal = CNNExpert(seq_len, pred_len, input_channels, hidden_dim)

            # Gating Weights
            self.trend_gate = nn.Parameter(torch.tensor(-0.01))
            self.seasonal_gate = nn.Parameter(torch.tensor(0.01))
        else:
            # 如果不開 CNN，就設為 None
            self.cnn_trend = None
            self.cnn_seasonal = None
            self.trend_gate = None
            self.seasonal_gate = None

    def forward(self, x):
        # ==========================================
        #  區域 B: 執行運算 (Execution)
        #  這裡只負責 "使用" 剛剛建立好的零件
        # ==========================================

        # 1. Decomposition
        if self.use_decomp and self.decomp is not None:
            seasonal_part, trend_part = self.decomp(x)
        else:
            # w/o Decomp: 假設全都是 Trend, Seasonal 為 0
            seasonal_part = torch.zeros_like(x)
            trend_part = x

        B, S, C = x.shape
        trend_flat = trend_part.reshape(B, -1)
        seasonal_flat = seasonal_part.reshape(B, -1)

        # 2. Linear Parts (Base)
        trend_out_linear = self.linear_trend(trend_flat)
        seasonal_out_linear = self.linear_seasonal(seasonal_flat)

        # Base Prediction
        output = x[:, -1, 0:1] + trend_out_linear + seasonal_out_linear

        # 3. CNN Booster (如果零件存在才用)
        if self.use_cnn and self.cnn_trend is not None:
            trend_out_cnn = self.cnn_trend(trend_part)
            seasonal_out_cnn = self.cnn_seasonal(seasonal_part)

            # Fusion with Gating
            trend_final = torch.tanh(self.trend_gate) * trend_out_cnn
            seasonal_final = torch.tanh(self.seasonal_gate) * seasonal_out_cnn

            # Add to output
            output = output + trend_final + seasonal_final

        # 回傳 weights 供紀錄
        weights = None
        if self.use_cnn:
            weights = torch.stack([self.trend_gate, self.seasonal_gate])

        return output, None, weights
