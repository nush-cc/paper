import torch
import torch.nn as nn
# 記得引入你的 SeriesDecomp 和 CNNExpert，假設它們在 layers.py
# 如果是在同一個檔案就不用 import
from src.models.layers import SeriesDecomp, CNNExpert, RevIN


class EnhancedDLinear(nn.Module):
    def __init__(self, seq_len, pred_len, input_channels,
                 use_cnn=True,  # <--- 這些是 init 的參數
                 use_decomp=True,
                 use_revin=True,
                 hidden_dim=32,
                 cnnExpert_KernelSize=5,
                 seriesDecomposition_KernelSize=15
                 ):
        super().__init__()

        # 1. 把參數存起來 (這樣 forward 才讀得到)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_cnn = use_cnn
        self.use_decomp = use_decomp
        self.use_revin = use_revin

        if self.use_revin:
            self.revin = RevIN(num_features=input_channels)
        else:
            self.revin = None

        # ==========================================
        #  區域 A: 建立層 (Building Layers)
        #  這裡只負責 "定義" 模型有什麼零件
        # ==========================================

        # --- 分解層 ---
        if self.use_decomp:
            self.decomp = SeriesDecomp(kernel_size=seriesDecomposition_KernelSize, input_channels=input_channels)
        else:
            self.decomp = None

        # --- Linear Backbone (一定會有) ---
        self.linear_trend = nn.Linear(seq_len * input_channels, pred_len)
        self.linear_seasonal = nn.Linear(seq_len * input_channels, pred_len)

        # --- CNN Booster (根據開關決定是否建立) ---
        if self.use_cnn:
            self.cnn_trend = CNNExpert(seq_len, pred_len, input_channels, hidden_dim, cnnExpert_KernelSize)
            self.cnn_seasonal = CNNExpert(seq_len, pred_len, input_channels, hidden_dim, cnnExpert_KernelSize)

            # Gating Weights
            self.trend_gate = nn.Parameter(torch.tensor(-0.01))
            self.seasonal_gate = nn.Parameter(torch.tensor(0.01))

            self.trend_gate_net = nn.Sequential(
                nn.Linear(seq_len * input_channels, 16),  # 先壓縮特徵
                nn.Tanh(),  # 活化函數
                nn.Linear(16, 1),  # 輸出一個純量權重
                nn.Sigmoid()  # 限制在 0~1 之間 (代表信心程度)
            )

            self.seasonal_gate_net = nn.Sequential(
                nn.Linear(seq_len * input_channels, 16),
                nn.Tanh(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        else:
            # 如果不開 CNN，就設為 None
            self.cnn_trend = None
            self.cnn_seasonal = None
            self.trend_gate = None
            self.seasonal_gate = None
            self.trend_gate_net = None
            self.seasonal_gate_net = None

    def forward(self, x):
        # ==========================================
        #  區域 B: 執行運算 (Execution)
        #  這裡只負責 "使用" 剛剛建立好的零件
        # ==========================================
        if self.use_revin:
            x = self.revin(x, mode='norm')

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
        # output = x[:, -1, 0:1] + trend_out_linear + seasonal_out_linear
        base_output_norm = trend_out_linear + seasonal_out_linear

        # 3. CNN Booster (如果零件存在才用)
        cnn_correction = 0.0
        if self.use_cnn and self.cnn_trend is not None:
            trend_out_cnn = self.cnn_trend(trend_part)
            seasonal_out_cnn = self.cnn_seasonal(seasonal_part)

            # t_gate = self.trend_gate_net(trend_flat)  # Shape: (B, 1)
            # s_gate = self.seasonal_gate_net(seasonal_flat)
            #
            # trend_cnn_final = t_gate * trend_out_cnn
            # seasonal_cnn_final = s_gate * seasonal_out_cnn

            # Fusion with Gating
            trend_cnn_final = torch.tanh(self.trend_gate) * trend_out_cnn
            seasonal_cnn_final = torch.tanh(self.seasonal_gate) * seasonal_out_cnn

            # Add to output
            cnn_correction = trend_cnn_final + seasonal_cnn_final

        final_output_norm = base_output_norm + cnn_correction

        def denorm_volatility(out_tensor):
            if not self.use_revin:
                return out_tensor

            out_tensor = out_tensor.unsqueeze(-1)  # [B, Pred, 1]
            out_tensor = self.revin(out_tensor, mode='denorm', target_idx=0)
            return out_tensor.squeeze(-1)

        output_final = denorm_volatility(final_output_norm)
        output_base = denorm_volatility(base_output_norm)

        # 回傳 weights 供紀錄
        weights = None
        if self.use_cnn:
            weights = torch.stack([self.trend_gate, self.seasonal_gate])

        return output_final, output_base, weights
