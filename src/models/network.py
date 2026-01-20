import torch
import torch.nn as nn
from src.models.layers import SeriesDecomp, CNNExpert


class EnhancedDLinear(nn.Module):
    """
    Enhanced DLinear with Hybrid Residual Gating Mechanism.

    Architecture:
    1. Series Decomposition (Learnable or Fixed)
    2. Linear Backbone (Trend & Seasonal)
    3. CNN Expert (Booster)
    4. Hybrid Gating: Final_Weight = Sigmoid(Static_Base + Dynamic_Delta * Scale)
    """

    def __init__(self, seq_len, pred_len, input_channels,
                 use_cnn=True,
                 use_seasonal_cnn=True,
                 use_trend_cnn=False,
                 use_decomp=True,
                 hidden_dim=32,
                 trendCNNExpert_KernelSize=5,
                 seasonalCNNExpert_KernelSize=5,
                 seriesDecomposition_KernelSize=15
                 ):
        super().__init__()

        # 1. 保存設定參數
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_cnn = use_cnn
        self.use_decomp = use_decomp
        self.input_channels = input_channels
        self.use_seasonal_cnn = use_seasonal_cnn
        self.use_trend_cnn = use_trend_cnn

        # 2. Series Decomposition (序列分解)
        if self.use_decomp:
            # 建議在 layers.py 中將 SeriesDecomp 改為 LearnableMovingAvg 以獲得最佳效果
            self.decomp = SeriesDecomp(kernel_size=seriesDecomposition_KernelSize, input_channels=input_channels)
        else:
            self.decomp = None

        # 3. Linear Backbone (DLinear 核心)
        # 輸入必須攤平 (seq_len * channels)
        self.linear_trend = nn.Linear(seq_len * input_channels, pred_len)
        self.linear_seasonal = nn.Linear(seq_len * input_channels, pred_len)

        # 4. CNN Booster & Hybrid Gating (混合閘門機制)
        if self.use_cnn:
            # CNN Experts
            self.cnn_trend = CNNExpert(seq_len, pred_len, input_channels, hidden_dim, trendCNNExpert_KernelSize)
            self.cnn_seasonal = CNNExpert(seq_len, pred_len, input_channels, hidden_dim, seasonalCNNExpert_KernelSize)

            # --- [核心架構] Hybrid Residual Gating ---

            # A. 靜態基底 (Static Base):
            # 透過 nn.Parameter 學習一個全域最佳的 "底薪"。
            # 初始設為 0.0，經過 Sigmoid 後約為 0.5 (中立)，讓模型自己去調整。
            self.trend_base = nn.Parameter(torch.tensor(-1.0))
            self.seasonal_base = nn.Parameter(torch.tensor(-1.0))

            # B. 動態微調網路 (Dynamic Delta Net):
            # 負責根據輸入特徵計算 "績效獎金" (微調量)。
            # 使用瓶頸設計 (Bottleneck): 輸入 -> 8 -> 1，強迫過濾雜訊。
            self.gate_input_dim = seq_len * input_channels

            self.trend_delta_net = nn.Sequential(
                nn.Linear(self.gate_input_dim, 8),  # 降維壓縮
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Tanh()  # 輸出範圍 -1 ~ 1，代表增強或減弱
            )

            self.seasonal_delta_net = nn.Sequential(
                nn.Linear(self.gate_input_dim, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Tanh()
            )
        else:
            # 如果不使用 CNN，將相關元件設為 None
            self.cnn_trend = None
            self.cnn_seasonal = None
            self.trend_base = None
            self.seasonal_base = None
            self.trend_delta_net = None
            self.seasonal_delta_net = None

        if self.use_trend_cnn:
            self.cnn_trend = CNNExpert(seq_len, pred_len, input_channels, hidden_dim, trendCNNExpert_KernelSize)
            self.trend_base = nn.Parameter(torch.tensor(-1.0)) # 初始權重低
            self.trend_delta_net = nn.Sequential(...) # 同原本
        else:
            self.cnn_trend = None
            self.trend_base = None
            self.trend_delta_net = None

        if self.use_seasonal_cnn:
            self.cnn_seasonal = CNNExpert(seq_len, pred_len, input_channels, hidden_dim, seasonalCNNExpert_KernelSize)
            
            # [關鍵修改]：將 Base 初始化提高，強迫模型在初期重視它
            # -1.0 -> sigmoid ~ 0.26
            #  0.0 -> sigmoid = 0.50
            self.seasonal_base = nn.Parameter(torch.tensor(0.0)) 
            
            self.seasonal_delta_net = nn.Sequential(
                nn.Linear(seq_len * input_channels, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Tanh()
            )
        else:
            self.cnn_seasonal = None
            self.seasonal_base = None
            self.seasonal_delta_net = None

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Channels]

        # === 1. Series Decomposition ===
        if self.use_decomp and self.decomp is not None:
            seasonal_part, trend_part = self.decomp(x)
        else:
            # Fallback: 若無分解，假設全是 Trend (或視需求調整)
            seasonal_part = torch.zeros_like(x)
            trend_part = x

        # 準備 Flatten 特徵
        B, S, C = x.shape
        trend_flat = trend_part.reshape(B, -1)
        seasonal_flat = seasonal_part.reshape(B, -1)

        # === 2. Linear Backbone Prediction (基底) ===
        trend_out_linear = self.linear_trend(trend_flat)
        seasonal_out_linear = self.linear_seasonal(seasonal_flat)
        
        base_output_norm = trend_out_linear + seasonal_out_linear

        # === 3. CNN Expert Correction (分工處理) ===
        cnn_correction = 0.0
        
        # 預設回傳權重 (用於 Log 監控)，沒開的部分補 0
        t_weight_val = torch.tensor(0.0, device=x.device)
        s_weight_val = torch.tensor(0.0, device=x.device)

        # Part A: Trend CNN (如果開啟才跑)
        if self.use_trend_cnn and self.cnn_trend is not None:
            trend_out_cnn = self.cnn_trend(trend_part)
            delta_t = self.trend_delta_net(trend_flat)
            t_gate = torch.sigmoid(self.trend_base + delta_t)
            
            cnn_correction = cnn_correction + (t_gate * trend_out_cnn)
            t_weight_val = t_gate.mean()

        # Part B: Seasonal CNN (重點部分)
        if self.use_seasonal_cnn and self.cnn_seasonal is not None:
            seasonal_out_cnn = self.cnn_seasonal(seasonal_part)
            
            # 計算動態權重
            delta_s = self.seasonal_delta_net(seasonal_flat)
            
            # 融合 Static Base + Dynamic Delta
            s_gate = torch.sigmoid(self.seasonal_base + delta_s)
            
            # 加權並累加到修正量
            cnn_correction = cnn_correction + (s_gate * seasonal_out_cnn)
            s_weight_val = s_gate.mean()

        # === 4. Final Output ===
        final_output_norm = base_output_norm + cnn_correction

        # 堆疊權重回傳 [Trend_W, Seasonal_W]
        weights_to_return = torch.stack([t_weight_val, s_weight_val])

        return final_output_norm, base_output_norm, weights_to_return