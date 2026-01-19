import torch
import torch.nn as nn
from src.models.layers import SeriesDecomp, CNNExpert, RevIN


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
                 use_decomp=True,
                 use_revin=True,
                 hidden_dim=32,
                 cnnExpert_KernelSize=5,
                 seriesDecomposition_KernelSize=15
                 ):
        super().__init__()

        # 1. 保存設定參數
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_cnn = use_cnn
        self.use_decomp = use_decomp
        self.use_revin = use_revin
        self.input_channels = input_channels

        # 2. RevIN (可逆歸一化)
        if self.use_revin:
            self.revin = RevIN(num_features=input_channels)
        else:
            self.revin = None

        # 3. Series Decomposition (序列分解)
        if self.use_decomp:
            # 建議在 layers.py 中將 SeriesDecomp 改為 LearnableMovingAvg 以獲得最佳效果
            self.decomp = SeriesDecomp(kernel_size=seriesDecomposition_KernelSize, input_channels=input_channels)
        else:
            self.decomp = None

        # 4. Linear Backbone (DLinear 核心)
        # 輸入必須攤平 (seq_len * channels)
        self.linear_trend = nn.Linear(seq_len * input_channels, pred_len)
        self.linear_seasonal = nn.Linear(seq_len * input_channels, pred_len)

        # 5. CNN Booster & Hybrid Gating (混合閘門機制)
        if self.use_cnn:
            # CNN Experts
            self.cnn_trend = CNNExpert(seq_len, pred_len, input_channels, hidden_dim, cnnExpert_KernelSize)
            self.cnn_seasonal = CNNExpert(seq_len, pred_len, input_channels, hidden_dim, cnnExpert_KernelSize)

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

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Channels]

        # 1. RevIN Normalization
        if self.use_revin:
            x = self.revin(x, mode='norm')

        # 2. Series Decomposition
        if self.use_decomp and self.decomp is not None:
            seasonal_part, trend_part = self.decomp(x)
        else:
            # Fallback: 如果不分解，假設全是趨勢
            seasonal_part = torch.zeros_like(x)
            trend_part = x

        # 準備攤平的特徵供 Linear 和 Gating 使用
        B, S, C = x.shape
        trend_flat = trend_part.reshape(B, -1)
        seasonal_flat = seasonal_part.reshape(B, -1)

        # 3. Linear Backbone Prediction
        trend_out_linear = self.linear_trend(trend_flat)
        seasonal_out_linear = self.linear_seasonal(seasonal_flat)

        # 基礎預測結果
        base_output_norm = trend_out_linear + seasonal_out_linear

        # 4. CNN Expert Correction (with Hybrid Gating)
        cnn_correction = 0.0
        weights_to_return = None  # 預設回傳值

        if self.use_cnn and self.cnn_trend is not None:
            # CNN 特徵提取
            trend_out_cnn = self.cnn_trend(trend_part)
            seasonal_out_cnn = self.cnn_seasonal(seasonal_part)

            # --- 計算混合權重 ---

            # Step A: 計算動態 Delta (限制幅度)
            # 乘上 0.2 是為了限制動態網路的權限 (Residual Scaling)，防止過擬合
            delta_t = self.trend_delta_net(trend_flat) * 0.2
            delta_s = self.seasonal_delta_net(seasonal_flat) * 0.2

            # Step B: 結合靜態與動態 (Base + Delta)
            # 使用 Sigmoid 確保最終權重在 0~1 之間 (融合比例)
            t_gate = torch.sigmoid(self.trend_base + delta_t)
            s_gate = torch.sigmoid(self.seasonal_base + delta_s)

            # Step C: 加權融合 (Gated Fusion)
            # t_gate [B, 1] * trend_out_cnn [B, Pred] -> Broadcast
            trend_cnn_final = t_gate * trend_out_cnn
            seasonal_cnn_final = s_gate * seasonal_out_cnn

            # Step D: 加總 CNN 修正量
            cnn_correction = trend_cnn_final + seasonal_cnn_final

            # Step E: 紀錄權重供分析
            # 回傳 batch 的平均權重，方便在訓練 log 中觀察變化
            weights_to_return = torch.stack([t_gate.mean(), s_gate.mean()])

        # 5. 最終加總
        final_output_norm = base_output_norm + cnn_correction

        # 6. Denormalization (還原)與輸出處理
        def denorm_volatility(out_tensor):
            """Helper function to denormalize output"""
            if not self.use_revin:
                return out_tensor
            # RevIN 需要 [B, Len, C] 的形狀
            out_tensor = out_tensor.unsqueeze(-1)
            # 這裡假設我們要預測的是 target_idx=0 (通常是 Close Price 或 Volatility)
            # 確保你的 RevIN denorm 邏輯與 metrics.py 中的一致
            out_tensor = self.revin(out_tensor, mode='denorm', target_idx=0)
            return out_tensor.squeeze(-1)

        output_final = denorm_volatility(final_output_norm)
        output_base = denorm_volatility(base_output_norm)

        # output_final: 模型最終預測 (Linear + CNN)
        # output_base: 僅 Linear 的預測 (用來做 Ablation 比較)
        # weights_to_return: 閘門權重 (用來做論文分析)
        return output_final, output_base, weights_to_return
