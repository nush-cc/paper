import torch
import torch.nn as nn


class LearnableMovingAvg(nn.Module):
    def __init__(self, kernel_size, input_channels):
        super(LearnableMovingAvg, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=input_channels,
            bias=False,
            padding_mode='replicate'

        )
        nn.init.constant_(self.conv.weight, 1.0 / kernel_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x


class FixedMovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super(FixedMovingAvg, self).__init__()
        # 使用 AvgPool1d，它本身就沒有可學習參數
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        # 假設輸入 x 為 (Batch, Time, Channel)
        x = x.permute(0, 2, 1)
        x = self.avg_pool(x)
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size, input_channels, use_learnable=True):
        super(SeriesDecomp, self).__init__()

        if use_learnable:
            print("Learnable Moving Average Decomposition Enabled")
            self.moving_avg = LearnableMovingAvg(kernel_size, input_channels)
        else:
            print("Fixed Moving Average Decomposition Enabled")
            self.moving_avg = FixedMovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return trend, seasonal


class CNNExpert(nn.Module):
    def __init__(self, seq_len, pred_len, input_channels, hidden_dim=64, cnnExpert_KernelSize=5):
        super().__init__()

        # Params needed for tuning
        k_size = cnnExpert_KernelSize  # <--- 在這裡修改 Kernel Size (3, 5, 7...)
        dropout_p = 0.35  # <--- 在這裡修改 Dropout (0.1 ~ 0.5)

        # 自動計算 Padding (保證輸入輸出長度一致：30 -> 30)
        # 公式：(kernel_size - 1) // 2
        # 例如：k=3 -> p=1; k=5 -> p=2; k=7 -> p=3
        padding = (k_size - 1) // 2

        self.net = nn.Sequential(
            # 第一層
            nn.Conv1d(input_channels, hidden_dim, kernel_size=k_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p),

            # 第二層 (為了簡化，我們讓第二層也跟著變，或者你可以固定第二層為 k=3)
            # 這裡示範兩層都改大，讓感受野最大化
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=k_size, padding=padding),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p)
        )

        # 只要 Padding 設定正確，長度 seq_len (30) 就不會變
        self.flatten_dim = seq_len * (hidden_dim * 2)
        self.fc = nn.Linear(self.flatten_dim, pred_len)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Channels] -> [Batch, Channels, Seq_Len]
        x = x.permute(0, 2, 1)
        feat = self.net(x)
        # Flatten
        feat = feat.reshape(feat.size(0), -1)
        out = self.fc(feat)
        return out