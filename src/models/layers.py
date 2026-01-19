import torch.nn as nn
import torch


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
    def __init__(self, kernel_size, input_channels):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = LearnableMovingAvg(kernel_size, input_channels)
        # self.moving_avg = FixedMovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class CNNExpert(nn.Module):
    def __init__(self, seq_len, pred_len, input_channels, hidden_dim=64, cnnExpert_KernelSize=5):
        super().__init__()

        # Params needed for tuning
        k_size = cnnExpert_KernelSize  # <--- 在這裡修改 Kernel Size (3, 5, 7...)
        dropout_p = 0.2  # <--- 在這裡修改 Dropout (0.1 ~ 0.5)

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


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_idx=None):
        # 這裡有小修改，支援只還原「特定欄位」
        if self.affine:
            # 如果是針對特定欄位還原，要取出對應的 affine 參數
            if target_idx is not None:
                x = x - self.affine_bias[target_idx]
                x = x / (self.affine_weight[target_idx] + self.eps * self.eps)
            else:
                x = x - self.affine_bias
                x = x / (self.affine_weight + self.eps * self.eps)

        # 還原 Mean 和 Std
        if target_idx is not None:
            # 取出對應 channel 的 mean/std (Shape: [Batch, 1, 1])
            mean = self.mean[:, :, target_idx:target_idx + 1]
            stdev = self.stdev[:, :, target_idx:target_idx + 1]
            x = x * stdev
            x = x + mean
        else:
            x = x * self.stdev
            x = x + self.mean
        return x

    def forward(self, x, mode: str, target_idx=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_idx)
        else:
            raise NotImplementedError
        return x
