
import torch
import torch.nn as nn
import torch.nn.functional as F

class BAM(nn.Module):
    def __init__(self, in_channels, reduction=16, dilation=4):
        super(BAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=1)
        )

    def forward(self, x):
        ch_att = self.channel_attention(x)
        sp_att = self.spatial_attention(x)
        att = torch.sigmoid(ch_att + sp_att)
        return x * att

class ECA(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
