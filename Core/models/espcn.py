# Core/models/espcn.py

import torch
import torch.nn as nn


class ESPCN(nn.Module):
    def __init__(self, scale=2, in_ch=3, feat=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, feat, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(feat, feat // 2, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(feat // 2, in_ch * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        return self.body(x)
