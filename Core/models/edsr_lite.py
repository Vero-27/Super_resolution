# Core/models/edsr_lite.py
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, n_feats=64, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1)
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats=64):
        m = []
        for _ in range(int(scale).bit_length() - 1):
            m += [nn.Conv2d(n_feats, 4*n_feats, 3, padding=1), nn.PixelShuffle(2), nn.ReLU(inplace=True)]
            scale //= 2
        super().__init__(*m)

class EDSRLite(nn.Module):
    def __init__(self, scale=2, n_resblocks=8, n_feats=64, in_ch=3):
        super().__init__()
        self.head = nn.Conv2d(in_ch, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)])
        self.tail = nn.Sequential(
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, in_ch, 3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.tail(x)
        return x
