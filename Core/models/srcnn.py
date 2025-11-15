# Core/models/srcnn.py
import torch
import torch.nn as nn

class SRCNN(nn.Module):
    """
    SRCNN travaille classiquement sur l’image déjà upscalée (bicubique),
    puis “nettoie”/améliore. Ici on respecte ce paradigme.
    """
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_ch, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.net(x)
