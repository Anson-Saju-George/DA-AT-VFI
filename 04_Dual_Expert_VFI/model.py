import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------
# Basic Conv Block (same style as 03)
# -------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------
# Dual Expert VFI
# -------------------------------------------------
class DualExpertVFI(nn.Module):
    def __init__(self, feat=64):
        super().__init__()

        # SAME INPUT CONTRACT AS 03
        # I0(3) + I1(3) + t(1) = 7
        self.enc = nn.Sequential(
            nn.Conv2d(7, feat, 3, padding=1),
            nn.ReLU(inplace=True),
            ConvBlock(feat, feat),
        )

        # Expert branches
        self.expert_rigid = ConvBlock(feat, feat)
        self.expert_deform = ConvBlock(feat, feat)

        # Gating head (global)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat, 2, 1)
        )

        # Decoder
        self.dec = nn.Sequential(
            ConvBlock(feat, feat),
            nn.Conv2d(feat, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, I0, I1, t):
        B, _, H, W = I0.shape
        t_map = t.view(B, 1, 1, 1).expand(B, 1, H, W)

        x = torch.cat([I0, I1, t_map], dim=1)  # ✅ 7 channels
        feat = self.enc(x)

        f_rigid = self.expert_rigid(feat)
        f_deform = self.expert_deform(feat)

        gate_logits = self.gate(feat).view(B, 2)
        gate = torch.softmax(gate_logits, dim=1)

        fused = (
            gate[:, 0].view(B, 1, 1, 1) * f_rigid +
            gate[:, 1].view(B, 1, 1, 1) * f_deform
        )

        out = self.dec(fused)
        return out, gate
