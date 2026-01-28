import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Basic Conv Block
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, k, 1, p),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Motion Encoder
# -----------------------------
class MotionEncoder(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, c, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, I0, I1):
        x = torch.cat([I0, I1], dim=1)
        return self.net(x).flatten(1)

# -----------------------------
# Gating Network
# -----------------------------
class GatingNet(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, 2)
        )

    def forward(self, m):
        return torch.softmax(self.fc(m), dim=1)

# -----------------------------
# VFI Backbone
# -----------------------------
class VFIBranch(nn.Module):
    def __init__(self, cin=7, c=64):
        super().__init__()
        self.enc = ConvBlock(cin, c)
        self.mid = ConvBlock(c, c)
        self.dec = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, I0, I1, t):
        B, _, H, W = I0.shape
        tmap = t.view(B, 1, 1, 1).expand(B, 1, H, W)
        x = torch.cat([I0, I1, tmap], dim=1)
        x = self.enc(x)
        x = self.mid(x)
        return torch.sigmoid(self.dec(x))

# -----------------------------
# Full Model
# -----------------------------
class MotionAwareVFI(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion = MotionEncoder()
        self.gate = GatingNet()

        self.rigid_branch = VFIBranch()
        self.deform_branch = VFIBranch()

    def forward(self, I0, I1, t):
        m = self.motion(I0, I1)
        g = self.gate(m)

        rigid = self.rigid_branch(I0, I1, t)
        deform = self.deform_branch(I0, I1, t)

        g1 = g[:, 0].view(-1, 1, 1, 1)
        g2 = g[:, 1].view(-1, 1, 1, 1)

        return g1 * rigid + g2 * deform
