import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utils
# -----------------------------
def warp(img, flow):
    """
    img: [B, 3, H, W]
    flow: [B, 2, H, W] (dx, dy)
    """
    B, C, H, W = img.shape
    device = img.device

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij"
    )

    grid = torch.stack((xx, yy), dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    flow_x = flow[:, 0] / (W / 2)
    flow_y = flow[:, 1] / (H / 2)

    flow_norm = torch.stack((flow_x, flow_y), dim=-1)
    grid = grid + flow_norm

    return F.grid_sample(img, grid, align_corners=True)


# -----------------------------
# Blocks
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(cout, cout, k, 1, p),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Motion Encoder
# -----------------------------
class FlowEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(6, 64),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            ConvBlock(128, 128),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            ConvBlock(256, 256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 4, 3, 1, 1)  # flow0 (2) + flow1 (2)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Refinement Network
# -----------------------------
class RefineNet(nn.Module):
    def __init__(self, channels=198):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(channels, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 128),
            ConvBlock(128, 64),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Main Model
# -----------------------------
class DiscreteVFI_v3_refine198(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow = FlowEncoder()
        self.refine = RefineNet(198)

        self.feat = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 182, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, I0, I1, t):
        B, _, H, W = I0.shape
        t = t.view(B, 1, 1, 1)

        flow_in = torch.cat([I0, I1], dim=1)
        flows = self.flow(flow_in)

        F0 = flows[:, 0:2]
        F1 = flows[:, 2:4]

        I0_w = warp(I0, F0 * t)
        I1_w = warp(I1, F1 * (1 - t))

        It_coarse = (1 - t) * I0_w + t * I1_w
        err = torch.abs(I0_w - I1_w)

        feats = self.feat(torch.cat([I0, I1], dim=1))

        refine_in = torch.cat([
            I0,
            I1,
            It_coarse,
            F0,
            F1,
            err,
            feats
        ], dim=1)

        delta = self.refine(refine_in)
        return torch.clamp(It_coarse + delta, 0.0, 1.0)
