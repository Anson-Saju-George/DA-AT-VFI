import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Basic Blocks
# -------------------------
def conv(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p),
        nn.LeakyReLU(0.1, inplace=True)
    )

# -------------------------
# FiLM Timestep Conditioning
# -------------------------
class FiLM(nn.Module):
    def __init__(self, feat_ch, t_dim=32):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, t_dim),
            nn.ReLU(inplace=True),
            nn.Linear(t_dim, t_dim),
            nn.ReLU(inplace=True),
        )
        self.to_gamma = nn.Linear(t_dim, feat_ch)
        self.to_beta = nn.Linear(t_dim, feat_ch)

    def forward(self, feat, t):
        # feat: [B, C, H, W]
        # t:    [B, 1]
        e = self.embed(t)
        gamma = self.to_gamma(e).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(e).unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta

# -------------------------
# Flow + Feature Block
# -------------------------
class FlowBlock(nn.Module):
    def __init__(self, in_ch, feat_ch):
        super().__init__()
        self.net = nn.Sequential(
            conv(in_ch, feat_ch),
            conv(feat_ch, feat_ch),
            conv(feat_ch, feat_ch)
        )
        self.flow = nn.Conv2d(feat_ch, 4, 3, 1, 1)   # F0 (2), F1 (2)
        self.mask = nn.Conv2d(feat_ch, 1, 3, 1, 1)

    def forward(self, x):
        feat = self.net(x)
        flow = self.flow(feat)
        mask = torch.sigmoid(self.mask(feat))
        return feat, flow, mask

# -------------------------
# Warp Utility
# -------------------------
def warp(img, flow):
    # img:  [B,3,H,W]
    # flow: [B,2,H,W]
    B, C, H, W = img.shape

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=img.device),
        torch.linspace(-1, 1, W, device=img.device),
        indexing="ij"
    )

    grid = torch.stack((grid_x, grid_y), dim=-1)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    fx = flow[:, 0] / (W / 2)
    fy = flow[:, 1] / (H / 2)

    flow_grid = torch.stack((fx, fy), dim=-1)
    return F.grid_sample(img, grid + flow_grid, align_corners=True)

# -------------------------
# Lazy HR Refinement Head
# -------------------------
class Refinement(nn.Module):
    def __init__(self, mid_ch=128):
        super().__init__()
        self.mid_ch = mid_ch
        self.net = None  # build on first forward

    def build(self, in_ch):
        self.net = nn.Sequential(
            conv(in_ch, self.mid_ch),
            conv(self.mid_ch, self.mid_ch),
            conv(self.mid_ch, self.mid_ch),
            nn.Conv2d(self.mid_ch, 3, 3, 1, 1)
        )

    def forward(self, x):
        if self.net is None:
            self.build(x.shape[1])
            self.net = self.net.to(x.device)
        return self.net(x)

# -------------------------
# Full Discrete VFI Model
# -------------------------
class DiscreteVFI(nn.Module):
    def __init__(self, base_ch=48):
        super().__init__()

        self.base_ch = base_ch

        # -----------------
        # Encoder
        # -----------------
        self.down1 = conv(6, base_ch)
        self.down2 = conv(base_ch, base_ch * 2, s=2)
        self.down3 = conv(base_ch * 2, base_ch * 4, s=2)

        # -----------------
        # Flow Blocks
        # -----------------
        self.flow1 = FlowBlock(base_ch, base_ch)
        self.flow2 = FlowBlock(base_ch * 2, base_ch * 2)
        self.flow3 = FlowBlock(base_ch * 4, base_ch * 4)

        # -----------------
        # FiLM Conditioning
        # -----------------
        self.film1 = FiLM(base_ch)
        self.film2 = FiLM(base_ch * 2)
        self.film3 = FiLM(base_ch * 4)

        # -----------------
        # HR Refinement (lazy)
        # -----------------
        self.refine = Refinement()

    def forward(self, I0, I1, t):
        # -----------------
        # Input
        # -----------------
        x = torch.cat([I0, I1], dim=1)  # [B,6,H,W]

        # -----------------
        # Feature Pyramid
        # -----------------
        f1 = self.down1(x)
        f2 = self.down2(f1)
        f3 = self.down3(f2)

        # -----------------
        # Flow + Mask
        # -----------------
        feat1, flow1, mask1 = self.flow1(f1)
        feat2, flow2, mask2 = self.flow2(f2)
        feat3, flow3, mask3 = self.flow3(f3)

        # -----------------
        # FiLM Modulation
        # -----------------
        feat1 = self.film1(feat1, t)
        feat2 = self.film2(feat2, t)
        feat3 = self.film3(feat3, t)

        # -----------------
        # Use Coarsest Scale Flow
        # -----------------
        flow = F.interpolate(flow3, scale_factor=4, mode="bilinear", align_corners=True)
        mask = F.interpolate(mask3, scale_factor=4, mode="bilinear", align_corners=True)

        F0 = flow[:, :2]
        F1 = flow[:, 2:]

        I0w = warp(I0, F0)
        I1w = warp(I1, F1)

        blend = mask * I0w + (1 - mask) * I1w

        # -----------------
        # HR Refinement
        # -----------------
        feat3_up = F.interpolate(feat3, scale_factor=4, mode="bilinear", align_corners=True)

        refine_in = torch.cat(
            [blend, I0, I1, feat3_up], dim=1
        )

        delta = self.refine(refine_in)

        return torch.clamp(blend + delta, 0.0, 1.0)
