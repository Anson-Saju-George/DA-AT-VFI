import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return x + self.net(x)


class DomainTokenVFI(nn.Module):
    def __init__(self, channels=64, num_domains=3):
        super().__init__()

        self.embed = nn.Embedding(num_domains, channels)

        self.enc = nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            ConvBlock(channels),
            ConvBlock(channels),
        )

        self.mid = nn.Sequential(
            ConvBlock(channels),
            ConvBlock(channels),
        )

        self.dec = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, I0, I1, t, domain_id):
        x = torch.cat([I0, I1], dim=1)
        f = self.enc(x)

        B, C, H, W = f.shape
        token = self.embed(domain_id).view(B, C, 1, 1)
        f = f + token

        f = self.mid(f)
        out = self.dec(f)
        return out

