import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from functools import partial
from utils.meanflow import Normalizer, adaptive_l2_loss


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=nn.ReLU):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
class Lighting_Encoder(nn.Module):
    def __init__(self, latent_dim=128, out_dim=512):
        super(Lighting_Encoder, self).__init__()
        self.conv = nn.Sequential(
            # VAE feat dim: 4
            ConvBNAct(4, latent_dim//2, kernel_size=4, stride=2, padding=1, activation=nn.SiLU),
            ConvBNAct(latent_dim//2, latent_dim, kernel_size=4, stride=2, padding=1, activation=nn.SiLU),
        )
        self.linear1 = nn.Linear(latent_dim, latent_dim)
        self.act1 = nn.SiLU()
        self.linear2 = nn.Linear(latent_dim, out_dim)

    def forward(self, x):
        # x: (B, 4, H, W)
        x = self.conv(x)  # (B, latent_dim, H/8, W/8)
        x = torch.mean(x, dim=(2, 3))  # (B, latent_dim)
        x = self.act1(self.linear1(x))  # (B, latent_dim)
        x = self.linear2(x)  # (B, out_dim)
        return x