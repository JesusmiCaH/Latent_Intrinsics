import torch
from torch import nn
from dinov3_decoder import DINOv3Decoder
import dinov3

class DINOv3VAE(nn.Module):
    def __init__(self, dino_model : str, encoder_cfg, with_extra_tokens = False):
        super(DINOv3VAE, self).__init__()
        self.encoder = getattr(dinov3, dino_model)(**encoder_cfg)
        
        # encoder is fixed as a pretrained DINOv3 model
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        n_lighting_tokens = 5 if with_extra_tokens else 1
        latent_dim = self.encoder.embed_dim
        self.lighting_encoder = nn.ModuleList(
            [
                nn.Linear(latent_dim * n_lighting_tokens, latent_dim), nn.GELU(),
                nn.Linear(latent_dim, latent_dim), nn.GELU(),
                nn.Linear(latent_dim, latent_dim)
            ]
        )
        self.decoder = DINOv3Decoder(in_channels=(latent_dim, latent_dim, latent_dim, latent_dim))

    def train(self, mode=True):
        """重写train方法，确保encoder始终保持eval模式"""
        super().train(mode)
        # 无论如何都保持encoder为eval模式
        self.encoder.eval()
        return self
    
    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out