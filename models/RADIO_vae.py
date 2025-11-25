import torch
from torch import nn
from .dinov3_decoder_ViT import *
from . import dinov3
from transformers import AutoModel

class RadioVAE(nn.Module):
    def __init__(
            self, 
            encoder_intermediate: tuple|list = [2, 5, 8, 11], 
            model_version: str = "c-radio_v3-b",
            extrinsic_dim = 16,
        ):
        super(RadioVAE, self).__init__()
        self.encoder = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True)

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.intermediate_indices = encoder_intermediate
        
        latent_dim = self.encoder.embed_dim

        self.lighting_encoder = nn.Sequential(
            nn.Linear(self.encoder.summary_dim, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, extrinsic_dim), nn.LayerNorm(extrinsic_dim)
        )

        # Decoder needs to upsample back to original image size
        # If patch_size=16 and img_size=224, encoder output is 224/16 = 14x14
        # So we need upsample_factor=16 to get back to 224x224
        patch_size = self.encoder.patch_size
        
        self.decoder = DINOv3Decoder(
            in_dim=latent_dim,
            extrinsic_dim=extrinsic_dim,
            hidden_dim=512,
            out_channels=3,
            num_decoder_layers=len(encoder_intermediate),
            num_heads=8,
            alpha=0.5,
            upsample_factor=patch_size,
        )
        

    def forward_encoder(self, x: torch.Tensor):
        rets, intrinsics = self.encoder.forward_intermediates(
            x, indices=self.intermediate_indices,
        )
        # intrinsics: List of feature maps at specified indices [B, C, H, W]
        # rets: summary vector

        summary, featmap = rets     # summary: [B, C], featmap: [B, C, H, W]
        extrinsic = self.lighting_encoder(summary)
        return intrinsics, extrinsic

    def forward_decoder(self, intrinsics, extrinsic):
        out = self.decoder(intrinsics, extrinsic)
        return out
    
    def forward(self, x):
        intrinsics, extrinsic = self.forward_encoder(x)
        out = self.forward_decoder(intrinsics, extrinsic)
        return out