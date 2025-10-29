import torch
from torch import nn
from .dinov3_decoder_ViT import *
from dinov3_utils.backbone_out import _get_backbone_out_indices, BackboneLayersSet
from . import dinov3

class DINOv3VAE(nn.Module):
    def __init__(
            self, 
            dino_model : str, 
            dino_checkpoint_path : str, 
            encoder_cfg, 
            encoder_intermediate: str|tuple, 
            with_extra_tokens = False,
            decoder_cfg = None,
        ):
        super(DINOv3VAE, self).__init__()
        self.encoder = getattr(dinov3, dino_model)(**encoder_cfg)

        # encoder is fixed as a pretrained DINOv3 model
        checkpoint = torch.load(dino_checkpoint_path)
        self.encoder.load_state_dict(checkpoint)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        encoder_intermediate = BackboneLayersSet(encoder_intermediate)
        self.encoder_intermediate_layers = _get_backbone_out_indices(self.encoder, encoder_intermediate)
        
        self.with_extra_tokens = with_extra_tokens
        self.n_lighting_tokens = 5 if with_extra_tokens else 1
        latent_dim = self.encoder.embed_dim
        self.lighting_encoder = nn.Sequential(
            nn.Linear(latent_dim * self.n_lighting_tokens, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Decoder needs to upsample back to original image size
        # If patch_size=16 and img_size=224, encoder output is 224/16 = 14x14
        # So we need upsample_factor=16 to get back to 224x224
        patch_size = encoder_cfg.get('patch_size', 16)
        
        self.decoder = dinov3_decoder_base(upsample_factor=patch_size)
        

    def forward_encoder(self, x: torch.Tensor):
        outputs = self.encoder.get_intermediate_layers(
            x, 
            n = self.encoder_intermediate_layers,
            reshape = True,
            return_class_token = True,
            return_extra_tokens = self.with_extra_tokens,
            )
        # features: List of [vis_tokens, cls_token, extra_tokens]
        intrinsics = [feat[0] for feat in outputs]
        lighting_feat = torch.cat([outputs[-1][1].unsqueeze(1), outputs[-1][2]], dim=1).reshape(x.size(0), -1)
        extrinsic = self.lighting_encoder(lighting_feat)
        return intrinsics, extrinsic

    def forward_decoder(self, intrinsics, extrinsic):
        out = self.decoder(intrinsics, extrinsic)
        return out
    
    def forward(self, x):
        intrinsics, extrinsic = self.forward_encoder(x)
        out = self.forward_decoder(intrinsics, extrinsic)
        return out