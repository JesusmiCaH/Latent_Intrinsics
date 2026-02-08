import torch
from torch import nn
from .dinov3_decoder_ViT import *
from .dinov3.utils.backbone_out import _get_backbone_out_indices, BackboneLayersSet
from .dinov3 import vision_transformer as dinov3
from transformers import AutoModel

class DINOv3VAE(nn.Module):
    def __init__(
            self, 
            dino_model : str, 
            dino_checkpoint_path : str, 
            encoder_cfg, 
            encoder_intermediate: str|tuple, 
            extrinsic_dim = 16, 
            with_extra_tokens = False,
            decoder_cfg = None,
            train_encoder = True,
            lora_r = 16,
            lora_alpha = 16,
            lora_dropout = 0.05,
            extrinsic_token_idx = None, # None means use CLS token
        ):
        super(DINOv3VAE, self).__init__()
        self.encoder = getattr(dinov3, dino_model)(**encoder_cfg)
        checkpoint = torch.load(dino_checkpoint_path)
        self.encoder.load_state_dict(checkpoint)

        if train_encoder:
            from peft import get_peft_model, LoraConfig, TaskType
            # Freeze base model
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            peft_config = LoraConfig(
                inference_mode=False, 
                r=lora_r, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                target_modules=["qkv", "proj", "fc1", "fc2", "w1", "w2", "w3"] # Targets for DINOv3
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            self.encoder.print_trainable_parameters()
        else:
            # Frozen
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        encoder_intermediate = BackboneLayersSet(encoder_intermediate)
        self.encoder_intermediate_layers = _get_backbone_out_indices(self.encoder, encoder_intermediate)
        
        self.with_extra_tokens = with_extra_tokens
        self.n_lighting_tokens = 5 if with_extra_tokens else 1
        
        latent_dim = self.encoder.embed_dim
        self.extrinsic_token_idx = extrinsic_token_idx
        # If using a single token (CLS or specific register), input dim is just latent_dim
        # If using multiple tokens (old logic), it was latent_dim * n_lighting_tokens
        # Here we switch to single token logic as primary, but keep old logic if compatible?
        # The prompt implies we WANT single token. Let's enforce single token input for lighting encoder.
        self.lighting_encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, extrinsic_dim), nn.LayerNorm(extrinsic_dim)
        )

        # Decoder needs to upsample back to original image size
        # If patch_size=16 and img_size=224, encoder output is 224/16 = 14x14
        # So we need upsample_factor=16 to get back to 224x224
        patch_size = encoder_cfg.get('patch_size', 16)

        self.decoder = dinov3_decoder_base(patch_size=patch_size, extrinsic_dim=extrinsic_dim)


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
        
        # Select specific token for lighting
        # outputs[-1] is the last layer output
        # outputs[-1][1] is cls_token [B, D]
        # outputs[-1][2] is extra_tokens [B, N_extra, D]
        
        if self.extrinsic_token_idx is None:
            # Use CLS token
            lighting_feat = outputs[-1][1] # [B, D]
        else:
            # Use specific register token
            # Check if we have enough registers
            if outputs[-1][2].shape[1] <= self.extrinsic_token_idx:
                raise ValueError(f"Requested extrinsic_token_idx {self.extrinsic_token_idx} but only have {outputs[-1][2].shape[1]} extra tokens")
            lighting_feat = outputs[-1][2][:, self.extrinsic_token_idx, :] # [B, D]
            
        # lighting_feat is [B, D]
        extrinsic = self.lighting_encoder(lighting_feat)
        return intrinsics, extrinsic

    def forward_decoder(self, intrinsics, extrinsic):
        out = self.decoder(intrinsics, extrinsic)
        return out
    
    def forward(self, x):
        intrinsics, extrinsic = self.forward_encoder(x)
        out = self.forward_decoder(intrinsics, extrinsic)
        return out