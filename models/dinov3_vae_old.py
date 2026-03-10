import torch
from torch import nn
from .dinov3_decoder_ViT_old import *
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
            conditioning: str = "ada_ln",
            affine_scale: float = 5e-3,
            decoder_cfg = None,
            train_encoder = True,
            lora_r = 16,
            lora_alpha = 16,
            lora_dropout = 0.05,
            register_token_num = 4, # kept for signature compatibility
        ):
        super(DINOv3VAE, self).__init__()
        self.encoder = getattr(dinov3, dino_model)(**encoder_cfg)
        checkpoint = torch.load(dino_checkpoint_path)
        self.encoder.load_state_dict(checkpoint)

        # Freeze base model
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Unfreeze storage tokens (registers) and cls token if we are using them
        if with_extra_tokens:
            print("Unfreezing storage_tokens (registers) and cls_token for concatenated lighting tokens")
            for name, param in self.encoder.named_parameters():
                if 'storage_tokens' in name or 'cls_token' in name:
                    param.requires_grad = True
        
        if train_encoder:
            from peft import get_peft_model, LoraConfig, TaskType            
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
            self.encoder.eval()
            for name, param in self.encoder.named_parameters():
                if param.requires_grad:
                    print("🍎",name)

        encoder_intermediate = BackboneLayersSet(encoder_intermediate)
        self.encoder_intermediate_layers = _get_backbone_out_indices(self.encoder, encoder_intermediate)
        
        self.with_extra_tokens = with_extra_tokens
        self.n_lighting_tokens = 5 if with_extra_tokens else 1
        
        latent_dim = self.encoder.embed_dim

        # The logic here: lighting encoder takes the concatenated CLS + extra tokens
        self.lighting_encoder = nn.Sequential(
            nn.Linear(latent_dim * self.n_lighting_tokens, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, extrinsic_dim), nn.LayerNorm(extrinsic_dim)
        )

        # Decoder needs to upsample back to original image size
        # If patch_size=16 and img_size=224, encoder output is 224/16 = 14x14
        # So we need upsample_factor=16 to get back to 224x224
        patch_size = encoder_cfg.get('patch_size', 16)

        self.decoder = dinov3_decoder_base(patch_size=patch_size, extrinsic_dim=extrinsic_dim, conditioning=conditioning, affine_scale=affine_scale)


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
        
        # Select tokens for lighting
        # outputs[-1] is the last layer output
        # outputs[-1][1] is cls_token [B, D]
        # outputs[-1][2] is extra_tokens [B, N_extra, D]
        
        if self.with_extra_tokens:
            cls_token = outputs[-1][1].unsqueeze(1) # [B, 1, D]
            extra_tokens = outputs[-1][2] # [B, N_extra, D]
            # Concatenate CLS token with all extra tokens
            lighting_feat = torch.cat([cls_token, extra_tokens], dim=1) # [B, 1 + N_extra, D]
            lighting_feat = lighting_feat.flatten(1) # [B, (1 + N_extra) * D]
        else:
            lighting_feat = outputs[-1][1] # [B, D]
            
        # lighting_feat is [B, D * n_lighting_tokens]
        extrinsic = self.lighting_encoder(lighting_feat)
        return intrinsics, extrinsic

    def forward_decoder(self, intrinsics, extrinsic):
        out = self.decoder(intrinsics, extrinsic)
        return out
    
    def forward(self, x):
        intrinsics, extrinsic = self.forward_encoder(x)
        out = self.forward_decoder(intrinsics, extrinsic)
        return out
