
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .dinov3.layers import SelfAttentionBlock, Mlp, SwiGLUFFN, RMSNorm, LayerScale
from .dinov3.vision_transformer import ffn_layer_dict, norm_layer_dict
from .dinov3.layers.rope_position_encoding import RopePositionEmbedding
from .conv_modules import ConvLayer, ConvPixelShuffleUpSampleLayer, InterpolateConvUpSampleLayer

class DecoderBlock(nn.Module):
    """
    Transformer-based Decoder Block with Hypercolumn Fusion and Lighting Injection.
    """
    def __init__(
        self,
        dim: int,
        skip_dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        ffn_layer=Mlp,
        init_values: Optional[float] = None,
    ):
        super().__init__()
        
        # Fusion of current tokens and skip connection
        self.fusion_proj = nn.Linear(dim + skip_dim, dim)
        self.norm_fusion = norm_layer(dim)
        
        # Self-Attention Block (DINOv3 standard)
        self.attn_block = SelfAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            drop_path=drop_path,
            norm_layer=norm_layer,
            act_layer=act_layer,
            ffn_layer=ffn_layer,
            init_values=init_values,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, light_emb: torch.Tensor, rope: tuple = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]
            skip: [B, N, C_skip]
            light_emb: [B, C] - Lighting embedding to be used as CLS token
            rope: (sin, cos) tuple for RoPE
        """
        # 1. Fuse x and skip
        x = torch.cat([x, skip], dim=-1)
        x = self.fusion_proj(x)
        x = self.norm_fusion(x)
        
        # 2. Inject Lighting as CLS token
        # light_emb: [B, C] -> params [B, 1, C]
        cls_token = light_emb.unsqueeze(1)
        x = torch.cat([cls_token, x], dim=1) # [B, N+1, C]
        
        # 3. Attention Block
        x = self.attn_block(x, rope_or_rope_list=rope)
        
        # 4. Remove CLS token
        x = x[:, 1:, :] # [B, N, C]
        
        return x

class FeaturePyramidNeck(nn.Module):
    """
    Upsamples decoder outputs to different scales and fuses them.
    Assumes 4 scales.
    """
    def __init__(
        self,
        in_channels: int,
        target_channels: int,
        upsample_mode: str = "pixel_shuffle", # "pixel_shuffle" or "interpolate"
    ):
        super().__init__()
        
        self.upsample_mode = upsample_mode
        
        # Upsamplers for each level
        # Level 0 (Deepest): 1/16 -> 1/16 (Identity/Conv)
        self.up0 = ConvLayer(in_channels, target_channels, kernel_size=3, padding=1)
        
        # Level 1: 1/16 -> 1/8 (2x)
        if upsample_mode == "pixel_shuffle":
            self.up1 = ConvPixelShuffleUpSampleLayer(in_channels, target_channels, factor=2)
            self.up2 = ConvPixelShuffleUpSampleLayer(in_channels, target_channels, factor=4)
            self.up3 = ConvPixelShuffleUpSampleLayer(in_channels, target_channels, factor=8)
        else:
            self.up1 = InterpolateConvUpSampleLayer(in_channels, target_channels, factor=2)
            self.up2 = InterpolateConvUpSampleLayer(in_channels, target_channels, factor=4)
            self.up3 = InterpolateConvUpSampleLayer(in_channels, target_channels, factor=8)

    def forward(self, features: List[torch.Tensor], H: int, W: int) -> torch.Tensor:
        """
        Args:
            features: List of [B, N, C] tensors. Order: [Deepest, ..., Shallowest] or vice versa?
                     In DINOv3Decoder, we append results of sequential blocks.
                     Block 0 uses intrinsics[-1] (Deepest).
                     So features[0] is deepest.
            H, W: Dimensions of the 1/16 feature map (N = H*W)
        """
        
        # Reshape to spatial
        # features[i]: [B, H*W, C] -> [B, C, H, W]
        spatial_feats = []
        for feat in features:
            B, N, C = feat.shape
            spatial_feats.append(feat.reshape(B, H, W, C).permute(0, 3, 1, 2))
            
        # Upsample
        # Assuming features are [Block1_out, Block2_out, Block3_out, Block4_out]
        # We assign them to "scales". 
        # Making an assumption: Block 1 -> 1/16, Block 2 -> 1/8, Block 3 -> 1/4, Block 4 -> 1/2
        
        f0 = self.up0(spatial_feats[0]) # 1/16
        f1 = self.up1(spatial_feats[1]) # 1/8
        f2 = self.up2(spatial_feats[2]) # 1/4
        f3 = self.up3(spatial_feats[3]) # 1/2
        
        # Fuse: Upsample all to 1/2 and concat
        # max resolution is f3 (1/2 input res).
        target_size = f3.shape[-2:]
        
        f0_up = F.interpolate(f0, size=target_size, mode='bilinear', align_corners=False)
        f1_up = F.interpolate(f1, size=target_size, mode='bilinear', align_corners=False)
        f2_up = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        f3_up = f3 # Already at target size
        
        fused = torch.cat([f0_up, f1_up, f2_up, f3_up], dim=1)
        
        return fused

class DINOv3Decoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        extrinsic_dim: int = 16,
        hidden_dim: int = 512,
        out_channels: int = 3,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        img_size: int = 224,
        patch_size: int = 16,
        upsample_mode: str = "pixel_shuffle",
    ):
        super().__init__()
        
        self.num_decoder_layers = num_decoder_layers
        self.patch_size = patch_size
        self.img_size = img_size
        
        # Project Extrinsic to Hidden Dim
        self.light_proj = nn.Linear(extrinsic_dim, hidden_dim)
        
        # Initial projection for DINO feature
        self.init_proj = nn.Linear(in_dim, hidden_dim)
        
        # RoPE
        self.rope = RopePositionEmbedding(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            base=100.0 # Standard base
        )
        
        # Decoder Blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_decoder_layers):
            self.blocks.append(
                DecoderBlock(
                    dim=hidden_dim,
                    skip_dim=in_dim, # Assuming skip connections come from DINO encoder (same dim)
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    ffn_layer=Mlp, # Standard MLP
                )
            )
            
        # Feature Pyramid Neck
        # Fuses 4 levels. Each projects to hidden_dim/2 maybe? Or hidden_dim//4 to keep concat size reasonable?
        # Let's say each level outputs 'hidden_dim'. Concat 4 levels -> 4*hidden_dim.
        # This might be too large. Let's project to 128 channels each in Neck?
        # The user's provided upsample layers take in_channels and out_channels.
        # Setup Neck to output 128 channels per level -> Concat = 512.
        
        NECK_DIM = 64
        self.neck = FeaturePyramidNeck(
            in_channels=hidden_dim,
            target_channels=NECK_DIM,
            upsample_mode=upsample_mode
        )
        
        fused_dim = NECK_DIM * 4 # 256
        
        # Final Restoration Head (2 Conv Layers)
        # Input is 1/2 resolution (from Neck)
        # Need to upsample to 1/1
        
        self.final_head = nn.Sequential(
            # Conv 1
            ConvLayer(fused_dim, 128, kernel_size=3, padding=1, norm=nn.BatchNorm2d, act_func=nn.ReLU),
            # Upsample to 1/1
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # Conv 2 (Output)
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, intrinsics: List[torch.Tensor], extrinsic: torch.Tensor) -> torch.Tensor:
        """
        intrinsics: List of DINO features [B, C, H, W] or [B, N, C]?? 
                   DINOv3 usually returns [B, N, C].
                   If [B, C, H, W], we need to flatten.
                   Checking `dinov3_vae.py` -> `intrinsics` comes from `forward_encoder`.
                   `forward_encoder` -> `outputs = self.encoder.get_intermediate_layers(..., reshape=True)`
                   Wait, `dinov3_vae.py` sets `reshape=True`.
                   So intrinsics are [B, C, H, W].
                   
        extrinsic: [B, D]
        """
        # Ensure extrinsic is [B, D]
        if extrinsic.dim() == 3: extrinsic = extrinsic.squeeze(1)
        
        light_emb = self.light_proj(extrinsic) # [B, hidden_dim]
        
        # We need N (H*W) for Transformers
        # Deepest feature
        x_feat = intrinsics[-1] # [B, C, H, W]
        B, C, H, W = x_feat.shape
        x = x_feat.flatten(2).transpose(1, 2) # [B, N, C]
        
        # Initial projection
        x = self.init_proj(x)
        
        # Prepare RoPE
        # RoPE expects H, W
        rope = self.rope(H=H, W=W)
        
        decoder_outputs = []
        
        # Iterate Blocks
        # We have 4 blocks. We need 4 skips.
        # intrinsics has 4 elements? `encoder_intermediate = 'FOUR_EVEN_INTERVALS'`
        # Yes.
        # Block 0 uses skip from intrinsics[-2] ? 
        # Or should we align differently?
        # UNet:
        # Enc1 -> Dec4
        # Enc2 -> Dec3
        # Enc3 -> Dec2
        # Enc4 -> Dec1 (Deepest)
        # Here x is initialized from Enc4 (Deepest).
        # So Block 1 should use Enc3 (intrinsics[-2]).
        # Block 2 uses Enc2 (intrinsics[-3]).
        # Block 3 uses Enc1 (intrinsics[-4]).
        # Block 4 uses ?? Enc0? If we have 4 intervals.
        
        # Let's assume len(intrinsics) >= num_decoder_layers.
        
        for i in range(self.num_decoder_layers):
            # Skip connection
            # i=0 -> skip = intrinsics[-2]
            # i=1 -> skip = intrinsics[-3]
            # ...
            if i + 2 <= len(intrinsics):
                skip_feat = intrinsics[-(i+2)]
            else:
                # Fallback: reuse shallowest feature if more blocks than encoder layers
                skip_feat = intrinsics[0]
            
            skip = skip_feat.flatten(2).transpose(1, 2)
            
            x = self.blocks[i](x, skip, light_emb, rope=rope)
            decoder_outputs.append(x)
        
        # Neck
        fused = self.neck(decoder_outputs, H, W)
        
        # Head
        out = self.final_head(fused)
        
        return out

# Factory functions
def dinov3_decoder_base(**kwargs):
    return DINOv3Decoder(
        in_dim=768,
        hidden_dim=512,
        num_decoder_layers=4,
        upsample_mode="pixel_shuffle",
        **kwargs
    )
