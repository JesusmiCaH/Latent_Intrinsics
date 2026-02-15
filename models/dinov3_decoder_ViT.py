
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .dinov3.layers import SelfAttentionBlock, Mlp, SwiGLUFFN, RMSNorm, LayerScale, SelfAttention
from .dinov3.vision_transformer import ffn_layer_dict, norm_layer_dict
from .dinov3.layers.rope_position_encoding import RopePositionEmbedding
from .conv_modules import ConvLayer, ConvPixelShuffleUpSampleLayer, InterpolateConvUpSampleLayer

class AdaLNModulator(nn.Module):
    def __init__(self, dim, condition_dim):
        super().__init__()
        # Predict 6 parameters: 
        # (scale_1, shift_1, gate_1, scale_2, shift_2, gate_2)
        # scale/shift for Norms, gate for Residual scaling
        
        self.dim = dim
        self.silu = nn.SiLU()
        self.lin = nn.Linear(condition_dim, 6 * dim, bias=True)
        
        # Init zero for scale/shift to start as identity/zero effect?
        # AdaLN-Zero initializes final projection to zero.
        nn.init.constant_(self.lin.weight, 0)
        nn.init.constant_(self.lin.bias, 0)

    def forward(self, condition):
        # condition: [B, C_cond]
        res = self.lin(self.silu(condition)) # [B, 6*dim]
        # Split
        scale_1, shift_1, gate_1, scale_2, shift_2, gate_2 = res.chunk(6, dim=-1)
        # Shapes: [B, dim] -> [B, 1, dim] for broadcasting
        return (scale_1.unsqueeze(1), shift_1.unsqueeze(1), gate_1.unsqueeze(1),
                scale_2.unsqueeze(1), shift_2.unsqueeze(1), gate_2.unsqueeze(1))

class CrossAttention(nn.Module):
    """
    Cross Attention Layer.
    """
    def __init__(self, dim, condition_dim, num_heads, qkv_bias=True, proj_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = (dim // num_heads) ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(condition_dim, dim, bias=qkv_bias)
        self.v = nn.Linear(condition_dim, dim, bias=qkv_bias)
        
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        # Norm is handled in DecoderBlock now
        
    def forward(self, x, condition):
        # x: [B, N, C] (Queries)
        # condition: [B, C] or [B, L, C] (Keys/Values)
        
        B, N, C = x.shape
        if condition.dim() == 2:
            condition = condition.unsqueeze(1) # [B, 1, C]
            
        q = self.q(x)
        k = self.k(condition)
        v = self.v(condition)
        
        # Reshape for multi-head
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, condition.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, condition.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Attention
        x = F.scaled_dot_product_attention(q, k, v) # [B, nH, N, d]
        
        x = x.transpose(1, 2).flatten(2)
        x = self.proj(x)
        
        return x

class DecoderBlock_AdaLN(nn.Module):
    """
    Decoder Block with AdaLN-Zero conditioning.
    Predicts scale/shift for norms and gatings for residuals.
    No LayerScale used.
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
        init_values: Optional[float] = None, # Unused here but kept for signature compatibility if needed
    ):
        super().__init__()
        
        # Fusion
        self.fusion_proj = nn.Linear(dim + skip_dim, dim)
        self.norm_fusion = norm_layer(dim)
        self.silu = nn.SiLU()
        
        # AdaLN Modulator (must be defined/passed, here we assume it is instantiated inside or passed?
        # The design in DINOv3Decoder iterates blocks.
        # It's better if the block manages its own modulator OR DINOv3Decoder passes the modulation params.
        # But previous design: block has self.ada_ln. DINOv3Decoder passes light_emb.
        # We will keep that.
        self.ada_ln = AdaLNModulator(dim, dim)

        # Block 1: Self-Attention
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            proj_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
        )
        
        # Block 2: FFN
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=0.0,
            bias=True,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, light_emb: torch.Tensor, rope: tuple = None) -> torch.Tensor:
        # 1. Fuse
        x = torch.cat([x, skip], dim=-1)
        x = self.fusion_proj(x)
        x = self.norm_fusion(x)
        
        # 2. Predict parameters
        scale1, shift1, gate1, scale2, shift2, gate2 = self.ada_ln(light_emb)
        
        # 3. SA Block
        x_norm1 = self.norm1(x)
        x_norm1 = x_norm1 * (1 + scale1) + shift1
        x = x + gate1 * self.attn(x_norm1, rope=rope)
        
        # 4. FFN Block
        x_norm2 = self.norm2(x)
        x_norm2 = x_norm2 * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(x_norm2)
        
        return x

class DecoderBlock_CrossAttn(nn.Module):
    """
    Decoder Block with Cross-Attention conditioning.
    Uses LayerScale.
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
        affine_scale: float = 0.0,
    ):
        super().__init__()
        
        # Fusion
        self.fusion_proj = nn.Linear(dim + skip_dim, dim)
        self.norm_fusion = norm_layer(dim)
        self.silu = nn.SiLU()
        
        # Block 1: Self-Attention
        self.norm1 = norm_layer(dim)
        self.self_attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            proj_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
        )
        
        # Block 2: Cross-Attention
        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim=dim, 
            condition_dim=dim, 
            num_heads=num_heads,
            qkv_bias=True
        )
        self.affine_scale = affine_scale
        
        # Block 3: FFN
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=0.0,
            bias=True,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, light_emb: torch.Tensor, rope: tuple = None) -> torch.Tensor:
        # 1. Fuse
        x = torch.cat([x, skip], dim=-1)
        x = self.fusion_proj(x)
        x = self.norm_fusion(x)
        
        # 2. SA (Residual)
        x = x + self.self_attn(self.norm1(x), rope=rope)
        
        # 3. CA (Residual)
        x = x + self.affine_scale * self.cross_attn(self.norm2(x), light_emb)
        
        # 4. FFN (Residual)
        x = x + self.mlp(self.norm3(x))
        
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
        conditioning: str = "ada_ln",
        affine_scale: float = 0.0,
        init_values: float = 0.1,
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

        # Affine value for conditioning
        self.affine_scale = affine_scale
        
        # Decoder Blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_decoder_layers):
            if conditioning == "ada_ln":
                block = DecoderBlock_AdaLN(
                    dim=hidden_dim,
                    skip_dim=in_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    ffn_layer=Mlp,
                    init_values=None, # Not used in AdaLN
                )
            elif conditioning == "cross_attn":
                block = DecoderBlock_CrossAttn(
                    dim=hidden_dim,
                    skip_dim=in_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    ffn_layer=Mlp,
                    affine_scale=affine_scale,
                )
            else:
                raise ValueError(f"Unknown conditioning: {conditioning}")
            
            self.blocks.append(block)
            
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
        intrinsics: List of DINO features [B, C, H, W]
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
