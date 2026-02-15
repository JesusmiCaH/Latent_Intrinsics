"""
ViT-based Decoder for DINOv3 with Hypercolumn Fusion

This decoder combines multi-scale information from DINOv3 intermediate layers using:
- Normalized hypercolumn fusion: F.normalize(last_layer) + alpha * F.normalize(hypercolumns(earlier_layers))
- ViT transformer blocks for processing
- Convolutional refinement head to reduce tile/blur artifacts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from functools import partial

from layers import SelfAttentionBlock, Mlp, SwiGLUFFN, RMSNorm, LayerScale
from .dinov3 import ffn_layer_dict, norm_layer_dict, dtype_dict
from layers.CNN_blocks import ConvModule, ResidualBlock


class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding for transformer decoder
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        channels = int(channels / 2)
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, h: int, w: int, device=None) -> torch.Tensor:
        """
        Returns positional encoding [1, C, H, W]
        """
        device = device or self.inv_freq.device
        
        # Create position indices
        pos_h = torch.arange(h, device=device).type_as(self.inv_freq)
        pos_w = torch.arange(w, device=device).type_as(self.inv_freq)
        
        # Compute sin/cos embeddings
        sin_inp_h = torch.einsum("i,j->ij", pos_h, self.inv_freq)
        sin_inp_w = torch.einsum("i,j->ij", pos_w, self.inv_freq)
        
        emb_h = torch.cat((sin_inp_h.sin(), sin_inp_h.cos()), dim=-1)  # [H, C/2]
        emb_w = torch.cat((sin_inp_w.sin(), sin_inp_w.cos()), dim=-1)  # [W, C/2]
        
        # Combine height and width embeddings
        emb = torch.zeros((h, w, self.channels), device=device, dtype=emb_h.dtype)
        emb[:, :, :self.channels//2] = emb_h.unsqueeze(1).expand(h, w, -1)
        emb[:, :, self.channels//2:] = emb_w.unsqueeze(0).expand(h, w, -1)
        
        return emb.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

class DecoderBlock(nn.Module):
    """
    Single transformer decoder block
    """
    def __init__(
        self,
        in_dim = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path: float = 0.0,
        norm_layer = nn.LayerNorm,
        act_layer = nn.GELU,
        ffn_layer = "mlp",
        init_values: Optional[float] = None,
        alpha: float = 0.01,
    ):
        super().__init__()
        self.lighting_affine = nn.Linear(hidden_dim, hidden_dim)

        self.hypercolumn_proj = nn.Linear(in_dim+hidden_dim, hidden_dim)
        self.norm = norm_layer(hidden_dim, eps=1e-6)
        self.silu = nn.SiLU()

        self.attn_block = SelfAttentionBlock(
            dim=hidden_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop_path=drop_path,
            norm_layer=norm_layer,
            act_layer=act_layer,
            ffn_layer=ffn_layer,
            init_values=init_values,
        )
        self.alpha = alpha

    def forward(self, x: torch.Tensor, skip: torch.Tensor, light_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] input tokens
            skip: [B, N, C] skip tokens
            light_emb: [B, C] lighting tokens
        Returns:
            [B, N, C] output tokens
        """
        # Constraint Scaling
        param = self.lighting_affine(light_emb[:, None]).tanh() * self.alpha  # B, 1, C
        x = self.silu(self.norm(x) * (1 + param))
        # Hypercolumn fusion
        x = torch.cat([x, skip], dim=-1)
        x = self.hypercolumn_proj(x)
        # Self-attention block
        x = self.attn_block(x)
        return x

class ConvRefinementHead(nn.Module):
    """
    Convolutional refinement head to reduce tile effects and blur artifacts
    Uses residual connections and progressive upsampling
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        hidden_channels: int = 128,
        num_layers: int = 3,
        upsample_factor: int = 16,
    ):
        super().__init__()
        
        self.upsample_factor = upsample_factor
        
        # Initial projection
        self.proj_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        
        # Progressive upsampling
        self.upsample_layers = nn.ModuleList()
        current_factor = 1
        while current_factor < upsample_factor:
            self.upsample_layers.append(
                ConvModule(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    norm=nn.GroupNorm,
                    norm_cfg={"num_groups": min(32, hidden_channels)},
                    activation=nn.GELU,
                    is_transpose=True,
                    upsample_factor=1,
                    bias=False,
                )
            )
            current_factor *= 2
        
        # Refinement layers with residual connections
        self.refine_layers = nn.ModuleList()
        for i in range(num_layers):
            self.refine_layers.append(
                ResidualBlock(
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    out_channels=hidden_channels,
                    num_groups=min(32, hidden_channels),
                    activation=nn.GELU,
                )
            )
        
        # Final output projection
        self.proj_out = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input features
        Returns:
            [B, out_channels, H*upsample_factor, W*upsample_factor] output image
        """
        # Initial projection
        x = self.proj_in(x)
        
        # Progressive upsampling
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)

        # Refinement with residual connections
        for refine_layer in self.refine_layers:
            x = refine_layer(x)
        
        # Final projection to output
        x = self.proj_out(x)
        return x


class DINOv3Decoder(nn.Module):
    """
    ViT-based decoder for DINOv3 features
    
    Architecture:
    1. Hypercolumn fusion from multiple DINO layers with normalization
    2. Positional encoding addition
    3. Stack of transformer decoder blocks
    4. Convolutional refinement head for reducing artifacts
    
    Args:
        in_dim: Input dimension from DINO encoder (default: 768 for ViT-B)
        hidden_dim: Hidden dimension for decoder (default: 512)
        out_channels: Number of output channels (default: 3 for RGB)
        num_decoder_layers: Number of transformer decoder blocks (default: 4)
        num_heads: Number of attention heads (default: 8)
        ffn_ratio: FFN expansion ratio (default: 4.0)
        alpha: Weight for constraint scaling (default: 0.5)
        upsample_factor: Upsampling factor for final output (default: 16)
        norm_layer: Normalization layer type (default: "layernorm")
        ffn_layer: FFN layer type (default: "mlp")
    """
    def __init__(
        self,
        in_dim: int = 768,
        extrinsic_dim: int = 16, 
        hidden_dim: int = 512,
        out_channels: int = 3,
        num_decoder_layers: int = 4,
        layer_depth: int = 1,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        alpha: float = 0.5,
        upsample_factor: int = 16,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_decoder_layers = num_decoder_layers
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(hidden_dim)

        self.lighting_decoder = nn.Sequential(
            nn.Linear(extrinsic_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=norm_layer_dict[norm_layer],
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_dict[ffn_layer],
                init_values=None,
                alpha=alpha,
            ) for _ in range(num_decoder_layers)
        ])

        # Final normalization
        if norm_layer == "layernorm":
            self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        else:
            self.final_norm = RMSNorm(hidden_dim)
        
        # Convolutional refinement head
        self.conv_head = ConvRefinementHead(
            in_channels=hidden_dim,
            out_channels=out_channels,
            hidden_channels=hidden_dim // 2,
            num_layers=1,
            upsample_factor=upsample_factor,
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, intrinsics: List[torch.Tensor], extrinsic: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            intrinsics: List of feature maps from DINO encoder [B, C, H, W]
                       Typically 4 layers from different depths
            extrinsic: Optional extrinsic/lighting code [B, D] (not used in this version,
                      but kept for compatibility with the VAE interface)
        
        Returns:
            Reconstructed image [B, 3, H_out, W_out]
        """
        # Formula: F.normalize(last_layer) + alpha * F.normalize(hypercolumns(earlier_layers))
        
        latent_outputs = intrinsics[-1]
        B, C, H, W = latent_outputs.shape

        pos_enc = self.pos_encoding(H, W, device=latent_outputs.device)
        pos_enc_tokens = pos_enc.flatten(2).transpose(1, 2)  # [1, H*W, C]

        light_emb = self.lighting_decoder(extrinsic) # B, C 
        x = pos_enc_tokens     # B, H*W, C

        for layer, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(
                x, intrinsics[-1 - layer].flatten(2).transpose(1, 2), light_emb
            )  # B, H*W, C

        # Apply final normalization (operate on channel dimension)
        # Reshape for LayerNorm: [B, H*W, C] -> [B, H, W, C]
        x = x.reshape(B, H, W, -1)
        x = self.final_norm(x)
        # Reshape: [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        
        # 5. Convolutional refinement head to reduce tile/blur artifacts
        output = self.conv_head(x)
        
        return output


# Factory functions for different decoder sizes
def dinov3_decoder_small(**kwargs):
    """Small decoder for ViT-Small"""
    return DINOv3Decoder(
        in_dim=384,
        hidden_dim=256,
        num_decoder_layers=3,
        num_heads=4,
        **kwargs
    )


def dinov3_decoder_base(**kwargs):
    """Base decoder for ViT-Base"""
    return DINOv3Decoder(
        in_dim=768,
        hidden_dim=512,
        num_decoder_layers=4,
        num_heads=8,
        **kwargs
    )


def dinov3_decoder_large(**kwargs):
    """Large decoder for ViT-Large"""
    return DINOv3Decoder(
        in_dim=1024,
        hidden_dim=768,
        num_decoder_layers=6,
        num_heads=12,
        **kwargs
    )