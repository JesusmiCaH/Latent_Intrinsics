import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- small building blocks ----------

class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, gn_groups=16):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.GroupNorm(num_groups=min(gn_groups, out_ch), num_channels=out_ch),
            nn.GELU()
        )
    def forward(self, x):
        return self.block(x)

class FiLM(nn.Module):
    """Feature-wise linear modulation from a latent code."""
    def __init__(self, latent_dim, num_channels):
        super().__init__()
        self.gamma = nn.Linear(latent_dim, num_channels)
        self.beta  = nn.Linear(latent_dim, num_channels)
    def forward(self, x, z):  # x: [B,C,H,W], z: [B,D]
        # print("🙉", x.shape, z.shape)
        g = self.gamma(z).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        b = self.beta(z).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + g) + b

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # bilinear upsample + 1x1 align then concat skip -> convs
        self.align = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.fuse  = nn.Sequential(
            ConvGNAct(out_ch + skip_ch, out_ch),
            ConvGNAct(out_ch, out_ch)
        )
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.align(x)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)

# ---------- ViT Decoder that builds a pyramid from intermediate layers ----------
class DINOv3Decoder(nn.Module):
    """
    Build a 4-level pyramid from ViT features (all at same HxW),
    then a UNet-like top-down path with FiLM modulation from extrinsic code.
    """
    def __init__(
        self,
        in_dim=768,
        pyramid_channels=(64, 128, 256, 384),
        out_channels=3,
        extri_latent_dim=256,              # dim of extrinsic (lighting) code
        smooth_tail=True
    ):
        super().__init__()
        P1, P2, P3, P4 = pyramid_channels  # high->low resolutions during decode

        # Project each chosen ViT layer to a target channel count
        self.pj4 = nn.Conv2d(in_dim, P4, kernel_size=1, bias=False)  # "deepest"
        self.pj3 = nn.Conv2d(in_dim, P3, kernel_size=1, bias=False)
        self.pj2 = nn.Conv2d(in_dim, P2, kernel_size=1, bias=False)
        self.pj1 = nn.Conv2d(in_dim, P1, kernel_size=1, bias=False)  # "shallowest"

        # After projection, we resize to a canonical multi-scale pyramid:
        # assume encoder feat map size = HxW (e.g., 14x14 for patch=16).
        # We'll form target sizes [H/8, H/4, H/2, H] by repeated upsample.
        # (works even if H is small; change to taste.)
        self.post4 = ConvGNAct(P4, P4)
        self.post3 = ConvGNAct(P3, P3)
        self.post2 = ConvGNAct(P2, P2)
        self.post1 = ConvGNAct(P1, P1)

        # FiLM per stage
        self.film4 = FiLM(extri_latent_dim, P4)
        self.film3 = FiLM(extri_latent_dim, P3)
        self.film2 = FiLM(extri_latent_dim, P2)
        self.film1 = FiLM(extri_latent_dim, P1)

        # Top-down decode with skip fusions
        self.up34 = UpBlock(P4, P3, P3)  # 4 -> 3
        self.up23 = UpBlock(P3, P2, P2)  # 3 -> 2
        self.up12 = UpBlock(P2, P1, P1)  # 2 -> 1

        # Final "de-tokenize" head: a small conv tail to smooth/blocky edges
        tail = [
            ConvGNAct(P1, P1),
            nn.Conv2d(P1, out_channels, kernel_size=3, padding=1)
        ]
        if smooth_tail:
            # light smoothing: 3x3 -> GELU -> 3x3
            tail = [
                ConvGNAct(P1, P1),
                nn.Conv2d(P1, P1, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(P1, out_channels, kernel_size=3, padding=1)
            ]
        self.tail = nn.Sequential(*tail)

    def _target_sizes(self, H, W):
        # choose a gentle pyramid; if H=W=14 -> sizes: 4->7->14->14 (reasonable).
        s4 = (H * 2, W * 2)  # "deepest"
        s3 = (H * 4, W * 4)                             # next
        s2 = (H * 8, W * 8)
        s1 = (H * 16, W * 16)
        return s4, s3, s2, s1

    def forward(self, intrinsics, extrinsic):
        """
        intrinsics: list of ViT feature maps [B, C=768, H, W], from shallow->deep or deep->shallow.
        We’ll pick 4 layers: [l1,l2,l3,l4] (shallow->deep). If you pass more, we sample 4 evenly.
        extrinsic: [B, latent_dim]
        """
        assert len(intrinsics) >= 4, "Pass >=4 intermediate layers (set n>=4 in get_intermediate_layers)."
        # pick 4 roughly evenly spaced layers, shallow->deep
        idxs = torch.linspace(0, len(intrinsics)-1, 4).round().long().tolist()
        f1, f2, f3, f4 = [intrinsics[i] for i in idxs]  # each [B,768,H,W]
        B, _, H, W = f1.shape
        s4, s3, s2, s1 = self._target_sizes(H, W)

        # project & resize into a pyramid
        p4 = F.interpolate(self.pj4(f4), size=s4, mode='bilinear', align_corners=False)  # deepest/smallest
        p3 = F.interpolate(self.pj3(f3), size=s3, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.pj2(f2), size=s2, mode='bilinear', align_corners=False)
        p1 = F.interpolate(self.pj1(f1), size=s1, mode='bilinear', align_corners=False)  # largest

        # per-stage normalization + FiLM modulation
        p4 = self.film4(self.post4(p4), extrinsic)
        p3 = self.film3(self.post3(p3), extrinsic)
        p2 = self.film2(self.post2(p2), extrinsic)
        p1 = self.film1(self.post1(p1), extrinsic)

        # top-down decode with skip concatenations
        x = p4                       # smallest
        x = self.up34(x, p3)         # -> size of p3
        x = self.up23(x, p2)         # -> size of p2
        x = self.up12(x, p1)         # -> size of p1 (largest)

        # final smoothing tail to 3 channels
        out = self.tail(x)           # [B,3,H_out,W_out] == p1 spatial size
        return out
