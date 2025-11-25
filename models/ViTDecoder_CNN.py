import torch
from torch import nn
import torch.nn.functional as F
from Latent_Intrinsics.layers.CNN_blocks import ConvModule


class PreActResidualConvUnit(nn.Module):
    """ResidualConvUnit, pre-activate residual unit.
    Args:
        in_channels (int): number of channels in the input feature map.
        act_cfg (dict): dictionary to construct and config activation layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        stride (int): stride of the first block. Default: 1
        dilation (int): dilation rate for convs layers. Default: 1.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(self, in_channels, act, norm, stride=1, dilation=1, init_cfg=None):
        super().__init__()  # init_cfg)
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            norm=norm,
            activation=act,
            bias=False,
            order=("act", "conv", "norm"),
        )
        self.conv2 = ConvModule(
            in_channels,
            in_channels,
            3,
            padding=1,
            norm=norm,
            activation=act,
            bias=False,
            order=("act", "conv", "norm"),
        )

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x + inputs

class FeatureFusionBlock(nn.Module):
    """FeatureFusionBlock, merge feature map from different stages.
    Args:
        in_channels (int): Input channels.
        act_cfg (dict): The activation config for ResidualConvUnit.
        norm_cfg (dict): Config dict for normalization layer.
        expand (bool): Whether expand the channels in post process block.
            Default: False.
        align_corners (bool): align_corner setting for bilinear upsample.
            Default: True.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(self, in_channels, act, norm, expand=False, align_corners=True, init_cfg=None):
        super(FeatureFusionBlock, self).__init__()  # init_cfg)
        self.in_channels = in_channels
        self.expand = expand
        self.align_corners = align_corners
        self.out_channels = in_channels
        if self.expand:
            self.out_channels = in_channels // 2
        self.project = ConvModule(self.in_channels, self.out_channels, kernel_size=1)
        self.res_conv_unit1 = PreActResidualConvUnit(in_channels=self.in_channels, act=act, norm=norm)
        self.res_conv_unit2 = PreActResidualConvUnit(in_channels=self.in_channels, act=act, norm=norm)

    def forward(self, *inputs):
        x = inputs[0]

        if len(inputs) == 2:
            if x.shape != inputs[1].shape:
                res = torch.nn.functional.interpolate(
                    inputs[1],
                    size=(x.shape[2], x.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)  # ok

        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)

        x = self.project(x)  # ok
        return x

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x
    
class ViTDecoder(nn.Module):
    def __init__(
            self, 
            in_dim = 1024,
            extrinsic_dim = 16,
            pyramid_dims = [128, 256, 512, 1024],
            post_process_channel = 256,
            out_channel = 3,
            readout_type = "project",
        ):
        super().__init__()
        self.readout_type = readout_type

        self.lighting_decoder = nn.Linear(extrinsic_dim, in_dim)

        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            for idx in range(len(pyramid_dims)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2*in_dim, in_dim),
                        nn.GELU(),
                    )
                )

        # Build pyramid blocks
        conv_settings = {
            4: {"is_transpose": True, "kernel_size": 8, "stride": 4, "padding": 2},
            2: {"is_transpose": True, "kernel_size": 4, "stride": 2, "padding": 1},
            1: {"is_transpose": False, "kernel_size": 3, "stride": 1, "padding": 1},
            0.5: {"is_transpose": False, "kernel_size": 3, "stride": 2, "padding": 1},
        }
        self.pyramid_layers = nn.ModuleList()
        for idx, ch in enumerate(pyramid_dims):
            self.pyramid_layers.append(nn.Sequential(
                # initial conv to adjust channels
                ConvModule(
                    in_channels=in_dim,
                    out_channels=ch,
                    activation=nn.GELU,
                    **conv_settings[1],
                ),
                # Upsample or Downsample
                ConvModule(
                    in_channels=ch,
                    out_channels=ch,
                    **conv_settings[2 ** (len(pyramid_dims) - 2 - idx)]
                ),
                # post-process to unify channels
                ConvModule(
                    in_channels=ch,
                    out_channels=post_process_channel,
                    **conv_settings[1],
                ))
                )
            
        # Build fusion blocks
        self.fusion_blocks = nn.ModuleList()
        for _ in pyramid_dims:
            self.fusion_blocks.append(
                FeatureFusionBlock(
                    in_channels=post_process_channel,
                    act=nn.ReLU,
                    norm=nn.BatchNorm2d,
                )
            )
        # self.fusion_blocks[0].res_conv_unit1 = None

        self.upconv_head = nn.Sequential(
            ConvModule(post_process_channel, post_process_channel, kernel_size=3, padding=1),
            ConvModule(post_process_channel, post_process_channel//2),
            Interpolate(scale_factor=2, mode='bilinear', align_corners=True),
            ConvModule(post_process_channel//2, post_process_channel//4),
            ConvModule(post_process_channel//4, out_channel, activation=None, norm=None),
        )


    def forward(self, intrinsics, extrinsic):
        lighting_emb = self.lighting_decoder(extrinsic)   # B, C
        # Build pyramid features
        pyramid_features = []
        for idx, vis_tokens in enumerate(intrinsics):
            # intrinsics: List of [vis_tokens]
            vis_token_shape = vis_tokens.shape      # B, C, H, W
            # vis_tokens interact with extrinsic
            if self.readout_type == "project":
                x = vis_tokens.flatten(2).transpose(1, 2)   # B, H*W, C
                readout = lighting_emb.unsqueeze(1).expand_as(x)   # B, H*W, C
                x = self.readout_projects[idx](torch.cat([x, readout], dim=-1))   # B, H*W, C
                x = x.transpose(1, 2).reshape(vis_token_shape)   # B, C, H, W
            elif self.readout_type == "add":
                x = vis_tokens.flatten(2) + lighting_emb.unsqueeze(-1)   # B, C, H*W
                x = x.reshape(vis_token_shape)   # B, C, H, W
            elif self.readout_type == "product":
                # TODO: apply constraint scaling here!!!!!!
                x = vis_tokens.flatten(2) * lighting_emb.unsqueeze(-1)   # B, C, H*W
                x = x.reshape(vis_token_shape)   # B, C, H, W
            else:
                pass
            x = F.normalize(x, dim=1)   # normalize after interaction

            x = self.pyramid_layers[idx](x)
            pyramid_features.append(x)
        
        # Fuse pyramid features in reverse order
        for i, feature in enumerate(pyramid_features):
            print(f"🫐feature {i} shape:", feature.shape)
            if i == 0:
                y = self.fusion_blocks[i](feature)
            else:
                y = self.fusion_blocks[i](y, feature)
        print("🫐fused feature shape:", y.shape)

        out = self.upconv_head(y)
        return out