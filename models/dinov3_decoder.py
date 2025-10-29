import torch
from torch import nn
from layers import ConvModule, UpConvHead, FeatureFusionBlock

class DINOv3Decoder(nn.Module):
    def __init__(
            self, 
            in_channels = (1024, 1024, 1024, 1024),     
            pyramid_channels = [128, 256, 512, 1024],
            post_process_channel = 256,
            out_channel = 3,
            readout_type = "project",
            use_batchnorm = False,
        ):
        super().__init__()
        self.readout_type = readout_type

        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            for channel in in_channels:
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2*channel, channel),
                        nn.GELU(),
                    )
                )
        # BatchNorm layers
        self.batchnorm_layers = nn.ModuleList(
            [
                nn.BatchNorm2d(num_features=channel) if use_batchnorm else nn.Identity()
                for channel in in_channels
            ]
        )
        # Project into Conv Receptive Fields
        self.projects = nn.ModuleList(
            [
                ConvModule(
                    in_channels=in_channels[idx], 
                    out_channels=out_channel,
                    kernel_size=1,
                    act_cfg = None,
                )
                for idx, out_channel in enumerate(pyramid_channels)
            ]
        )

        # Build pyramid fusion blocks
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=pyramid_channels[0], out_channels=pyramid_channels[0], kernel_size=4, stride=4, padding=0,
                ), 
                nn.ConvTranspose2d(
                    in_channels=pyramid_channels[1], out_channels=pyramid_channels[1], kernel_size=2, stride=2, padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=pyramid_channels[3], out_channels=pyramid_channels[3], kernel_size=3, stride=2, padding=1,
                ),
            ]
        )

        # Post-processing on pyramid features
        self.post_convs = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()
        self.act_cfg = {"type": "ReLU"}
        self.norm_cfg = None  # TODO CHECK THIS
        for channel in pyramid_channels:
            self.post_convs.append(
                ConvModule(
                    in_channels=channel,
                    out_channels=post_process_channel,
                    kernel_size=3,
                    padding=1,
                    act_cfg=None,
                )
            )
            self.fusion_blocks.append(
                FeatureFusionBlock(
                    in_channels=post_process_channel,
                    act_cfg=self.act_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )
        self.fusion_blocks[0].res_conv_unit1 = None

        # Final projection and upsampling
        self.final_project = ConvModule(
            in_channels=post_process_channel, out_channels=post_process_channel, kernel_size=3, padding=1, norm_cfg=self.norm_cfg
        )
        self.upconv_head = UpConvHead(post_process_channel, out_channel, n_hidden_channels=32)

    def forward(self, intrinsics, extrinsic):
        # Build pyramid features
        pyramid_features = []
        for idx, vis_tokens in enumerate(intrinsics):
            # intrinsics: List of [vis_tokens]
            vis_token_shape = vis_tokens.shape      # B, C, H, W
            # vis_tokens interact with extrinsic
            if self.readout_type == "project":
                x = vis_tokens.flatten(2).transpose(1, 2)   # B, H*W, C
                readout = extrinsic.unsqueeze(1).expand_as(x)   # B, H*W, C
                x = self.readout_projects[idx](torch.cat([x, readout], dim=-1))   # B, H*W, C
                x = x.transpose(1, 2).reshape(vis_token_shape)   # B, C, H, W
            elif self.readout_type == "add":
                x = vis_tokens.flatten(2) + extrinsic.unsqueeze(-1)   # B, C, H*W
                x = x.reshape(vis_token_shape)   # B, C, H, W
            elif self.readout_type == "product":
                # TODO: apply constraint scaling here!!!!!!
                x = vis_tokens.flatten(2) * extrinsic.unsqueeze(-1)   # B, C, H*W
                x = x.reshape(vis_token_shape)   # B, C, H, W
            else:
                pass
            x = self.batchnorm_layers[idx](x)
            x = self.projects[idx](x)
            x = self.resize_layers[idx](x)
            pyramid_features.append(x)
        
        # Fuse pyramid features in reverse order
        for i in range(len(pyramid_features)):
            feature = self.post_convs[-i-1](pyramid_features[-i-1])
            print("🍓feature shape:", i, feature.shape)
            if i == 0:
                x = self.fusion_blocks[i](feature)
            else:
                x = self.fusion_blocks[i](x, feature)
        print("🫐fused feature shape:", x.shape)
        x = self.final_project(x)
        out = self.upconv_head(x)
        return out