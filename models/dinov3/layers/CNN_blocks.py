import torch
from torch import nn


class ConvModule(nn.Module):
    """A Unit of CNN Layer."""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            activation=nn.ReLU,
            norm=nn.BatchNorm2d,
            norm_cfg: dict = {},
            is_transpose: bool = False,
            upsample_factor: int = 2,
            order: tuple = ('conv', 'norm', 'act'),
            **kwargs,
    ):
        super().__init__()

        conv_func = nn.ConvTranspose2d if is_transpose else nn.Conv2d

        self.conv = conv_func(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs
        )

        if norm == nn.BatchNorm2d:
            self.norm = norm(out_channels, **norm_cfg)
        elif norm == nn.GroupNorm:
            self.norm = norm(num_channels=out_channels, **norm_cfg)
        else:
            self.norm = nn.Identity()
        
        if activation is not None:
            try:
                self.act = activation(inplace=True)
            except TypeError:
                self.act = activation()
        else:
            self.act = nn.Identity()
        
        self.upsample_factor = upsample_factor
        self.order = order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample_factor > 1:
            x = nn.functional.interpolate(
                x,
                scale_factor=self.upsample_factor,
                mode='bilinear',
                align_corners=False,
            )
            
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm':
                x = self.norm(x)
            elif layer == 'act':
                x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    """Residual Block with two ConvModules."""
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            activation=nn.ReLU,
            num_groups: int = 32,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, hidden_channels)
        self.act1 = activation() if activation is not None else nn.Identity()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

        self.final_act = activation() if activation is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x += residual
        x = self.final_act(x)
        return x