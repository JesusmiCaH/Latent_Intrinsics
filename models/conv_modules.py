import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = False,
        dropout: float = 0.0,
        norm: Optional[nn.Module] = nn.BatchNorm2d,
        act_func: Optional[Callable] = nn.ReLU,
    ):
        super(ConvLayer, self).__init__()
        
        if padding is None:
            padding = kernel_size // 2
            if stride == 2:
                padding = 1 # Match typical stride 2 behavior

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
        )
        self.norm = norm(out_channels) if norm else nn.Identity()
        self.act = act_func() if act_func else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class ConvPixelShuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        factor: int = 2,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels * out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_shuffle(x, self.factor)
        return x


class InterpolateConvUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        factor: int = 2,
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale factor upsampling
        x = F.interpolate(x, scale_factor=self.factor, mode=self.mode)
        x = self.conv(x)
        return x
