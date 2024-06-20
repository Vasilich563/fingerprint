#Author: Vodohleb04
from abc import ABC, abstractmethod
import torch
from torch import nn as nn


def conv3x3(
        in_channels: int, out_channels: int,
        stride: int = 1, groups: int = 1, dilation: int = 1, device=None, dtype=None
):
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float64
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        groups=groups,
        padding=dilation,
        dilation=dilation,
        bias=False,
        device=device,
        dtype=dtype
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1, device=None, dtype=None):
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float64
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, device=device, dtype=dtype)


class ABCResNet(nn.Module, ABC):

    @abstractmethod
    def _make_section(
            self, block, in_channels, channels_on_section, amount_of_blocks, dilation, base_width,
            stride=1, groups=1, dilate=False, device=None, dtype=None
    ):
        raise NotImplementedError

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def __init__(
            self, block, blocks_on_section, latent_dim, dropblock_conv_keep_p, dropblock_size, dropout_linear_keep_p,
            groups=1, width_per_group=64, is_v2=False, device=None, dtype=None
    ):
        if device is None:
            device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float64
        super().__init__()
        in_channels = 64
        dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.dropblock_conv_keep_p = dropblock_conv_keep_p
        self.dropblock_size = dropblock_size
        self.dropout_linear_keep_p = dropout_linear_keep_p

        self.conv1 = nn.Conv2d(
            1, in_channels, kernel_size=7, stride=2, padding=3, bias=False, device=device, dtype=dtype
        )
        if not is_v2:
            self.bn1 = nn.BatchNorm2d(in_channels, device=device, dtype=dtype)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.section1 = self._make_section(
            block, in_channels, 64, blocks_on_section[0], dilation, self.base_width,
            stride=1, groups=1, device=device, dtype=dtype
        )
        in_channels = 64 * block.expansion  # Channels on prev section * block.expansion

        self.section2 = self._make_section(
            block, in_channels, 128, blocks_on_section[1], dilation, self.base_width,
            stride=2, groups=1, device=device, dtype=dtype
        )
        in_channels = 128 * block.expansion  # Channels on prev section * block.expansion

        self.section3 = self._make_section(
            block, in_channels, 256, blocks_on_section[2], dilation, self.base_width,
            stride=2, groups=1, device=device, dtype=dtype
        )
        in_channels = 256 * block.expansion  # Channels on prev section * block.expansion

        self.section4 = self._make_section(
            block, in_channels, 512, blocks_on_section[3], dilation, self.base_width,
            stride=2, groups=1, device=device, dtype=dtype
        )
        in_channels = 512 * block.expansion  # Channels on prev section * block.expansion
        if is_v2:
            self.bn1 = nn.BatchNorm2d(in_channels, device=device, dtype=dtype)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_layer = nn.Linear(in_channels, latent_dim, device=device, dtype=dtype)

    @abstractmethod
    def forward(self, input_x):
        raise NotImplementedError
