#Author: Vodohleb04
import torch
from torch import nn as nn
from torch.nn import functional as f
from torchvision.ops import drop_block2d
from res_net.abstract_res_net import ABCResNet, conv1x1, conv3x3


class Bottleneck(nn.Module):
    """Residual block of ResNet-50,
                conv1x1->
                conv3x3->
                conv1x1->
                +=input
        """
    expansion = 4

    def __init__(
            self, in_channels, out_channels, dropblock_conv_keep_p, dropblock_size,
            stride=1, downsample=None, groups=1, base_width=64, dilation=1, device=None, dtype=None
    ):
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float64

        super().__init__()

        width = int(out_channels * (base_width / 64)) * groups

        self.conv1 = conv1x1(in_channels, width, device=device, dtype=dtype)
        self.bn1 = nn.BatchNorm2d(width, device=device, dtype=dtype)

        self.conv2 = conv3x3(width, width, stride, groups, dilation, device=device, dtype=dtype)
        self.bn2 = nn.BatchNorm2d(width, device=device, dtype=dtype)

        self.conv3 = conv1x1(width, out_channels * self.expansion, device=device, dtype=dtype)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion, device=device, dtype=dtype)

        self.downsample = downsample
        self.stride = stride

        self.dropblock_conv_keep_p = dropblock_conv_keep_p
        self.dropblock_size = dropblock_size

    def forward(self, input_x):
        print(self.training)
        out = self.conv1(input_x)
        out = f.leaky_relu_(
            self.bn1(out)
        )
        out = drop_block2d(out, p=1 - self.dropblock_conv_keep_p, block_size=self.dropblock_size, training=self.training)

        out = self.conv2(out)
        out = f.leaky_relu_(
            self.bn2(out)
        )
        out = drop_block2d(out, p=1 - self.dropblock_conv_keep_p, block_size=self.dropblock_size, training=self.training)

        out = self.conv3(out)
        out = self.bn3(out)
        out = drop_block2d(out, p=1 - self.dropblock_conv_keep_p, block_size=self.dropblock_size, training=self.training)

        if self.downsample is not None:
            input_x = self.downsample(input_x)

        input_x = drop_block2d(input_x, p=1 - self.dropblock_conv_keep_p, block_size=self.dropblock_size, training=self.training)

        return f.leaky_relu_(out + input_x)


class ResNet(ABCResNet):

    def __init__(
            self, block, blocks_on_section, latent_dim, dropblock_conv_keep_p, dropblock_sizes, dropout_linear_keep_p,
            groups=1, width_per_group=64, device=None, dtype=None
    ):
        super().__init__(
            block, blocks_on_section, latent_dim, dropblock_conv_keep_p, dropblock_sizes, dropout_linear_keep_p,
            groups=groups, width_per_group=width_per_group, is_v2=False, device=device, dtype=dtype
        )

    def _make_section(
            self, block, in_channels, channels_on_section, amount_of_blocks, dropblock_size, dilation, base_width,
            stride=1, groups=1, dilate=False, device=None, dtype=None
    ):
        downsample = None
        previous_dilation = dilation
        if dilate:
            dilation *= stride
            stride = 1
        if stride != 1 or in_channels != channels_on_section * block.expansion:
            downsample = nn.Sequential(
                conv1x1(in_channels, channels_on_section * block.expansion, stride, device=device, dtype=dtype),
                nn.BatchNorm2d(channels_on_section * block.expansion, device=device, dtype=dtype)
            )

        blocks_of_section = nn.ModuleList()
        blocks_of_section.append(
            block(
                in_channels, channels_on_section, self.dropblock_conv_keep_p, dropblock_size, stride,
                downsample, groups, base_width, previous_dilation, device=device, dtype=dtype
            )
        )
        in_channels = channels_on_section * block.expansion
        for _ in range(1, amount_of_blocks):
            blocks_of_section.append(
                block(
                    in_channels, channels_on_section, self.dropblock_conv_keep_p, dropblock_size,
                    groups=groups, base_width=base_width, dilation=dilation, device=device, dtype=dtype
                )
            )
        return nn.Sequential(*blocks_of_section)

    def forward(self, input_x):
        print(self.training)
        input_x = f.leaky_relu_(
            self.bn1(
                self.conv1(input_x)
            )
        )
        input_x = self.max_pool(input_x)

        input_x = self.section1(input_x)
        input_x = self.section2(input_x)
        input_x = self.section3(input_x)
        input_x = self.section4(input_x)

        input_x = self.avg_pool(input_x)

        input_x = torch.flatten(input_x, 1)
        input_x = f.dropout(input_x, p=1 - self.dropout_linear_keep_p, training=self.training)
        input_x = self.linear_layer(input_x)

        return input_x


def res_net_50(
        output_features, dropblock_conv_keep_p=0.8, dropblock_sizes=None, dropout_linear_keep_p=0.5, device=None, dtype=None
):
    if dropblock_sizes is None:
        dropblock_sizes = [3, 3, 3, 3]
    if len(dropblock_sizes) != 4:
        raise ValueError("4 dropblock sizes are required")
    network = ResNet(
        Bottleneck, [3, 4, 6, 3], output_features, dropblock_conv_keep_p, dropblock_size, dropout_linear_keep_p,
        device=device, dtype=dtype
    )
    network.init_weights()
    network = network.to(device)
    return network

