#Author: Vodohleb04
import torch
from torch import nn as nn
from torch.nn import functional as f
from abstract_res_net import ABCResNet, conv1x1, conv3x3


class BottleneckV2(nn.Module):
    """
        Residual block of ResNet-50:
            conv1x1->
            conv3x3->
            conv1x1->
            +=input
    """
    expansion = 4

    def __init__(
            self, in_channels, out_channels, dropout_conv_keep_p,
            stride=1, downsample=None, groups=1, base_width=64, dilation=1, device=None, dtype=None
    ):
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float64

        super().__init__()

        width = int(out_channels * (base_width / 64)) * groups

        self.bn1 = nn.BatchNorm2d(in_channels, device=device, dtype=dtype)
        self.conv1 = conv1x1(in_channels, width, device=device, dtype=dtype)

        self.bn2 = nn.BatchNorm2d(width, device=device, dtype=dtype)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, device=device, dtype=dtype)

        self.bn3 = nn.BatchNorm2d(width, device=device, dtype=dtype)
        self.conv3 = conv1x1(width, out_channels * self.expansion, device=device, dtype=dtype)

        self.downsample = downsample
        self.stride = stride

        self.dropout_conv_p = dropout_conv_keep_p

    def forward(self, input_x):
        # out = self.conv1(
        #     f.relu_(
        #         self.bn1(input_x)
        #     )
        # )
        #
        # out = self.conv2(
        #     f.relu_(
        #         self.bn2(out)
        #     )
        # )
        #
        # out = f.relu_(
        #     self.bn3(
        #         out
        #     )
        # )
        #
        # out = f.dropout(out, 1 - self.dropout_conv_keep_p)
        # out = self.conv3(out)
        #
        # if self.downsample is not None:
        #     input_x = self.downsample(input_x)
        #
        # return out + input_x
        return (
            input_x if self.downsample is None else self.downsample(input_x)
            +
            self.conv3(
                f.dropout(
                    f.relu_(
                        self.bn3(
                            self.conv2(
                                f.relu_(
                                    self.bn2(
                                        self.conv1(
                                            f.relu_(
                                                self.bn1(
                                                    input_x
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    ),
                    p=1 - self.dropout_conv_p
                )
            )
        )


class ResNetV2(ABCResNet):

    def _make_section(
            self, block, in_channels, channels_on_section, amount_of_blocks, dilation, base_width,
            stride=1, groups=1, dilate=False, device=None, dtype=None
    ):
        downsample = None
        previous_dilation = dilation
        if dilate:
            dilation *= stride
            stride = 1
        if stride != 1 or in_channels != channels_on_section * block.expansion:
            downsample = nn.Sequential(
                nn.BatchNorm2d(in_channels, device=device, dtype=dtype),
                conv1x1(in_channels, channels_on_section * block.expansion, stride)
            )

        blocks_of_section = nn.ModuleList()
        blocks_of_section.append(
            block(
                in_channels, channels_on_section, self.dropout_conv_keep_p,
                stride, downsample, groups, base_width, previous_dilation, device, dtype
            )
        )
        in_channels = channels_on_section * block.expansion
        for _ in range(1, amount_of_blocks):
            blocks_of_section.append(
                block(
                    in_channels, channels_on_section, self.dropout_conv_keep_p,
                    groups=groups, base_width=base_width, dilation=dilation, device=device, dtype=dtype
                )
            )
        return nn.Sequential(*blocks_of_section)

    def __init__(
            self, block, blocks_on_section, latent_dim, dropout_conv_keep_p, dropout_linear_keep_p,
            groups=1, width_per_group=64, device=None, dtype=None
    ):
        self.dropout_conv_keep_p = dropout_conv_keep_p
        self.dropout_linear_keep_p = dropout_linear_keep_p
        super().__init__(block, blocks_on_section, latent_dim, groups, width_per_group, device, dtype)

    def forward(self, input_x):
        # input_x = f.relu_(
        #     self.bn1(
        #         self.conv1(input_x)
        #     )
        # )
        # input_x = self.max_pool(input_x)
        #
        # input_x = self.section1(input_x)
        # input_x = self.section2(input_x)
        # input_x = self.section3(input_x)
        # input_x = self.section4(input_x)
        #
        # input_x = f.dropout(input_x, 1 - self.dropout_linear_keep_p)
        # input_x = self.avg_pool(input_x)
        #
        # input_x = torch.flatten(input_x, 1)
        # input_x = self.linear_layer(input_x)
        #
        # return input_x
        return self.linear_layer(
            torch.flatten(
                self.avg_pool(
                    f.dropout(
                        self.section4(
                            self.section3(
                                self.section2(
                                    self.section1(
                                        self.max_pool(
                                            f.relu_(
                                                self.bn1(
                                                    self.conv1(input_x)
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        ),
                        p=1 - self.dropout_linear_keep_p
                    )
                )
            )
        )


def res_net_50_v2(latent_dim, dropout_conv_keep_p=0.8, dropout_linear_keep_p=0.5, device=None, dtype=None):
    network = ResNetV2(
        BottleneckV2, [3, 4, 6, 3], latent_dim,
        dropout_conv_keep_p=dropout_conv_keep_p, dropout_linear_keep_p=dropout_linear_keep_p, device=device, dtype=dtype
    )
    network.init_weights()
    network = network.to(device)
    return network

