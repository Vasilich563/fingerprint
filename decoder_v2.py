#Author: Vodohleb04
import torch
from torch import nn
from torch.nn import functional as f


class DecoderV2(nn.Module):
    def _init_symmetric_weights(self):
        nn.init.xavier_uniform_(self.conv_t5_sigmoid.weight)

    def _init_asymmetric_weights(self):
        for module in self.modules():
            if module is not self.conv_t5_sigmoid:
                if isinstance(module, nn.ConvTranspose2d):
                    nn.init.kaiming_uniform_(module.weight)
                elif isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight)
                    nn.init.trunc_normal_(module.bias, mean=0, std=0.1)
                elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)

    def init_weights(self):
        self._init_symmetric_weights()
        self._init_asymmetric_weights()

    def __init__(self, latent_dim, dropout_conv_keep_p, dropout_linear_keep_p, device=None, dtype=None):
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float64

        self.dropout_conv_keep_p = dropout_conv_keep_p
        self.dropout_linear_keep_p = dropout_linear_keep_p

        in_conv_channels = 512

        self.bn1 = nn.BatchNorm1d(latent_dim, device=device, dtype=dtype)
        self.linear1 = nn.Linear(latent_dim, 256, device=device, dtype=dtype)

        # self.linear2 = nn.Linear(1024, 256 * 6 * 6, device=device, dtype=dtype)
        # self.resize = (256, 6, 6)
        self.linear2 = nn.Linear(256, 1024, device=device, dtype=dtype)

        self.linear3 = nn.Linear(1024, in_conv_channels * 4 * 4, device=device, dtype=dtype)
        self.resize = (in_conv_channels, 4, 4)

        self.bn11 = nn.BatchNorm2d(in_conv_channels, device=device, dtype=dtype)
        self.conv_t11 = nn.ConvTranspose2d(
           in_conv_channels, in_conv_channels, 3, stride=1, padding=1, bias=False, device=device, dtype=dtype
        )
        self.bn12 = nn.BatchNorm2d(in_conv_channels, device=device, dtype=dtype)
        self.conv_t12 = nn.ConvTranspose2d(
            in_conv_channels, in_conv_channels, 5, stride=2, padding=0, bias=False, device=device, dtype=dtype
        )
        self.bn13 = nn.BatchNorm2d(in_conv_channels, device=device, dtype=dtype)
        self.conv_t13 = nn.ConvTranspose2d(
            in_conv_channels, in_conv_channels // 2, 3, stride=1, padding=1, bias=False, device=device, dtype=dtype
        )

        self.residual_conv_t1 = nn.ConvTranspose2d(
            in_conv_channels, in_conv_channels // 2, 5, stride=2, padding=0, bias=False, device=device, dtype=dtype
        )

        in_conv_channels //= 2

        self.bn21 = nn.BatchNorm2d(in_conv_channels, device=device, dtype=dtype)
        self.conv_t21 = nn.ConvTranspose2d(
            in_conv_channels, in_conv_channels, 3,
            stride=1, output_padding=0, padding=1, bias=False, device=device, dtype=dtype
        )
        self.bn22 = nn.BatchNorm2d(in_conv_channels, device=device, dtype=dtype)
        self.conv_t22 = nn.ConvTranspose2d(
            in_conv_channels, in_conv_channels, 5, stride=2, padding=0, bias=False, device=device, dtype=dtype
        )
        self.bn23 = nn.BatchNorm2d(in_conv_channels, device=device, dtype=dtype)
        self.conv_t23 = nn.ConvTranspose2d(
            in_conv_channels, in_conv_channels // 2, 3, stride=1, padding=1, bias=False, device=device, dtype=dtype
        )

        self.residual_conv_t2 = nn.ConvTranspose2d(
            in_conv_channels, in_conv_channels // 2, 5, stride=2, padding=0, bias=False, device=device, dtype=dtype
        )
        in_conv_channels //= 2

        self.bn31 = nn.BatchNorm2d(in_conv_channels, device=device, dtype=dtype)
        self.conv_t31 = nn.ConvTranspose2d(
            in_conv_channels, in_conv_channels, 3,
            stride=1, output_padding=0, padding=1, bias=False, device=device, dtype=dtype
        )
        self.bn32 = nn.BatchNorm2d(in_conv_channels, device=device, dtype=dtype)
        self.conv_t32 = nn.ConvTranspose2d(
            in_conv_channels, in_conv_channels, 5, stride=2, padding=0, bias=False, device=device, dtype=dtype
        )
        self.bn33 = nn.BatchNorm2d(in_conv_channels, device=device, dtype=dtype)
        self.conv_t33 = nn.ConvTranspose2d(
            in_conv_channels, in_conv_channels // 2, 3, stride=1, padding=1, bias=False, device=device, dtype=dtype
        )

        self.residual_conv_t3 = nn.ConvTranspose2d(
            in_conv_channels, in_conv_channels // 2, 5, stride=2, padding=0, bias=False, device=device, dtype=dtype
        )
        in_conv_channels //= 2

        self.bn41 = nn.BatchNorm2d(in_conv_channels, device=device, dtype=dtype)
        self.conv_t41 = nn.ConvTranspose2d(
            in_conv_channels, in_conv_channels, 3,
            stride=1, output_padding=0, padding=1, bias=False, device=device, dtype=dtype
        )
        self.bn42 = nn.BatchNorm2d(in_conv_channels, device=device, dtype=dtype)
        self.conv_t42 = nn.ConvTranspose2d(
            in_conv_channels, in_conv_channels, 5, stride=2, padding=0, bias=False, device=device, dtype=dtype
        )
        self.bn43 = nn.BatchNorm2d(in_conv_channels, device=device, dtype=dtype)
        self.conv_t43 = nn.ConvTranspose2d(
            in_conv_channels, in_conv_channels, 3, stride=1, padding=1, bias=False, device=device, dtype=dtype
        )

        self.residual_conv_t4 = nn.ConvTranspose2d(
            in_conv_channels, in_conv_channels, 5, stride=2, padding=0, bias=False, device=device, dtype=dtype
        )

        self.bn5_sigmoid = nn.BatchNorm2d(in_conv_channels, device=device, dtype=dtype)
        self.conv_t5_sigmoid = nn.ConvTranspose2d(
            64, 1, 3, padding=1, stride=1, bias=False, device=device, dtype=dtype
        )

    def _batch_drop_conv_relu(self, x, batch_layer, conv_layer, dropout=True):
        x = batch_layer(x)
        if dropout:
            x = f.dropout(x, p=1 - self.dropout_linear_keep_p, training=self.training)
        x = conv_layer(x)
        x = f.relu_(x)
        return x

    def compute_block(self, x, bn1, conv_t1, bn2, conv_t2, bn3, conv_t3, residual_con=None):
        shortcut = x
        # print(f"0 {out.shape}")
        out = self._batch_drop_conv_relu(x, bn1, conv_t1, dropout=False)
        out = self._batch_drop_conv_relu(out, bn2, conv_t2)
        out = self._batch_drop_conv_relu(out, bn3, conv_t3)
        # print(f"1 {out.shape}")

        if residual_con is not None:
            shortcut = residual_con(shortcut)
        shortcut = f.dropout(shortcut, p=1 - self.dropout_linear_keep_p, training=self.training)
        # print(shortcut.shape)
        return out + shortcut

    def forward(self, input_x):
        """
        Block

        BatchNorm
        relu
        Conv
        """
        out = self.bn1(input_x)
        out = f.dropout(out, p=1 - self.dropout_linear_keep_p, training=self.training)
        out = self.linear1(out)
        out = f.relu_(out)

        out = f.dropout(out, p=1 - self.dropout_linear_keep_p, training=self.training)
        out = self.linear2(out)
        out = f.relu_(out)

        out = f.dropout(out, p=1 - self.dropout_linear_keep_p, training=self.training)
        out = self.linear3(out)
        out = f.relu_(out)

        out = out.view(-1, self.resize[0], self.resize[1], self.resize[2])
        # # out = self.upsample(out)

        out = self.compute_block(
            out, self.bn11, self.conv_t11, self.bn12, self.conv_t12, self.bn13, self.conv_t13, self.residual_conv_t1
        )

        out = self.compute_block(
            out, self.bn21, self.conv_t21, self.bn22, self.conv_t22, self.bn23, self.conv_t23, self.residual_conv_t2
        )

        out = self.compute_block(
            out, self.bn31, self.conv_t31, self.bn32, self.conv_t32, self.bn33, self.conv_t33, self.residual_conv_t3
        )

        out = self.compute_block(
            out, self.bn41, self.conv_t41, self.bn42, self.conv_t42, self.bn43, self.conv_t43, self.residual_conv_t4
        )

        out = self.conv_t5_sigmoid(out)
        out = f.sigmoid(out)
        # print(f"out {out.shape}")
        return out


def create_decoder_v2(latent_dim, dropout_conv_keep_p=0.8, dropout_linear_keep_p=0.5, device=None, dtype=None):
    decoder = DecoderV2(latent_dim, dropout_conv_keep_p, dropout_linear_keep_p, device=device, dtype=dtype)
    decoder.init_weights()
    decoder = decoder.to(device)
    return decoder


