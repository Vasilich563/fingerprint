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

        self.bn1 = nn.BatchNorm1d(latent_dim, device=device, dtype=dtype)
        self.linear1 = nn.Linear(latent_dim, 1024, device=device, dtype=dtype)

        self.linear2 = nn.Linear(1024, 256 * 6 * 6, device=device, dtype=dtype)
        self.resize = (256, 6, 6)

        self.upsample = nn.Upsample(scale_factor=2)

        self.conv_t1 = nn.ConvTranspose2d(
            256, 512, 3, stride=1, bias=False, device=device, dtype=dtype
        )
        self.conv_t2 = nn.ConvTranspose2d(
            512, 256, 3, stride=1, bias=False, device=device, dtype=dtype
        )
        self.conv_t3 = nn.ConvTranspose2d(
            256, 128, 5,
            stride=3, output_padding=1, padding=0, bias=False, device=device, dtype=dtype
        )
        self.conv_t4 = nn.ConvTranspose2d(
            128, 64, 3, stride=2, padding=0, bias=False, device=device, dtype=dtype
        )
        self.conv_t5_sigmoid = nn.ConvTranspose2d(
            64, 1, 3, stride=1, bias=False, device=device, dtype=dtype
        )

    def forward(self, input_x):
        # out = self.bn1(input_x)
        # out = self.linear1(out)
        # out = f.relu_(out)
        #
        # out = f.dropout(out, 1 - self.dropout_linear_keep_p)
        # out = self.linear2(out)
        # out = f.relu_(out)
        #
        # out = f.dropout(out, 1 - self.dropout_conv_keep_p)
        # out = out.view(-1, self.resize[0], self.resize[1], self.resize[2])
        # out = self.upsample(out)
        # out = self.conv_t1(out)
        # out = f.relu_(out)
        #
        # out = f.dropout(out, 1 - self.dropout_conv_keep_p)
        # out = self.conv_t2(out)
        # out = f.relu_(out)
        #
        # out = f.dropout(out, 1 - self.dropout_conv_keep_p)
        # out = f.relu_(
        #     self.conv_t3(out)
        # )
        #
        # out = f.dropout(out, 1 - self.dropout_conv_keep_p)
        # out = self.conv_t4(out)
        # out = f.relu_(out)
        #
        # out = self.conv_t5_sigmoid(out)
        # out = f.sigmoid(out)
        #
        # return out
        return f.sigmoid(
            self.conv_t5_sigmoid(
                f.relu_(
                    self.conv_t4(
                        f.dropout(
                            f.relu_(
                                self.conv_t3(
                                    f.dropout(
                                        f.relu_(
                                            self.conv_t2(
                                                f.dropout(
                                                    f.relu_(
                                                        self.conv_t1(
                                                            self.upsample(
                                                                f.dropout(
                                                                    f.relu_(
                                                                        self.linear2(
                                                                            f.dropout(
                                                                                f.relu_(
                                                                                    self.linear1(
                                                                                        self.bn1(input_x)
                                                                                    )
                                                                                ),
                                                                                p=1 - self.dropout_linear_keep_p
                                                                            )
                                                                        )
                                                                    ),
                                                                    p=1 - self.dropout_conv_keep_p
                                                                ).view(-1, self.resize[0], self.resize[1], self.resize[2])
                                                            )
                                                        )
                                                    ),
                                                    p=1 - self.dropout_conv_keep_p
                                                )
                                            )
                                        ),
                                        p=1 - self.dropout_conv_keep_p
                                    )
                                )
                            ),
                            p=1 - self.dropout_conv_keep_p
                        )
                    )
                )
            )
        )


def create_decoder_v2(latent_dim, dropout_conv_keep_p=0.8, dropout_linear_keep_p=0.5, device=None, dtype=None):
    decoder = DecoderV2(latent_dim, dropout_conv_keep_p, dropout_linear_keep_p, device=device, dtype=dtype)
    decoder.init_weights()
    decoder = decoder.to(device)
    return decoder


