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

        # self.linear2 = nn.Linear(1024, 256 * 6 * 6, device=device, dtype=dtype)
        # self.resize = (256, 6, 6)
        self.linear2 = nn.Linear(1024, 512 * 4 * 4, device=device, dtype=dtype)
        self.resize = (512, 4, 4)

        self.conv_t0 = nn.ConvTranspose2d(
           512, 512, 3, stride=1, bias=False, device=device, dtype=dtype
        )
        self.conv_t1 = nn.ConvTranspose2d(
            512, 512, 3, stride=1, bias=False, device=device, dtype=dtype
        )

        # self.upsample = nn.Upsample(scale_factor=2)
        # self.conv_t1 = nn.ConvTranspose2d(
        #     256, 512, 3, stride=1, bias=False, device=device, dtype=dtype
        # )

        self.conv_t2 = nn.ConvTranspose2d(
            512, 256, 5, stride=2, padding=1, bias=False, device=device, dtype=dtype
        )
        self.conv_t3 = nn.ConvTranspose2d(
            256, 128, 7,
            stride=3, output_padding=0, padding=1, bias=False, device=device, dtype=dtype
        )
        self.conv_t4 = nn.ConvTranspose2d(
            128, 64, 5, stride=2, padding=1, bias=False, device=device, dtype=dtype
        )
        self.conv_t5_sigmoid = nn.ConvTranspose2d(
            64, 1, 3, stride=1, bias=False, device=device, dtype=dtype
        )

    def forward(self, input_x):
        # out = self.bn1(input_x)
        # out = self.linear1(out)
        # out = f.relu_(out)
        #
        # out = f.dropout(out, p=1 - self.dropout_linear_keep_p, training=self.training)
        # out = self.linear2(out)
        # out = f.relu_(out)
        #
        # out = out.view(-1, self.resize[0], self.resize[1], self.resize[2])
        # out = f.dropout(out, p=1 - self.dropout_conv_keep_p, training=self.training)
        # # out = self.upsample(out)
        #
        # out = f.dropout(out, p=1 - self.dropout_conv_keep_p, training=self.training)
        # out = self.conv_t0(out)
        # print(f"0 {out.shape}")
        #
        # out = f.dropout(out, p=1 - self.dropout_conv_keep_p, training=self.training)
        # out = f.relu_(out)
        # out = self.conv_t1(out)
        # print(f"1 {out.shape}")
        #
        # out = f.dropout(out, p=1 - self.dropout_conv_keep_p, training=self.training)
        # out = f.relu_(out)
        # out = self.conv_t2(out)
        # print(f"2 {out.shape}")
        #
        # out = f.dropout(out, p=1 - self.dropout_conv_keep_p, training=self.training)
        # out = f.relu_(out)
        # out = self.conv_t3(out)
        # print(f"3 {out.shape}")
        #
        # out = f.dropout(out, p=1 - self.dropout_conv_keep_p, training=self.training)
        # out = f.relu_(out)
        # out = self.conv_t4(out)
        # print(f"4 {out.shape}")
        #
        # out = self.conv_t5_sigmoid(out)
        # print(f"5 {out.shape}")
        # out = f.sigmoid(out)
        # return out

        return f.sigmoid(
            self.conv_t5_sigmoid(
                self.conv_t4(
                    f.relu_(
                        f.dropout(
                            self.conv_t3(
                                f.relu_(
                                    f.dropout(
                                        self.conv_t2(
                                            f.relu_(
                                                f.dropout(
                                                    self.conv_t1(
                                                        f.relu_(
                                                            f.dropout(
                                                                self.conv_t0(
                                                                    f.dropout(
                                                                        f.relu_(
                                                                            self.linear2(
                                                                                f.dropout(
                                                                                    f.relu_(
                                                                                        self.linear1(
                                                                                            self.bn1(input_x)
                                                                                        )
                                                                                    ),
                                                                                    p=1 - self.dropout_linear_keep_p,
                                                                                    training=self.training
                                                                                )
                                                                            )
                                                                        ).view(-1, self.resize[0], self.resize[1], self.resize[2]),
                                                                        p=1 - self.dropout_conv_keep_p, training=self.training
                                                                    )
                                                                ),
                                                                p=1 - self.dropout_conv_keep_p, training=self.training
                                                            )
                                                        )
                                                    ),
                                                    p=1 - self.dropout_conv_keep_p, training=self.training
                                                )
                                            )
                                        ),
                                        p=1 - self.dropout_conv_keep_p, training=self.training
                                    )
                                )
                            ),
                            p=1 - self.dropout_conv_keep_p, training=self.training
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


