#Author: Vodohleb04
import torch
from torch import nn
from torch.nn import functional as f


class Decoder(nn.Module):
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

    def __init__(self, latent_dim, device=None, dtype=None):
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float64

        self.linear1 = nn.Linear(latent_dim, 1024, device=device, dtype=dtype)
        self.bn1 = nn.BatchNorm1d(1024, device=device, dtype=dtype)
        self.linear2 = nn.Linear(1024, 256 * 6 * 6, device=device, dtype=dtype)
        self.bn2 = nn.BatchNorm1d(256 * 6 * 6, device=device, dtype=dtype)
        self.resize = (256, 6, 6)

        self.upsample = nn.Upsample(scale_factor=2)

        self.conv_t1 = nn.ConvTranspose2d(
            256, 512, 3, stride=1, bias=False, device=device, dtype=dtype
        )
        self.bn2d1 = nn.BatchNorm2d(512, device=device, dtype=dtype)
        self.conv_t2 = nn.ConvTranspose2d(
            512, 256, 3, stride=1, bias=False, device=device, dtype=dtype
        )
        self.bn2d2 = nn.BatchNorm2d(256, device=device, dtype=dtype)
        self.conv_t3 = nn.ConvTranspose2d(
            256, 128, 5,
            stride=3, output_padding=1, padding=0, bias=False, device=device, dtype=dtype
        )
        self.bn2d3 = nn.BatchNorm2d(128, device=device, dtype=dtype)
        self.conv_t4 = nn.ConvTranspose2d(
            128, 64, 3, stride=2, padding=0, bias=False, device=device, dtype=dtype
        )
        self.bn2d4 = nn.BatchNorm2d(64, device=device, dtype=dtype)
        self.conv_t5_sigmoid = nn.ConvTranspose2d(
            64, 1, 3, stride=1, bias=False, device=device, dtype=dtype
        )

    def forward(self, input_x):
        out = f.relu(
            self.bn1(self.linear1(input_x))
        )
        out = f.relu(
            self.bn2(self.linear2(out))
        )
        out = out.view(-1, self.resize[0], self.resize[1], self.resize[2])
        out = self.upsample(out)

        out = f.relu(
            self.bn2d1(self.conv_t1(out))
        )
        out = f.relu(
            self.bn2d2(self.conv_t2(out))
        )
        out = f.relu(
            self.bn2d3(self.conv_t3(out))
        )
        out = f.relu(
            self.bn2d4(self.conv_t4(out))
        )

        out = f.sigmoid(
            self.conv_t5_sigmoid(out)
        )
        return out


def create_decoder(latent_dim, device=None, dtype=None):
    decoder = Decoder(latent_dim, device=device, dtype=dtype)
    decoder.init_weights()
    decoder = decoder.to(device)
    return decoder


