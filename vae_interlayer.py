# Author: Vodohleb04
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as f


class VAEInterlayer(nn.Module):

    def __init__(
            self, input_features, middle_layers_features, output_features, dropout_keep_p,
            activation_on_last_layer=True, device=None, dtype=None
    ):
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float64
        if not isinstance(middle_layers_features, list):
            raise TypeError(
                f'Expected middle_layers_features to be a list, got {type(middle_layers_features)} instead'
            )
        if len(middle_layers_features) <= 0:
            ValueError(f'middle_layers_features should be not empty list of int')

        self.dropout_keep_p = dropout_keep_p
        self.activation_on_last_layer = activation_on_last_layer
        self.input_bn = nn.BatchNorm1d(middle_layers_features[0], device=device, dtype=dtype)
        self.input_layer = nn.Linear(input_features, middle_layers_features[0], device=device, dtype=dtype)

        middle_layers = nn.ModuleList()
        i = 0
        while i < len(middle_layers_features) - 1:
            middle_layers.append(
                nn.Linear(middle_layers_features[i], middle_layers_features[i + 1], device=device, dtype=dtype)
            )
            middle_layers.append(
                nn.BatchNorm1d(middle_layers_features[i + 1], device=device, dtype=dtype)
            )
            middle_layers.append(
                nn.Dropout(1 - self.dropout_keep_p, inplace=True)
            )
            middle_layers.append(
                nn.LeakyReLU(inplace=True)
            )
            i += 1

        self.middle_net = nn.Sequential(*middle_layers)
        self.z_mean_layer = nn.Linear(middle_layers_features[-1], output_features, device=device, dtype=dtype)
        self.z_logvar_layer = nn.Linear(middle_layers_features[-1], output_features, device=device, dtype=dtype)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :returns Tuple[torch.Tensor, torch.Tensor] z_mean, z_log_var
        """
        out = self.input_layer(x)
        out = self.input_bn(out)
        out = f.dropout(out, p=1 - self.dropout_keep_p, training=self.training, inplace=True)
        out = f.leaky_relu_(out)

        out = self.middle_net(out)

        z_mean = self.z_mean_layer(out)
        z_logvar = self.z_logvar_layer(out)

        return z_mean, z_logvar

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.z_mean_layer and module is not self.z_logvar_layer:
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def create_vae_interlayer(
        input_features, middle_layers_features, output_features,
        dropout_keep_p=0.5, activation_on_last_layer=True, device=None, dtype=None
):
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float64
    interlayer = VAEInterlayer(
        input_features, middle_layers_features, output_features, dropout_keep_p, activation_on_last_layer, device, dtype
    )
    interlayer.init_weights()
    interlayer = interlayer.to(device)
    return interlayer
