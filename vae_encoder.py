# Author: Vodohleb04
from torch import nn
from res_net_v2 import res_net_50_v2
from res_net import res_net_50
from vae_interlayer import create_vae_interlayer


class VAEEncoder(nn.Module):

    def __init__(self, main_network, interlayer):
        super().__init__()

        self.main_network = main_network
        self.interlayer = interlayer

    def forward(self, x):
        x = self.main_network(x)
        z_mean, z_logvar = self.interlayer(x)
        return z_mean, z_logvar




def res_net_50_vae_encoder(
        latent_features, dropblock_conv_keep_p=0.8, dropblock_sizes=None, dropout_linear_keep_p=0.5, device=None, dtype=None
):
    main_network = res_net_50(
        1024,
        dropblock_conv_keep_p=dropblock_conv_keep_p,
        dropblock_sizes=dropblock_sizes,
        dropout_linear_keep_p=dropout_linear_keep_p,
        device=device,
        dtype=dtype
    )
    interlayer = create_vae_interlayer(
        1024, [728], latent_features, dropout_linear_keep_p,
        activation_on_last_layer=False, device=device, dtype=dtype
    )
    encoder = VAEEncoder(main_network, interlayer).to(device)
    return encoder


def res_net_50_v2_vae_encoder(
        latent_features, dropblock_conv_keep_p=0.8, dropblock_sizes=None, dropout_linear_keep_p=0.5, device=None, dtype=None
):
    main_network = res_net_50_v2(
        1024,
        dropblock_conv_keep_p=dropblock_conv_keep_p,
        dropblock_sizes=dropblock_sizes,
        dropout_linear_keep_p=dropout_linear_keep_p,
        device=device,
        dtype=dtype
    )
    interlayer = create_vae_interlayer(
        1024, [728], latent_features, dropout_linear_keep_p,
        activation_on_last_layer=False, device=device, dtype=dtype
    )
    encoder = VAEEncoder(main_network, interlayer).to(device)
    return encoder
