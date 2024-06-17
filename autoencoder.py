from torch import nn
from res_net import res_net_50
from decoder import create_decoder
from res_net_v2 import res_net_50_v2
from decoder_v2 import create_decoder_v2


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_x):
        return self.decoder(
            self.encoder(input_x)
        )


def create_autoencoder(latent_dim, device=None, dtype=None):
    encoder = res_net_50(latent_dim, device, dtype)
    decoder = create_decoder(latent_dim, device, dtype)
    return Autoencoder(encoder, decoder).to(device)


def create_autoencoder_v2(latent_dim, dropout_conv_keep_p=0.8, dropout_linear_keep_p=0.5, device=None, dtype=None):
    encoder = res_net_50_v2(latent_dim, dropout_conv_keep_p, dropout_linear_keep_p, device, dtype)
    decoder = create_decoder_v2(latent_dim, dropout_conv_keep_p, dropout_linear_keep_p, device, dtype)
    return Autoencoder(encoder, decoder).to(device)

