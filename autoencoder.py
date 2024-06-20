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




