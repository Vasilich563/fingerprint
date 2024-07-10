import torch
from torch import nn
import numpy as np


class VariationalAutoencoder(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def reparameterize(z_mean, z_logvar):
        eps = torch.from_numpy(np.random.normal(0, 1, z_logvar.shape)).type(z_mean.dtype).to(z_mean.device)
        z = eps * torch.exp(z_logvar * 0.5) + z_mean
        return z

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        x = self.decoder(z)
        return x

