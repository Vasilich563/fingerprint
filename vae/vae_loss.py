import torch
from torch import nn


class VAELoss(nn.Module):

    def __init__(self, ):
        super().__init__()

    def forward(self, target_x, decoded_x, z_mean, z_logvar):
        decoded_x = torch.clip(decoded_x, min=1e-10, max=1)
        reconstruction_loss = nn.functional.binary_cross_entropy(decoded_x, target_x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - torch.exp(z_logvar))
        return reconstruction_loss + kl_loss
