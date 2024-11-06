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
        return x, z_mean, z_logvar
    

# async def m():
#     import chromadb
#     client = await chromadb.AsyncHttpClient(
#             host="0.0.0.0", port=8000
#         #,settings=chromadb.config.Settings(allow_reset=True, anonymized_telemetry=False)
#     )
#     collection = await client.get_or_create_collection(name="fingerptints", metadata={"hnsw:space": "l2"})
#     await collection.delete(ids= (await collection.get())["ids"])

# import asyncio 
# asyncio.run(m())

