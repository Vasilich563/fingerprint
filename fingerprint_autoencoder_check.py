#Author: Vodohleb04
import datetime
import torch
from torch import nn as nn
from torch.nn import functional as f
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('cuda')
else:
    device = torch.device("cpu")
    print('cpu')


def init_asymmetric_weights(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)


class Encoder(nn.Module):

    def init_weights(self):
        self.apply(init_asymmetric_weights)

    def __init__(self, device):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, device=device, dtype=torch.float64)
        self.conv2 = nn.Conv2d(64, 64, 3, device=device, dtype=torch.float64)
        self.conv3 = nn.Conv2d(64, 128, 3, device=device, dtype=torch.float64)
        self.conv4 = nn.Conv2d(128, 128, 3, device=device, dtype=torch.float64)
        self.linear = nn.Linear(128 * 8 * 8, 50, dtype=torch.float64, device=device)
        self.indices = [None for _ in range(4)]  # TODO remove
        self.output_sizes = [None for _ in range(4)]  # TODO remove

    def forward(self, x):

        def operation(x_, conv, index):
            print(f"input: {x_.size()}")
            self.output_sizes[index] = x_.size()
            x_, indices = f.max_pool2d(
                f.conv2d(
                    x_, conv.weight.clone(), conv.bias.clone()
                ),
                kernel_size=(2, 2),
                return_indices=True,
            )
            self.indices[index] = indices
            print(f"indices: {indices.size()}")  # TODO remove
            return f.relu_(x_)

        buffer = x.view(-1, 1, 160, 160)
        buffer = operation(buffer, self.conv1, 0)
        buffer = operation(buffer, self.conv2, 1)
        buffer = operation(buffer, self.conv3, 2)
        buffer = operation(buffer, self.conv4, 3)
        print(buffer.size())
        buffer = buffer.view(-1, 128 * 8 * 8)
        buffer = f.relu(
            f.linear(buffer, self.linear.weight.clone(), self.linear.bias.clone())
        )

        return buffer


class Decoder(nn.Module):

    def init_weights(self):
        self.apply(init_asymmetric_weights)

    def __init__(self, device):
        super().__init__()
        self.linear = nn.Linear(50, 128 * 8 * 8, dtype=torch.float64, device=device)
        self.conv1 = nn.ConvTranspose2d(128, 128, 3, device=device, dtype=torch.float64)
        self.conv2 = nn.ConvTranspose2d(128, 64, 3, device=device, dtype=torch.float64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 3, device=device, dtype=torch.float64)
        self.conv4 = nn.ConvTranspose2d(64, 1, 3, device=device, dtype=torch.float64)

    def forward(self, x, indices, output_sizes):
        def operation(x_, conv, indices_, output_size_):
            return (
                f.relu(
                    f.max_unpool2d(
                        f.conv_transpose2d(
                            x_, conv.weight.clone(), conv.bias.clone()
                        ),
                        indices_,
                        kernel_size=(2, 2),
                        output_size=output_size_
                    )
                )
            )

        buffer = x

        buffer = f.relu(
            f.linear(buffer, self.linear.weight.clone(), self.linear.bias.clone())
        )
        buffer = buffer.view(-1, 128, 8, 8)
        buffer = operation(buffer, self.conv1, indices[3], output_sizes[3])
        buffer = operation(buffer, self.conv2, indices[2], output_sizes[2])
        buffer = operation(buffer, self.conv3, indices[1], output_sizes[1])
        buffer = operation(buffer, self.conv4, indices[0], output_sizes[0])

        print(buffer.size())

        return buffer.view(-1, 160, 160)


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def init_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()

    def forward(self, x):
        return self.decoder(
            self.encoder(x), self.encoder.indices, self.encoder.output_sizes
        )


x_train, y_train = np.random.random(25600 * 3500).reshape(3500, 160, 160), []  # one class from 600
x_validation, y_validation = np.random.random(25600 * 500).reshape(500, 160, 160), []  # one class from 600
x_test, y_test = np.random.random(25600 * 1500).reshape(1500, 160, 160), []  # one class from 600


class CustomDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]

        if self.transforms is not None:
            image = self.transforms(image)
        return image


train_data = CustomDataset(x_train, transforms=transforms.Compose([transforms.ToTensor()]))
validation_data = CustomDataset(x_validation, transforms=transforms.Compose([transforms.ToTensor()]))
test_data = CustomDataset(x_test, transforms=transforms.Compose([transforms.ToTensor()]))


train_loader = DataLoader(train_data, 64)
validation_loader = DataLoader(validation_data, 64)
test_loader = DataLoader(validation_data, 64)

encoder = Encoder(device)
decoder = Decoder(device)
autoencoder = Autoencoder(encoder, decoder)
autoencoder.init_weights()
autoencoder.train()

encoder = encoder.to(device)
decoder = decoder.to(device)
autoencoder = autoencoder.to(device)

loss = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

print(encoder)

# epochs = 20
# start = datetime.datetime.now()
# for epoch in range(epochs):
#     epoch_start = datetime.datetime.now()
#     print(f"epoch: {epoch + 1}/{epochs}")
#     print("\ttrain")
#     losses = []
#     val_losses = []
#     for batch in train_loader:
#         optimizer.zero_grad()
#         batch = batch.type(torch.float64).to(device)
#
#         decoded_batch = autoencoder(batch)
#
#         step_loss = loss(decoded_batch, batch)
#
#         step_loss.backward()
#         optimizer.step()
#         losses.append(step_loss)
#
#     print("\tvalidation")
#     with torch.no_grad():
#         for val_batch in validation_loader:
#             val_batch = val_batch.type(torch.float64).to(device)
#             decoded_val_batch = autoencoder(val_batch)
#             step_val_loss = loss(decoded_val_batch, val_batch)
#             val_losses.append(step_val_loss)
#     print("\t\tTime spent: {}".format(datetime.datetime.now() - epoch_start))
#     print(f"\t\ttrain losses: {sum(losses)/len(losses)}\n\t\tval losses: {sum(val_losses)/len(val_losses)}")
#
# autoencoder.train(False)
#
# print("Testing")
# test_start = datetime.datetime.now()
# test_losses = []
# with torch.no_grad():
#     for test_batch in test_loader:
#         test_batch = test_batch.type(torch.float64).to(device)
#         decoded_test_batch = autoencoder(test_batch)
#         test_step_loss = loss(decoded_test_batch, test_batch)
#         test_losses.append(test_step_loss)
#
#     print("Time spent on test: {}".format(datetime.datetime.now() - test_start))
#     print("test losses: {}".format(sum(test_losses)/len(test_losses)))
#     print("Time spent at all: {}".format(datetime.datetime.now() - start))

