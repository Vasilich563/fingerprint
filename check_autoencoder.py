import datetime
import random

import torch
from torch import nn as nn
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader

from autoencoder import create_autoencoder

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("gpu")
else:
    device = torch.device("cpu")
    print("cpu")

mtrx = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ]
)
def get_label():
    choice = random.randint(0, 4)
    return mtrx[choice]


x_train, y_train = np.random.random(12100 * 3500).reshape(3500, 110, 110), np.array([get_label() for _ in range(3500)])  # one class from 600
x_validation, y_validation = np.random.random(12100 * 500).reshape(500, 110, 110), [get_label() for _ in range(500)]  # one class from 600
x_test, y_test = np.random.random(12100 * 1500).reshape(1500, 110, 110), [get_label() for _ in range(1500)]  # one class from 600


class CustomDataset(Dataset):
    def __init__(self, data, labels, transforms=None):
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]

        if self.transforms is not None:
            image = self.transforms(image)
            label = self.transforms(label)
        return image, label


train_data = CustomDataset(x_train, x_train, transforms=transforms.Compose([transforms.ToTensor()]))
validation_data = CustomDataset(x_validation, x_validation, transforms=transforms.Compose([transforms.ToTensor()]))
test_data = CustomDataset(x_test, x_test, transforms=transforms.Compose([transforms.ToTensor()]))


train_loader = DataLoader(train_data, 64)
validation_loader = DataLoader(validation_data, 64)
test_loader = DataLoader(validation_data, 64)

network = create_autoencoder(100, device, torch.float64)
print(network)

loss = nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)


# epochs = 2
# start = datetime.datetime.now()
# for epoch in range(epochs):
#     epoch_start = datetime.datetime.now()
#     print(f"epoch: {epoch + 1}/{epochs}")
#     print("\ttrain")
#     losses = []
#     val_losses = []
#     for x, y in train_loader:
#         optimizer.zero_grad()
#         x = x.type(torch.float64).to(device)
#         #y = torch.from_numpy(y).type(torch.float64).to(device)
#         y = y.type(torch.float64).to(device)
#
#         y_net = network(x)
#
#         step_loss = loss(y_net, y)
#
#         step_loss.backward()
#         optimizer.step()
#         losses.append(step_loss)
#
#     print("\tvalidation")
#     with torch.no_grad():
#         for val_x, val_y in validation_loader:
#             val_x = val_x.type(torch.float64).to(device)
#             # val_y = torch.from_numpy(val_y).type(torch.float64).to(device)
#             val_y = val_y.type(torch.float64).to(device)
#
#             val_y_net = network(val_x)
#             step_val_loss = loss(val_y_net, val_y)
#             val_losses.append(step_val_loss)
#     print("\t\tTime spent: {}".format(datetime.datetime.now() - epoch_start))
#     print(f"\t\ttrain losses: {sum(losses)/len(losses)}\n\t\tval losses: {sum(val_losses)/len(val_losses)}")
#
# network.train(False)
#
# print("Testing")
# test_start = datetime.datetime.now()
# test_losses = []
# with torch.no_grad():
#     for test_x, test_y in test_loader:
#         test_x = test_y.type(torch.float64).to(device)
#         #test_y = torch.from_numpy(test_y).type(torch.float64).to(device)
#         test_y = test_y.type(torch.float64).to(device)
#         test_y_net = network(test_x)
#         test_step_loss = loss(test_y_net, test_y)
#         test_losses.append(test_step_loss)
#
#     print("Time spent on test: {}".format(datetime.datetime.now() - test_start))
#     print("test losses: {}".format(sum(test_losses)/len(test_losses)))
#     print("Time spent at all: {}".format(datetime.datetime.now() - start))

