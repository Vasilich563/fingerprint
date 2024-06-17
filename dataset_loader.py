import matplotlib.pyplot as plt
import glob
import re
from torch.utils.data import Dataset, DataLoader
import cv2


data_pathes = [
    "/home/vodohleb/PycharmProjects/tensor_flow/SOCOFing/Real",
    "/home/vodohleb/PycharmProjects/tensor_flow/SOCOFing/Altered/Altered-Easy",
    "/home/vodohleb/PycharmProjects/tensor_flow/SOCOFing/Altered/Altered-Hard",
    "/home/vodohleb/PycharmProjects/tensor_flow/SOCOFing/Altered/Altered-Medium"
]
images_paths = []
classes = []


for data_path in data_pathes:
    for path in glob.glob(data_path + "/*"):
        match = re.search(r"^(?P<number>\d+).*$", path.split("/")[-1])
        if match is not None:
            images_paths.append(data_path + "/" + match.group(0))
            classes.append(match.group("number"))


class CustomDataset(Dataset):

    def normalize(self, img):
        return (img - img.min()) / (img.max() - img.min())

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index, padding_for_size=(0, 0)):
        filepath = self.data[index]
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)[:, :, 0]
        #print(image.shape)
        #image = image.squeeze(axis=2)
        if self.transform is not None:
            image = self.transform(image)
        return self.normalize(image), self.labels[index]


dataset = CustomDataset(images_paths, classes)

dl = DataLoader(dataset)

i = 0
for x, y in dl:
    if i == 4:
        break
    print(x, y, x.shape)
    x = x.squeeze(axis=0)
    plt.imshow(x)
    plt.show()
    i += 1


import numpy as np
x = [1, 2, 3]
y = [0.09, 0.07, 0.067]

x_epochs = np.array(x)
y_train_losses = np.array(y)

plt.plot(x_epochs, y_train_losses, label="Train loss", color="green")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Losses logs")

plt.legend()
plt.show()



