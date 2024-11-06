from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np


class CustomDataset(Dataset):
    @staticmethod
    def normalize(img, inverse=False):
        if inverse:
            return 1 - ((img - img.min()) / (img.max() - img.min()))
        else:
            return (img - img.min()) / (img.max() - img.min())

    def __init__(self, x_, y_, numbers_, transform=None, padding_size=(0, 0), inverse=False):
        self.x = x_
        self.y = y_
        self.numbers = numbers_
        self.transform = transform
        self.padding_size = padding_size
        self.inverse = inverse

    def __len__(self):
        return len(self.x)

    def remove_by_index(self, index):
        self.x.pop(index)
        self.y.pop(index)
        self.numbers.pop(index)

    def __getitem__(self, index):
        x_filepath = self.x[index]
        y_filepath = self.y[index]

        x_image = cv2.imread(x_filepath)
        x_image = np.pad(
            cv2.cvtColor(x_image, cv2.IMREAD_GRAYSCALE)[:, :, 0],
            [(0, self.padding_size[0]), (0, self.padding_size[1])],
            mode="constant",
            constant_values=((255.,), (255.,))
        )
        if x_filepath == y_filepath:
            y_image = x_image
        else:
            y_image = cv2.imread(y_filepath)
            y_image = np.pad(
                cv2.cvtColor(y_image, cv2.IMREAD_GRAYSCALE)[:, :, 0],
                [(0, self.padding_size[0]),
                 (0, self.padding_size[1])],
                mode="constant",
                constant_values=((255.,), (255.,))
            )
        if self.transform is not None:
            x_image = self.transform(x_image)
            y_image = self.transform(y_image)
        return (
            self.normalize(x_image, inverse=self.inverse),
            self.normalize(y_image, inverse=self.inverse),
            self.numbers[index]
        )


def load_image_by_pass(path):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)[:, :, 0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import glob
    import re
    paths = {}
    real_paths = {}

    data_pathes = [
        "/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/Altered/Altered-Easy",
        "/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/Altered/Altered-Hard",
        "/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/Altered/Altered-Medium"
    ]

    for path in glob.glob("/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/Real" + "/*"):
        match = re.search(r"^(?P<number>\d+)__(M|F)_Left_index_.*$", path.split("/")[-1])
        if match is not None:
            real_paths[match.group("number")] = path
            paths[match.group("number")] = []

    counter = 0
    for data_path in data_pathes:
        for path in glob.glob(data_path + "/*"):
            match = re.search(r"^(?P<number>\d+)__(M|F)_Left_index_.*$", path.split("/")[-1])
            if match is not None:
                try:
                    paths[match.group("number")].append(path)
                    counter += 1
                except KeyError:
                    print(f"filename: {path}, number:{match.group('number')}")

    print(counter)

    x_set, y_set, numbers = [None for _ in range(counter)], [None for _ in range(counter)], [None for _ in range(counter)]
    i = 0
    for number, path_list in paths.items():
        for path in path_list:
            x_set[i] = path
            y_set[i] = real_paths[number]
            numbers[i] = number
            i += 1

    dataset = CustomDataset(x_set, y_set, numbers, padding_size=(2, 9))

    dl = DataLoader(dataset)

    i = 0
    shapes = []
    for x, y, number in dl:
        if i < 2:
            x = x.squeeze(axis=0)
            plt.imshow(x)
            plt.show()

            y = y.squeeze(axis=0)
            plt.imshow(y)
            plt.show()
            i += 1

        shapes.append(x.shape)
    for i in range(len(shapes) - 1):
        for j in range(i + 1, len(shapes[i])):
            for k in range(len(shapes[i])):
                if shapes[i][k] != shapes[j][k]:
                    print("Error: {}, {}".format(i, k))
                    print(shapes[i], shapes[j])
