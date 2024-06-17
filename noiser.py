#Author: Vodohleb04
import numpy as np
from PIL import Image, ImageFilter


class Noiser:

    @staticmethod
    def add_normal_noise(image: np.ndarray, noise_factor: float) -> np.ndarray:
        """
        Add normal distibuted noise (between 0 and 1) to the normalized image: image + noise_factor * normal_noise

        :param image: ndarray [height, width, channels] - normalized image
        :param noise_factor: float between 0 and 1
        :return: ndarray [height, width, channels]
        """
        return image + noise_factor * np.random.normal(size=image.size).reshape(image.shape)

    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate the normalized image around the counterclockwise. Angle is given in degrees

        :param image: ndarray [height, width, channels] - normalized image
        :param angle: float - angle in degrees
        :return: ndarray [height, width, channels]
        """
        image = Image.fromarray(image)
        return np.array(image.rotate(angle, fillcolor=1.))

    @staticmethod
    def blur_image(image: np.ndarray) -> np.ndarray:
        """
        Blurs the image using BoxBlur.

        :param image: ndarray [height, width, channels] - normalized image
        :param blur_factor: degree of blur
        :return: ndarray [height, width, channels]
        """
        image = Image.fromarray(image)
        return np.array(image.filter(ImageFilter.BLUR))


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    image = cv2.imread("/home/vodohleb/PycharmProjects/tensor_flow/SOCOFing/Real/1__M_Left_little_finger.BMP")
    image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)[:, :, 0]
    image = (image - image.min()) / (image.max() - image.min())

    noised_image = Noiser.add_normal_noise(image, 0.2)
    rotated_image = Noiser.rotate_image(image, 45)
    #blured_image = Noiser.blur_image(image)

    plt.figure(figsize=(100, 100))
    plt.subplot(2, 1, 1)
    plt.imshow(image.reshape(103, 96))
    plt.subplot(2, 1, 2)
    plt.imshow(rotated_image.reshape(103, 96))
    plt.title("Processed image")
    plt.show()

    import csv
    x_epochs = []
    y_train_losses = []
    y_val_losses = []
    with open("/home/vodohleb/PycharmProjects/tensor_flow/fingerprint_ae_train_log.csv", 'r') as log_f:
        reader = csv.reader(log_f, delimiter='|')

        for row in reader:
            x_epochs.append(int(row[0]))
            y_train_losses.append(float(row[1]))
            y_val_losses.append(float(row[2]))

    x_epochs = np.array(x_epochs)
    y_train_losses = np.array(y_train_losses)
    y_val_losses = np.array(y_val_losses)

    plt.plot(x_epochs, y_train_losses, label="Train loss", color="green")
    plt.plot(x_epochs, y_val_losses, label="Validation loss", color="red")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Losses logs")

    plt.legend()
    plt.show()

