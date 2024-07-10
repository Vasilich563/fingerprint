from noiser import Noiser
import matplotlib.pyplot as plt
import glob
import cv2
import random
import os


def alter(image_):
    choice = random.randint(0, 1)
    if choice == 0:
        noise_factor = random.randint(5, 50) / 100.
        image_ = Noiser.add_normal_noise(image_, noise_factor=noise_factor)
    # elif choice == 1:
    #     angle = random.randint(1, 359)
    #     image_ = Noiser.rotate_image(image_, angle)
    elif choice == 1:
        kernel_size = random.randint(1, 6)
        image_ = Noiser.blur_image(image_, (kernel_size, kernel_size))
    return image_


dirs = [
    "/home/vodohleb/PycharmProjects/dl/SOCOFing/Altered/Altered-Easy",
    "/home/vodohleb/PycharmProjects/dl/SOCOFing/Altered/Altered-Hard",
    "/home/vodohleb/PycharmProjects/dl/SOCOFing/Altered/Altered-Medium",
    "/home/vodohleb/PycharmProjects/dl/SOCOFing/Real"
]

files_amout = 0
for dir in dirs:
    for path in glob.glob(dir + "/*"):
        files_amout += 1

altered_amount_per_file = 4
shown_amount = 0
altered_counter = 0
for dir in dirs:
    for path in glob.glob(dir + "/*"):
        name = path.split("/")[-1]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)[:, :, 0]
        image = (image - image.min()) / (image.max() - image.min())

        not_altered_image = image.copy()
        for i in range(altered_amount_per_file):
            altered_counter += 1
            print(f"Processing {altered_counter}/{files_amout * altered_amount_per_file}")
            os.system('clear')
            choice = random.randint(0, 2)
            if choice < 2:
                image = alter(image)
            else:
                alter_amount = random.randint(1, 2)
                for _ in range(alter_amount):
                    image = alter(image)
            cv2.imwrite(
                f"/home/vodohleb/PycharmProjects/dl/SOCOFing/AlteredByMeNoRotation/{name.replace('.BMP', '')}_{i}.BMP",
                cv2.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            )

            if shown_amount < -1:
                shown_amount += 1
                plt.figure(figsize=(100, 100))
                plt.subplot(2, 1, 1)
                plt.imshow(not_altered_image.reshape(103, 96))
                plt.subplot(2, 1, 2)
                plt.imshow(image.reshape(103, 96))
                plt.title("Processed image")
                plt.show()

