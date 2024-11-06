from noiser import Noiser
import matplotlib.pyplot as plt
import glob
import cv2
import random


def alter(image_, choice_of_action):
    if choice_of_action == 0:
        noise_factor = random.randint(5, 50) / 100.
        image_ = Noiser.add_normal_noise(image_, noise_factor=noise_factor)
    # elif choice_of_action == 1:
    #     angle = random.randint(1, 359)
    #     image_ = Noiser.rotate_image(image_, angle)
    elif choice_of_action == 1:
        kernel_size = random.randint(1, 6)
        image_ = Noiser.blur_image(image_, (kernel_size, kernel_size))
    else:
        raise ValueError("choice_of_action must be 0 or 1")
    return image_


def random_alter_actions_on_image(image):
    one_or_multiple_actions = random.randint(0, 2)
    if one_or_multiple_actions < 2:
        choice_of_action = random.randint(0, 1)
        image = alter(image, choice_of_action)
    else:
        alter_amount = random.randint(1, 2)
        for _ in range(alter_amount):
            choice_of_action = random.randint(0, 1)
            image = alter(image, choice_of_action)
    return image


def show(original_image, image):
    plt.figure(figsize=(100, 100))
    plt.subplot(2, 1, 1)
    plt.imshow(original_image.reshape(103, 96))
    plt.subplot(2, 1, 2)
    plt.imshow(image.reshape(103, 96))
    plt.title("Processed image")
    plt.show()


def actions_on_file(path, altered_amount_per_file, altered_counter, files_amount, shown_amount):
    name = path.split("/")[-1]
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)[:, :, 0]
    image = (image - image.min()) / (image.max() - image.min())

    original_image = image.copy()
    for i in range(altered_amount_per_file):
        altered_counter += 1
        print(f"Processing {altered_counter}/{files_amount * altered_amount_per_file}")
        
        image = random_alter_actions_on_image(image)
        if not cv2.imwrite(
            f"/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/AlteredByMeNoRotation/{name.replace('.BMP', '')}_{i}.BMP",
            cv2.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ):
            raise ValueError(f"Error. {cv2.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)} {f"\/home\/vodohleb\/PycharmProjects\/fingerprint\/SOCOFing\/AlteredByMeNoRotation\/{name.replace('.BMP', '')}_{i}.BMP"}")

    if shown_amount < 1:
        shown_amount += 1
        show(original_image, image)
    return altered_counter, shown_amount

def main():
    dirs = [
        "/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/Altered/Altered-Easy",
        "/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/Altered/Altered-Hard",
        "/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/Altered/Altered-Medium",
        "/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/Real"
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
            altered_counter, shown_amount = actions_on_file(
                path, altered_amount_per_file, altered_counter, files_amout, shown_amount
            )            


if __name__ == "__main__":
    main()
