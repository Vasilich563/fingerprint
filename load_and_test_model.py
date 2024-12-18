# Author: Vodohleb04
import matplotlib.pyplot as plt
import sys
import glob
import re
import csv
import datetime
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from random import shuffle, randint

from vae.vae_decoder import create_vae_decoder
from vae.vae_encoder import res_net_50_v2_vae_encoder
from vae.variational_autoencoder import VariationalAutoencoder
from vae.vae_loss import VAELoss
from dataset_loader import CustomDataset, load_image_by_pass



COUNT_LATENT_VECTORS_FROM_REAL_IMAGE = True  # Counts latent vectors of fingerprint from real images if set to True
FINGERPRINTS_TO_SEARCH_AMOUNT = 1000
LIMIT_RETURNING_VARIANTS = 3
IMG_SHAPE_WITH_PAD = (109, 109)

latent_database_dict = {}  # Dict to save latent vectors of fingerprints
train_log_path = "./vae_fingerprint_ae_train_log.csv"
log_text_file = './vae_fingerprint_ae_text_log.log'
logging.basicConfig(level=logging.INFO, filename=log_text_file, filemode="w")
regexp_filename = re.compile(r"^(?P<real_filename>.+finger).*?\.BMP$")
#regexp_filename = re.compile(r"^(?P<real_filename>\d+)__(M|F)_Left_index_.*$")



try:
    latent_dim = int(sys.argv[1])
except Exception as exp:
    print("Error! Takes 2 positional arguments: latent_dim (int > 0) and epochs_amount (int > 0)")
    logging.error("Error! Takes 2 positional arguments: latent_dim (int > 0) and epochs_amount (int > 0)")
    sys.exit(-1)


def define_device():
    if torch.cuda.is_available():
        print("Running on gpu")
        logging.info("Running on gpu")
        return torch.device('cuda:0')
    else:
        print("Running on cpu")
        logging.info("Running on cpu")
        return torch.device('cpu')


def plot_log(log_path):
    x_epochs = []
    y_train_losses = []
    y_val_losses = []
    with open(log_path, 'r') as log_f:
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


def sample_z(z_logvar, shape):
    z_mean = np.random.normal(0., 1., shape)
    z_mean = torch.from_numpy(z_mean).type(torch.float32).to(device)

    eps = np.random.normal(0., 1., shape)
    eps = torch.from_numpy(eps).type(torch.float32).to(device)

    z = eps * torch.exp(z_logvar * 0.5) + z_mean

    del z_mean
    del eps
    return z


def search(encoded_fingerprint_to_search, fingerprints_to_search, person_number_to_search_, shape, search_for_real=True,
           print_results=True):
    def print_search_result(shape):
        nonlocal variants_list, person_number_to_search_
        figure_ = plt.figure(figsize=(109 * 3, 109 * 3))
        figure_.suptitle(f"Search for {'real' if search_for_real else 'altered'} fingerprint of person {person_number_to_search_}")

        figure_.add_subplot(2 + len(variants_list), 3, 1)

        plt.imshow(fingerprints_to_search.cpu().reshape(shape[0], shape[1], 1))  # Image of fingerprint to find in database
        plt.title(f"Image to search")

        figure_.add_subplot(2 + len(variants_list), 3, 2)
        plt.imshow(
            load_image_by_pass(
                real_images_paths[person_number_to_search_]
            )
        )  # Real image of fingerprint to find in database
        plt.title(f"Real image to search")

        i = 0
        for variant in variants_list:
            i += 1
            print(
                f"Variant {i}, searched (search for {'real' if search_for_real else 'altered'}): {person_number_to_search_};\n\t"
                f"Variant person number: {variant['person_number']}; mse: {variant['mse']}"
            )
            figure_.add_subplot(2 + len(variants_list), 3, 3 * i)
            plt.imshow(
                load_image_by_pass(
                    real_images_paths[variant['person_number']]
                )
            )  # Image of fingerprint from database
            plt.title(f"Real image of variant, person number {variant['person_number']}")

        plt.show()
        del figure_

    variants_list = []
    for variant_person_number, variant_latent_vector in latent_database_dict.items():
        mse = torch.nn.functional.mse_loss(variant_latent_vector, encoded_fingerprint_to_search)
        if len(variants_list) < LIMIT_RETURNING_VARIANTS:
            variants_list.append(
                {"person_number": variant_person_number, "latent_vector": variant_latent_vector, "mse": mse}
            )
        else:
            maximal_mse_index = 0
            for i in range(1, len(variants_list)):
                if variants_list[i]["mse"] > variants_list[maximal_mse_index]["mse"]:
                    maximal_mse_index = i
            if mse < variants_list[maximal_mse_index]["mse"]:
                variants_list[maximal_mse_index] = {
                    "person_number": variant_person_number, "latent_vector": variant_latent_vector, "mse": mse
                }
    variants_list.sort(key=lambda variant: variant["mse"])
    if print_results:
        print_search_result(shape)
    for variant in variants_list:
        if variant["person_number"] == person_number_to_search_:
            return 1
    return 0


def remove_images_of_wrong_shape(dataset, right_shape=(0, 0)):
    right_shape = torch.Size((1, right_shape[0], right_shape[1]))
    removed_amount = 0
    i = 0
    left_bound = len(dataset)
    while i < left_bound:
        x_image, y_image, person_number_ = dataset[i]
        if x_image.shape != right_shape or y_image.shape != right_shape:
            dataset.remove_by_index(i)
            removed_amount += 1
            left_bound -= 1
            continue
        i += 1
    return removed_amount

# Defining device (Try to connect cuda, cpu is used otherwise)
device = define_device()

# Create dataset
altered_images_paths = {}
real_images_paths = {}

altered_images_dir_paths = [
    "/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/Altered/Altered-Easy",
    "/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/Altered/Altered-Hard",
    "/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/Altered/Altered-Medium",
    "/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/AlteredByMeNoRotation"
]
real_counter = 0
# Get paths of fingerprint images
for image_path in glob.glob("/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/Real" + "/*"):
    match = regexp_filename.search(image_path.split("/")[-1])
    if match is not None:
        real_images_paths[match.group("real_filename")] = image_path  # Get real fingerprints paths
        altered_images_paths[match.group("real_filename")] = []
        real_counter += 1


print(f"Real images found: {real_counter}")
logging.info(f"Real images found: {real_counter}")


altered_counter = 0
for altered_images_dir_path in altered_images_dir_paths:
    for image_path in glob.glob(altered_images_dir_path + "/*"):
        match = regexp_filename.search(image_path.split("/")[-1])
        if match is not None:
            try:
                if len(altered_images_paths[match.group("real_filename")]) < 20:
                    altered_images_paths[match.group("real_filename")].append(image_path)  # Get altered fingerprint paths
                    altered_counter += 1
            except KeyError:
                print(f"filename: {image_path}, number:{match.group('real_filename')}")
                logging.error(f"filename: {image_path}, number:{match.group('real_filename')}")

print(f"Altered images found: {altered_counter}")
logging.info(f"Altered images found: {altered_counter}")

altered_images_set = [None for _ in range(altered_counter)]
real_images_set = [None for _ in range(altered_counter)]
person_numbers = [None for _ in range(altered_counter)]

# Shuffle and division on subsets (train, validation, test)
altered_images_paths_items = list(altered_images_paths.items())
shuffle(altered_images_paths_items)
i = 0
for person_number, altered_images_paths in altered_images_paths_items:
    for altered_image_path in altered_images_paths:
        altered_images_set[i] = altered_image_path
        real_images_set[i] = real_images_paths[person_number]
        person_numbers[i] = person_number
        i += 1

del altered_images_paths
del altered_images_paths_items

train_size = int(altered_counter * 0.65)
test_size = int(altered_counter * 0.25)
validation_size = altered_counter - train_size - test_size

train_dataset = CustomDataset(
    x_=altered_images_set[:train_size],
    y_=real_images_set[:train_size],
    numbers_=person_numbers[:train_size],
    transform=transforms.Compose([transforms.ToTensor()]),
    padding_size=(IMG_SHAPE_WITH_PAD[0] - 103, IMG_SHAPE_WITH_PAD[1] - 96),
    inverse=True
)

validation_dataset = CustomDataset(
    x_=altered_images_set[train_size: train_size + validation_size],
    y_=real_images_set[train_size: train_size + validation_size],
    numbers_=person_numbers[train_size: train_size + validation_size],
    transform=transforms.Compose([transforms.ToTensor()]),
    padding_size=(IMG_SHAPE_WITH_PAD[0] - 103, IMG_SHAPE_WITH_PAD[1] - 96),
    inverse=True
)

test_dataset = CustomDataset(
    x_=altered_images_set[train_size + validation_size:],
    y_=real_images_set[train_size + validation_size:],
    numbers_=person_numbers[train_size + validation_size:],
    transform=transforms.Compose([transforms.ToTensor()]),
    padding_size=(IMG_SHAPE_WITH_PAD[0] - 103, IMG_SHAPE_WITH_PAD[1] - 96),
    inverse=True
)
# Remove bad examples
removed = remove_images_of_wrong_shape(train_dataset, (IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1]))
removed += remove_images_of_wrong_shape(validation_dataset, (IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1]))
removed += remove_images_of_wrong_shape(test_dataset, (IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1]))

print(f"Altered images were removed: {removed}. Result amount of altered images: {altered_counter - removed}")
print(f"Train dataset size: {len(train_dataset)}, "
      f"validation dataset size: {len(validation_dataset)}, "
      f"test dataset size: {len(test_dataset)}")
logging.info(
    f"Altered images were removed: {removed}. Result amount of altered images: {altered_counter - removed}"
    f"Train dataset size: {len(train_dataset)}, "
    f"validation dataset size: {len(validation_dataset)}, "
    f"test dataset size: {len(test_dataset)}"
)

# Creating data loaders
train_loader = DataLoader(train_dataset, 64, shuffle=True, drop_last=True)
validation_loader = DataLoader(validation_dataset, 1, shuffle=True)
test_loader = DataLoader(test_dataset, 1)

# Init model
encoder = res_net_50_v2_vae_encoder(latent_dim, 1, [7, 5, 3, 3], 1, device, torch.float32)
encoder.load_state_dict(torch.load("./vae_best_val_loss_encoder_state_dict.pt"))
decoder = create_vae_decoder(latent_dim, 1,  1, device, torch.float32)
decoder.load_state_dict(torch.load("./vae_best_loss_decoder_state_dict.pt"))

encoder = encoder.to(device)
decoder = decoder.to(device)

autoencoder = VariationalAutoencoder(encoder, decoder).to(device)

loss_function = VAELoss()

# Train and test model


# Testing phase
autoencoder.eval()

print("Testing")
logging.info("Testing")
test_start = datetime.datetime.now()
test_losses = torch.Tensor([0.0]).type(torch.float32).to(device)

random_test_fingerprints_to_check = []
# Define test sample to check search
while len(random_test_fingerprints_to_check) != FINGERPRINTS_TO_SEARCH_AMOUNT:
    fingerprint_index = randint(0, len(test_dataset) - 1)
    if fingerprint_index not in random_test_fingerprints_to_check:
        random_test_fingerprints_to_check.append(fingerprint_index)

with torch.no_grad():
    figure = plt.figure(figsize=(109 * 3, 109 * 3))

    j = 0  # TODO remove
    counter = 1
    for test_altered_image, test_real_image, test_person_number in test_loader:
        test_altered_image = test_altered_image.type(torch.float32).to(device)
        test_real_image = test_real_image.type(torch.float32).to(device)

        test_z_mean, test_z_logvar = encoder(test_altered_image)
        test_z = autoencoder.reparameterize(test_z_mean, test_z_logvar)
        test_decoded_image = decoder(test_z)

        test_step_loss = loss_function(test_real_image, test_decoded_image, test_z_mean, test_z_logvar)

        test_step_loss = test_step_loss.detach()
        test_losses += test_step_loss / len(test_loader)

        # # TODO remove
        if j < 225 and j % 75 == 0:
            figure.add_subplot(225 // 75, 2, counter)
            plt.imshow(test_real_image.cpu().reshape(IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1], 1))
            plt.title(f"Image, counter: {counter}")

            counter += 1
            figure.add_subplot(225 // 75, 2, counter)
            mean = encoder(test_real_image)[0]
            plt.imshow(decoder(mean).cpu().reshape(IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1], 1))
            plt.title(f"Decoded, counter: {counter - 1}")
            counter += 1
            del mean
        j += 1

        # Counting latent vectors

        if COUNT_LATENT_VECTORS_FROM_REAL_IMAGE:
            encoded_fingerprint_mean, _ = encoder(test_real_image)
        else:
            # Only the latest fingerprint will be saved in the dictionary
            encoded_fingerprint_mean, _ = encoder(test_altered_image)

        del test_altered_image
        del test_real_image
        del test_step_loss

        for i in range(encoded_fingerprint_mean.shape[0]):
            latent_database_dict[test_person_number[i]] = encoded_fingerprint_mean[i].reshape(1, -1)

    print("\tTime spent on test: {}".format(datetime.datetime.now() - test_start))
    print("\ttest losses: {}".format(test_losses.item()))
    logging.info(
        f"\tTime spent on test: {datetime.datetime.now() - test_start}\n"
        f"\ttest losses: {test_losses.item()}\n"
    )

    #figure.show()
    plt.show()
    del figure

    # TODO remove
    # for i in range(3):
    #     gen_altered_image, _, _ = test_dataset[i]
    #     gen_altered_image = gen_altered_image.type(torch.float32).to(device)
    #
    #     _, gen_z_logvar = encoder(gen_altered_image.reshape(1, 1, IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1]))
    #     gen_z = sample_z(test_z_logvar, test_z_logvar.shape)
    #
    #     generated_image = decoder(gen_z)
    #
    #     plt.imshow(generated_image.cpu().reshape(IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1], 1))
    #     plt.title("Generated image")
    #     plt.show()
    #     del gen_altered_image
    #     del gen_z_logvar
    #     del gen_z
    #     del generated_image


    # Search for fingerprints
    real_right = 0
    altered_right = 0
    iter = -1
    for fingerprint_index in random_test_fingerprints_to_check:
        iter += 1
        altered_fingerprint_to_search, real_fingerprint_to_search, person_number_to_search = test_dataset[
            fingerprint_index]

        altered_fingerprint_to_search = altered_fingerprint_to_search.type(torch.float32).to(device)
        real_fingerprint_to_search = real_fingerprint_to_search.type(torch.float32).to(device)
        # person_number_to_search = person_number_to_search.type(torch.float32).to(device)

        encoded_altered_fingerprint_to_search_mean, _ = encoder(
            altered_fingerprint_to_search.reshape(1, 1, IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1])
        )
        encoded_real_fingerprint_to_search_mean, _ = encoder(
            real_fingerprint_to_search.reshape(1, 1, IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1])
        )

        # Search for variants for encoded altered fingerprint
        real_right += search(
            encoded_real_fingerprint_to_search_mean,
            real_fingerprint_to_search,
            person_number_to_search,
            (IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1]),
            print_results=False  # True if iter < 3 else False
        )

        altered_right += search(
            encoded_altered_fingerprint_to_search_mean,
            altered_fingerprint_to_search,
            person_number_to_search,
            (IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1]),
            search_for_real=False,
            print_results=True if 3 <= iter < 6 else False
        )
        del altered_fingerprint_to_search
        del real_fingerprint_to_search

    print(
        f"Real fingerprint accuracy: {real_right / len(random_test_fingerprints_to_check)}\n"
        f"Altered fingerprint accuracy: {altered_right / len(random_test_fingerprints_to_check)}"
    )
    logging.info(
        f"Real fingerprint accuracy: {real_right / len(random_test_fingerprints_to_check)}\n"
        f"Altered fingerprint accuracy: {altered_right / len(random_test_fingerprints_to_check)}"
    )


plot_log(train_log_path)
