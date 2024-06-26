#Author: Vodohleb04
import sys
import glob
import re
import csv
import datetime
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from random import shuffle, randint

from res_net import res_net_50
from decoder import create_decoder
from autoencoder import Autoencoder
from autoencoder_dataset_loader import CustomDataset, load_image_by_pass
import matplotlib.pyplot as plt


COUNT_LATENT_VECTORS_FROM_REAL_IMAGE = True  # Counts latent vectors of fingerprint from real images if set to True
FINGERPRINTS_TO_SEARCH_AMOUNT = 200
LIMIT_RETURNING_VARIANTS = 3


latent_database_dict = {}  # Dict to save latent vectors of fingerprints
train_log_path = "./fingerprint_ae_train_log.csv"
log_text_file = './fingerprint_ae_text_log.log'
logging.basicConfig(level=logging.INFO, filename=log_text_file, filemode="w")
left_index_finger_regexp = re.compile(r"^(?P<number>\d+)__(M|F)_Left_index_.*$")


try:
    latent_dim = int(sys.argv[1])
    epochs_amount = int(sys.argv[2])  # TODO set normal amount
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


def search(encoded_fingerprint_to_search, fingerprints_to_search, person_number_to_search_, shape, search_for_real=True, print_results=True):
    def print_search_result(shape):
        plt.imshow(fingerprints_to_search.reshape(shape[0], shape[1], 1))  # Image of fingerprint to find in database
        plt.title(f"Image to search (search for {'real' if search_for_real else 'altered'})\n"
                  f"person number: {person_number_to_search_}")
        plt.show()

        plt.imshow(
            load_image_by_pass(
                real_images_paths[person_number_to_search_]
            )
        )  # Real image of fingerprint to find in database
        plt.title(f"Real image to search (search for {'real' if search_for_real else 'altered'})\n"
                  f"person number: {person_number_to_search_}")
        plt.show()
        i = 0
        for variant in variants_list:
            i += 1
            print(f"Variant {i}, searched (search for {'real' if search_for_real else 'altered'}): {person_number_to_search_};\n\t"
                  f"Person number from db: {variant['person_number']}; mse: {variant['mse']}")
            plt.imshow(
                load_image_by_pass(
                    real_images_paths[variant['person_number']]
                )
            )  # Image of fingerprint from database
            plt.title(
                f"Real image from database\n person number in db: {variant['person_number']}\n"
                f" person number to search (search for {'real' if search_for_real else 'altered'}): {person_number_to_search_}"
            )
            plt.show()
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


# clear log file
with open(train_log_path, 'w') as f:
    f.write("")

# Defining device (Try to connect cuda, cpu is used otherwise)
device = define_device()


# Create dataset
altered_images_paths = {}
real_images_paths = {}

altered_images_dir_paths = [
    "/home/vodohleb/PycharmProjects/dl/SOCOFing/Altered/Altered-Easy",
    "/home/vodohleb/PycharmProjects/dl/SOCOFing/Altered/Altered-Hard",
    "/home/vodohleb/PycharmProjects/dl/SOCOFing/Altered/Altered-Medium"
]

# Get paths of fingerprint images
for image_path in glob.glob("/home/vodohleb/PycharmProjects/dl/SOCOFing/Real" + "/*"):
    match = left_index_finger_regexp.search(image_path.split("/")[-1])
    if match is not None:
        real_images_paths[match.group("number")] = image_path  # Get real fingerprints paths
        altered_images_paths[match.group("number")] = []

counter = 0
for altered_images_dir_path in altered_images_dir_paths:
    for image_path in glob.glob(altered_images_dir_path + "/*"):
        match = left_index_finger_regexp.search(image_path.split("/")[-1])
        if match is not None:
            try:
                altered_images_paths[match.group("number")].append(image_path)  # Get altered fingerprint paths
                counter += 1
            except KeyError:
                print(f"filename: {image_path}, number:{match.group('number')}")
                logging.error(f"filename: {image_path}, number:{match.group('number')}")

print(f"Altered images found: {counter}")
logging.info(f"Altered images found: {counter}")

altered_images_set = [None for _ in range(counter)]
real_images_set = [None for _ in range(counter)]
person_numbers = [None for _ in range(counter)]

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

train_size = int(counter * 0.65)
test_size = int(counter * 0.25)
validation_size = counter - train_size - test_size

train_dataset = CustomDataset(
    x_=altered_images_set[:train_size],
    y_=real_images_set[:train_size],
    numbers_=person_numbers[:train_size],
    transform=transforms.Compose([transforms.ToTensor()]),
    padding_size=(105 - 103, 105 - 96)
)

validation_dataset = CustomDataset(
    x_=altered_images_set[train_size: train_size + validation_size],
    y_=real_images_set[train_size: train_size + validation_size],
    numbers_=person_numbers[train_size: train_size + validation_size],
    transform=transforms.Compose([transforms.ToTensor()]),
    padding_size=(105 - 103, 105 - 96)
)

test_dataset = CustomDataset(
    x_=altered_images_set[train_size + validation_size:],
    y_=real_images_set[train_size + validation_size:],
    numbers_=person_numbers[train_size + validation_size:],
    transform=transforms.Compose([transforms.ToTensor()]),
    padding_size=(105 - 103, 105 - 96)
)
# Remove bad examples
removed = remove_images_of_wrong_shape(train_dataset, (105, 105))
removed += remove_images_of_wrong_shape(validation_dataset, (105, 105))
removed += remove_images_of_wrong_shape(test_dataset, (105, 105))


print(f"Altered images were removed: {removed}. Result amount of altered images: {counter - removed}")
print(f"Train dataset size: {len(train_dataset)}, "
      f"validation dataset size: {len(validation_dataset)}, "
      f"test dataset size: {len(test_dataset)}")
logging.info(
    f"Altered images were removed: {removed}. Result amount of altered images: {counter - removed}"
    f"Train dataset size: {len(train_dataset)}, "
    f"validation dataset size: {len(validation_dataset)}, "
    f"test dataset size: {len(test_dataset)}"
)

# Creating data loaders
train_loader = DataLoader(train_dataset, 32, shuffle=True)
validation_loader = DataLoader(validation_dataset, 1, shuffle=True)
test_loader = DataLoader(test_dataset, 1)

# Init model
encoder = res_net_50(latent_dim, 0.8, 3, 0.5, device, torch.float64)
decoder = create_decoder(latent_dim, device, torch.float64)

autoencoder = Autoencoder(encoder, decoder)
autoencoder = autoencoder.to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)

# Train and test model
start = datetime.datetime.now()
best_val_mse = torch.Tensor([1.0]).to(device)
best_val_mse_epoch = 1
for epoch in range(epochs_amount):
    epoch_start = datetime.datetime.now()
    print(f"Epoch: {epoch + 1}/{epochs_amount}")
    print("\tTrain")
    logging.info(f"Epoch: {epoch + 1}/{epochs_amount}\n\tTrain")
    losses = []
    val_losses = []
    # Train phase
    i = 0
    for altered_image, real_image, _ in train_loader:
        autoencoder.train()
        i += 1
        print(f"\t\tTrain mini-batch number: {i}")
        logging.info(f"\t\tTrain mini-batch number: {i}")
        optimizer.zero_grad()

        altered_image = altered_image.type(torch.float64).to(device)
        real_image = real_image.type(torch.float64).to(device)

        processed_image = autoencoder(altered_image)

        # Gradient step
        step_loss = loss_function(processed_image, real_image)
        step_loss.backward()
        optimizer.step()
        losses.append(step_loss)

        del altered_image
        del real_image
        del processed_image
    # Validation phase of train phase
    print("\tValidation")
    logging.info("\tValidation")
    with torch.no_grad():
        for val_altered_image, val_real_image, _ in validation_loader:
            autoencoder.eval()
            val_altered_image = val_altered_image.type(torch.float64).to(device)
            val_real_image = val_real_image.type(torch.float64).to(device)

            val_processed_image = autoencoder(val_altered_image)
            step_val_loss = loss_function(val_processed_image, val_real_image)
            val_losses.append(step_val_loss)

            del val_altered_image
            del val_real_image
            del val_processed_image
    print(
        f"\t\tTime spent: {datetime.datetime.now() - epoch_start}\n"
        f"\t\ttrain losses: {sum(losses) / len(losses)}\n\t\tval losses: {sum(val_losses) / len(val_losses)}"
    )
    logging.info(
        f"\t\tTime spent: {datetime.datetime.now() - epoch_start}\n"
        f"\t\ttrain losses: {sum(losses) / len(losses)}\n\t\tval losses: {sum(val_losses) / len(val_losses)}")

    # Check if validation mse is the best during whole training
    if sum(val_losses) / len(val_losses) < best_val_mse:
        # Saves model with best validation mse
        best_val_mse = sum(val_losses) / len(val_losses)
        torch.save(encoder.state_dict(), "best_val_mse_encoder_state_dict.pt")
        torch.save(decoder.state_dict(), "best_val_mse_decoder_state_dict.pt")
        best_val_mse_epoch = epoch + 1

    with open(train_log_path, 'a') as log_f:
        writer = csv.writer(log_f, delimiter='|')
        writer.writerow([epoch + 1, (sum(losses) / len(losses)).item(), (sum(val_losses) / len(val_losses)).item()])

# Testing phase

autoencoder.eval()

print("Testing")
logging.info("Testing")
test_start = datetime.datetime.now()
test_losses = []
random_test_fingerprints_to_check = []
# Define test sample to check search
while len(random_test_fingerprints_to_check) != FINGERPRINTS_TO_SEARCH_AMOUNT:
    fingerprint_index = randint(0, len(test_dataset) - 1)
    if fingerprint_index not in random_test_fingerprints_to_check:
        random_test_fingerprints_to_check.append(fingerprint_index)

with torch.no_grad():
    # shown_decoded = 0  # TODO remove
    for test_altered_image, test_real_image, test_person_number in test_loader:
        test_altered_image = test_altered_image.type(torch.float64).to(device)
        test_real_image = test_real_image.type(torch.float64).to(device)
        test_processed_image = autoencoder(test_real_image)
        test_step_loss = loss_function(test_processed_image, test_real_image)
        test_losses.append(test_step_loss)

        # TODO remove
        # while shown_decoded < 5:
        #     shown_decoded += 1
        #     plt.imshow(test_processed_image[shown_decoded].reshape(105, 105, 1))
        #     plt.title(f"Check, decoded: {shown_decoded}")
        #     plt.show()
        #
        #     plt.imshow(test_altered_image[shown_decoded].reshape(105, 105, 1))
        #     plt.title(f"Check, before processing: {shown_decoded}")
        #     plt.show()
        # del test_processed_image

        # Counting latent vectors
        if COUNT_LATENT_VECTORS_FROM_REAL_IMAGE:
            encoded_fingerprint = encoder(test_real_image)
        else:
            encoded_fingerprint = encoder(test_altered_image)  # Only the latest fingerprint will be saved in the dictionary

        del test_altered_image
        del test_real_image

        for i in range(encoded_fingerprint.shape[0]):
            latent_database_dict[test_person_number[i]] = encoded_fingerprint[i].reshape(1, -1)

    print("\tTime spent on test: {}".format(datetime.datetime.now() - test_start))
    print("\ttest losses: {}".format(sum(test_losses)/len(test_losses)))
    print("Time spent at all: {}".format(datetime.datetime.now() - start))
    logging.info(
        f"\tTime spent on test: {datetime.datetime.now() - test_start}\n"
        f"\ttest losses: {sum(test_losses) / len(test_losses)}\n"
        f"Time spent at all: {datetime.datetime.now() - start}"
    )

    # Search for fingerprints
    real_right = 0
    altered_right = 0
    iter = -1
    for fingerprint_index in random_test_fingerprints_to_check:
        iter += 1
        altered_fingerprint_to_search, real_fingerprint_to_search, person_number_to_search = test_dataset[fingerprint_index]

        altered_fingerprint_to_search = altered_fingerprint_to_search.type(torch.float64).to(device)
        real_fingerprint_to_search = real_fingerprint_to_search.type(torch.float64).to(device)
        #person_number_to_search = person_number_to_search.type(torch.float64).to(device)

        encoded_altered_fingerprint_to_search = encoder(altered_fingerprint_to_search.reshape(1, 1, 105, 105))
        encoded_real_fingerprint_to_search = encoder(real_fingerprint_to_search.reshape(1, 1, 105, 105))

        # Search for variants for encoded altered fingerprint
        real_right += search(
            encoded_real_fingerprint_to_search,
            altered_fingerprint_to_search,
            person_number_to_search,
            (105, 105),
            print_results=True if iter < 3 else False
        )

        altered_right += search(
            encoded_altered_fingerprint_to_search,
            altered_fingerprint_to_search,
            person_number_to_search,
            (105, 105),
            search_for_real=False,
            print_results=True if 3 < iter < 6 else False
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
    torch.save(encoder.state_dict(), "after_last_epoch_encoder_state_dict.pt")
    torch.save(decoder.state_dict(), "after_last_epoch_decoder_state_dict.pt")

print(f"Best validation mse is counted on epoch {best_val_mse_epoch}")
logging.info(f"Best validation mse is counted on epoch {best_val_mse_epoch}")
plot_log(train_log_path)
