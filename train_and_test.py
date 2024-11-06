# Author: Vodohleb04
import sys
import glob
import re
import csv
import datetime
import threading
import logging
import asyncio
from random import shuffle, randint
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import chromadb

from vae.vae_decoder import create_vae_decoder
from vae.vae_encoder import res_net_50_v2_vae_encoder
from vae.variational_autoencoder import VariationalAutoencoder
from vae.vae_loss import VAELoss
from dataset_loader import CustomDataset


COUNT_LATENT_VECTORS_FROM_REAL_IMAGE = True  # Counts latent vectors of fingerprint from real images if set to True
FINGERPRINTS_TO_SEARCH_AMOUNT = 400
LIMIT_RETURNING_VARIANTS = 3
IMG_SHAPE_WITH_PAD = (109, 109)

latent_database_dict = {}  # Dict to save latent vectors of fingerprints
train_log_path = "./vae_fingerprint_ae_train_log.csv"
log_text_file = './vae_fingerprint_ae_text_log.log'
logging.basicConfig(level=logging.INFO, filename=log_text_file, filemode="w")
regexp_filename = re.compile(r"^(?P<real_filename>.+finger).*?\.BMP$")
#regexp_filename = re.compile(r"^(?P<real_filename>\d+)__(M|F)_Left_index_.*$")

real_images_dir_path = "/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/Real"

altered_images_dir_paths = [
    "/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/Altered/Altered-Easy",
    "/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/Altered/Altered-Hard",
    "/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/Altered/Altered-Medium",
    "/home/vodohleb/PycharmProjects/fingerprint/SOCOFing/AlteredByMeNoRotation"
]


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


def validate_input():
    try:
        latent_dim = 384
        epochs_amount = 10
        return latent_dim, epochs_amount
    except Exception as exp:
        print("Error! Takes 2 positional arguments: latent_dim (int > 0) and epochs_amount (int > 0)")
        logging.error("Error! Takes 2 positional arguments: latent_dim (int > 0) and epochs_amount (int > 0)")
        sys.exit(-1)


async def get_database_collection():
    client = await chromadb.AsyncHttpClient(
        host="0.0.0.0", port=8000
        #,settings=chromadb.config.Settings(allow_reset=True, anonymized_telemetry=False)
    )
    collection = await client.get_or_create_collection(name="fingerptints", metadata={"hnsw:space": "l2"})
    
    return collection


def clear_log_file():
    with open(train_log_path, 'w') as f:
        f.write("")


def get_real_images_paths(real_images_dir_path):
    altered_images_paths_dict = {}
    real_images_paths_dict = {}

    real_counter = 0
    # Get paths of fingerprint images
    for image_path in glob.glob(real_images_dir_path + "/*"):
        match = regexp_filename.search(image_path.split("/")[-1])
        if match is not None:
            real_images_paths_dict[match.group("real_filename")] = image_path  # Get real fingerprints paths
            altered_images_paths_dict[match.group("real_filename")] = []
            real_counter += 1

    print(f"Real images found: {real_counter}")
    logging.info(f"Real images found: {real_counter}")
    return real_images_paths_dict, altered_images_paths_dict


def get_altered_images_paths(altered_images_paths_dict, altered_images_dir_paths_dict):

    def actions_on_match():
        nonlocal altered_images_paths_dict, match, image_path, altered_counter
        try:
            altered_images_paths_dict[match.group("real_filename")].append(image_path)  # Get altered fingerprint paths
            altered_counter += 1
        except KeyError:
            print(f"filename: {image_path}, number:{match.group('real_filename')}")
            logging.error(f"filename: {image_path}, number:{match.group('real_filename')}")


    altered_counter = 0
    for altered_images_dir_path in altered_images_dir_paths_dict:
        for image_path in glob.glob(altered_images_dir_path + "/*"):
            match = regexp_filename.search(image_path.split("/")[-1])
            if match is not None:
                actions_on_match()

    print(f"Altered images found: {altered_counter}")
    logging.info(f"Altered images found: {altered_counter}")
    
    return altered_images_paths_dict, altered_counter


def get_images_paths():
    global real_images_dir_path, altered_images_dir_paths
    real_images_paths_dict, altered_images_paths_dict = get_real_images_paths(
        real_images_dir_path
    )

    altered_images_paths_dict, altered_counter = get_altered_images_paths(
        altered_images_paths_dict, altered_images_dir_paths
    )
    
    return real_images_paths_dict, altered_images_paths_dict, altered_counter


def unpack_paths_dicts(real_images_paths_dict, altered_images_paths_dict, altered_counter):
    altered_images_list = [None for _ in range(altered_counter)]
    real_images_list = [None for _ in range(altered_counter)]
    person_numbers_list = [None for _ in range(altered_counter)]

    # Shuffle and division on subsets (train, validation, test)
    altered_images_paths_items = list(altered_images_paths_dict.items())
    shuffle(altered_images_paths_items)
    i = 0
    for person_number, altered_images_paths_dict in altered_images_paths_items:
        for altered_image_path in altered_images_paths_dict:
            altered_images_list[i] = altered_image_path
            real_images_list[i] = real_images_paths_dict[person_number]
            person_numbers_list[i] = person_number
            i += 1
    
    return real_images_list, altered_images_list, person_numbers_list


def init_custom_dataset(altered_images, real_images, person_numbers):
    return CustomDataset(
        x_=altered_images, y_=real_images, numbers_=person_numbers,
        transform=transforms.Compose([transforms.ToTensor()]),
        padding_size=(IMG_SHAPE_WITH_PAD[0] - 103, IMG_SHAPE_WITH_PAD[1] - 96),
        inverse=True
    )


def divide_on_train_validation_test(
    real_images_list, altered_images_list, person_numbers_list, altered_counter
):
    train_size = int(altered_counter * 0.65)
    test_size = int(altered_counter * 0.25)
    validation_size = altered_counter - train_size - test_size

    train_dataset = init_custom_dataset(
        altered_images_list[:train_size],
        real_images_list[:train_size],
        person_numbers_list[:train_size]
    )
    validation_dataset = init_custom_dataset(
        altered_images_list[train_size: train_size + validation_size],
        real_images_list[train_size: train_size + validation_size],
        person_numbers_list[train_size: train_size + validation_size]
    )
    test_dataset = init_custom_dataset(
        altered_images_list[train_size + validation_size:],
        real_images_list[train_size + validation_size:],
        person_numbers_list[train_size + validation_size:]
    )
    return train_dataset, validation_dataset, test_dataset


def create_datasets():
    def message():
        nonlocal removed, altered_counter, train_dataset, validation_dataset, test_dataset
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
        
    real_images_paths_dict, altered_images_paths_dict, altered_counter = get_images_paths()

    real_images_list, altered_images_list, person_numbers_list = unpack_paths_dicts(
        real_images_paths_dict, altered_images_paths_dict, altered_counter
    )    
    
    train_dataset, validation_dataset, test_dataset = divide_on_train_validation_test(
        real_images_list, altered_images_list, person_numbers_list, altered_counter
    )
    # Remove bad examples
    removed = remove_images_of_wrong_shape(train_dataset, (IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1]))
    removed += remove_images_of_wrong_shape(validation_dataset, (IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1]))
    removed += remove_images_of_wrong_shape(test_dataset, (IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1]))

    message()

    return train_dataset, validation_dataset, test_dataset


def create_dataloaders():
    # Create datasets
    train_dataset, validation_dataset, test_dataset = create_datasets()

    # Create dataloaders
    train_loader = DataLoader(train_dataset, 64, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, 1, shuffle=True)
    test_loader = DataLoader(test_dataset, 1)

    return train_loader, validation_loader, test_loader


def init_networks(
    device,
    latent_dim,
    dropblock_conv_keep_p, dropblock_sizes,
    dropout_conv_transpose_keep_p, dropout_linear_keep_p,
    dtype
):
    encoder = res_net_50_v2_vae_encoder(
        latent_dim, dropblock_conv_keep_p, dropblock_sizes, dropout_linear_keep_p, device, dtype
    )

    decoder = create_vae_decoder(
        latent_dim, dropout_conv_transpose_keep_p,  dropout_linear_keep_p, device, dtype
    )

    autoencoder = VariationalAutoencoder(encoder, decoder).to(device)

    return autoencoder


def loss_gradient_step(
    loss_function, optimizer, dataset_length, real_image, decoded_image, z_mean, z_logvar
):
    step_loss = loss_function(real_image, decoded_image, z_mean, z_logvar)
    step_loss.backward()
    optimizer.step()

    return step_loss.detach() / dataset_length  # mean loss += step_loss


def train_phase(
    variational_autoencoder, loss_function, optimizer,
    train_loader,
    device, dtype
):
    losses = torch.Tensor([0.0]).type(dtype).to(device)
    variational_autoencoder.train()
    i = 0
    for altered_image, real_image, _ in train_loader:
        optimizer.zero_grad(set_to_none=True)

        i += 1
        print(f"\t\tTrain mini-batch number: {i}")
        logging.info(f"\t\tTrain mini-batch number: {i}")

        altered_image = altered_image.type(dtype).to(device)
        real_image = real_image.type(dtype).to(device)

        decoded_image, z_mean, z_logvar = variational_autoencoder(altered_image)

        # Gradient step
        losses += loss_gradient_step(
            loss_function, optimizer, len(train_loader.dataset), real_image, decoded_image, z_mean, z_logvar
        )
    return losses

        
def validation_phase(
    variational_autoencoder, loss_function, validation_loader, device, dtype
):
    losses = torch.Tensor([0.0]).type(dtype).to(device)
    variational_autoencoder.eval()
    for val_altered_image, val_real_image, _ in validation_loader:
        val_altered_image = val_altered_image.type(dtype).to(device)
        val_real_image = val_real_image.type(dtype).to(device)

        val_decoded_image, val_z_mean, val_z_logvar = variational_autoencoder(val_altered_image)

        step_loss = loss_function(val_real_image, val_decoded_image, val_z_mean, val_z_logvar)

        losses += step_loss.detach() / len(validation_loader.dataset)
    return losses


def save_models(variational_autoencoder, encoder_save_file, decoder_save_file):
    encoder_save_thread_best_val_loss = threading.Thread(
        target=torch.save,
        args=(variational_autoencoder.encoder.state_dict(), encoder_save_file)
    )
    encoder_save_thread_best_val_loss.start()

    decoder_save_thread_best_val_loss = threading.Thread(
        target=torch.save,
        args=(variational_autoencoder.decoder.state_dict(), decoder_save_file)
    )
    decoder_save_thread_best_val_loss.start()
    
    encoder_save_thread_best_val_loss.join()
    decoder_save_thread_best_val_loss.join()


def check_for_best_validation_loss(
    variational_autoencoder, best_val_loss, best_val_loss_epoch, validation_losses, epoch
):
    if validation_losses < best_val_loss:
        # Saves model with best validation loss
        save_models(
            variational_autoencoder,
            "vae_best_val_loss_encoder_state_dict.pt",
            "vae_best_loss_decoder_state_dict.pt"
        )

        return validation_losses, epoch
    else:
        return best_val_loss, best_val_loss_epoch


def train_and_validate(
    variational_autoencoder, loss_function, optimizer,
    train_loader, validation_loader,
    epochs_amount, device, dtype
):
    def train_message(epoch, epochs_amount):
        print(f"Epoch: {epoch}/{epochs_amount}\n\tTrain")
        logging.info(f"Epoch: {epoch}/{epochs_amount}\n\tTrain")

    def validation_message():
        print("\tValidation")
        logging.info("\tValidation")

    def time_spent_and_losses_on_epoch_message(epoch_start, losses, validation_losses):
        print(
            f"\t\tTime spent: {datetime.datetime.now() - epoch_start}\n"
            f"\t\ttrain losses: {losses.item()}\n\t\tval losses: {validation_losses.item()}"
        )
        logging.info(
            f"\t\tTime spent: {datetime.datetime.now() - epoch_start}\n"
            f"\t\ttrain losses: {losses.item()}\n\t\tval losses: {validation_losses.item()}"
        )

    def csv_epoch_log(epoch, losses, validation_losses):
        global train_log_path
        with open(train_log_path, 'a') as log_f:
            writer = csv.writer(log_f, delimiter='|')
            writer.writerow([epoch, losses.item(), validation_losses.item()])


    start = datetime.datetime.now()
    best_val_loss = torch.Tensor([1e10]).type(dtype).to(device)
    best_val_loss_epoch = 1
    for epoch in range(1, epochs_amount + 1):
        epoch_start = datetime.datetime.now()
        train_message(epoch, epochs_amount)
        losses = train_phase(
            variational_autoencoder, loss_function, optimizer, train_loader, device, dtype
        )
        
        validation_message()
        with torch.no_grad():  # Gradient is not needed during validation
            validation_losses = validation_phase(
                variational_autoencoder, loss_function, validation_loader, device, dtype
            )

        time_spent_and_losses_on_epoch_message(epoch_start, losses, validation_losses)
        
        # Check if validation loss is the best during whole training
        best_val_loss, best_val_loss_epoch = check_for_best_validation_loss(
            variational_autoencoder, best_val_loss, best_val_loss_epoch, validation_losses, epoch
        )
        csv_epoch_log(epoch, losses, validation_losses)
    
    return start, best_val_loss_epoch


def get_indexes_of_fingerprints_to_check(test_dataset_length):
    random_test_fingerprints_to_check = []
    # Define test sample to check search
    while len(random_test_fingerprints_to_check) != FINGERPRINTS_TO_SEARCH_AMOUNT:
        fingerprint_index = randint(0, test_dataset_length - 1)
        if fingerprint_index not in random_test_fingerprints_to_check:
            random_test_fingerprints_to_check.append(fingerprint_index)
    
    return random_test_fingerprints_to_check


def test_phase(
    variational_autoencoder, loss_function, test_loader, device, dtype, test_altered_image, test_real_image, j
):
    def show_real_decoded_img(variational_autoencoder, test_real_image, j):
        plt.imshow(test_real_image.cpu().reshape(IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1], 1))
        plt.title(f"Check, before processing: {j}")
        plt.show()
        mean = variational_autoencoder.encoder(test_real_image)[0]
        plt.imshow(variational_autoencoder.decoder(mean).cpu().reshape(IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1], 1))
        plt.title(f"Check, decoded: {j}")
        plt.show()


    test_altered_image = test_altered_image.type(dtype).to(device)
    test_real_image = test_real_image.type(dtype).to(device)

    test_decoded_image, test_z_mean, test_z_logvar = variational_autoencoder(test_altered_image)
    test_step_loss = loss_function(test_real_image, test_decoded_image, test_z_mean, test_z_logvar)
    

    if j < 90 and j % 9 == 0:  # TODO remove
        show_real_decoded_img(variational_autoencoder, test_real_image, j)
    
    return test_z_mean, test_z_logvar, test_step_loss.detach() / len(test_loader.dataset)


async def test_and_save_to_db(
    collection, variational_autoencoder, loss_function, test_loader, device, dtype, start
):
    def test_start_message():
        print("Testing")
        logging.info("Testing")

    def test_end_message(start, test_losses, test_start):
        print(
            f"\tTime spent on test: {datetime.datetime.now() - test_start}"
            f"\ttest losses: {test_losses.item()}"
            f"\tTime spent at all: {datetime.datetime.now() - start}"
        )
        logging.info(
        f"\tTime spent on test: {datetime.datetime.now() - test_start}\n"
        f"\ttest losses: {test_losses.item()}\n"
        f"\tTime spent at all: {datetime.datetime.now() - start}"
    )


    test_start_message()
    test_start = datetime.datetime.now()
    test_losses = torch.Tensor([0.0]).type(dtype).to(device)

    j = 0  # TODO remove
    for test_altered_image, test_real_image, test_person_number in test_loader:
        test_z_mean, test_z_logvar, loss_on_step = test_phase(
            variational_autoencoder, loss_function, test_loader, device, dtype, test_altered_image, test_real_image, j
        )
        test_losses += loss_on_step
        j += 1
        # Save to the database
        if COUNT_LATENT_VECTORS_FROM_REAL_IMAGE:
            fingerprint_mean, fingerprint_logvar = test_z_mean.clone(), test_z_logvar.clone()
        else:
            # Only the latest fingerprint will be saved in the dictionary
            fingerprint_mean, fingerprint_logvar = variational_autoencoder.encoder(test_altered_image)
        
        await collection.add(
            embeddings=torch.cat(
                (fingerprint_mean.detach().cpu(), fingerprint_logvar.detach().cpu()), dim=1
            ).tolist(),
            ids=list(test_person_number)
        )
    test_end_message(start, test_losses, test_start)


async def test_db_query(
    collection, variational_autoencoder, test_dataset, device, dtype, fingerprints_to_check
):
    def message(fingerprints_to_check, real_right, altered_right):
        global train_log_path
        print(
            f"Real fingerprint accuracy: {real_right / len(fingerprints_to_check)}\n"
            f"Altered fingerprint accuracy: {altered_right / len(fingerprints_to_check)}"
        )
        logging.info(
            f"Real fingerprint accuracy: {real_right / len(fingerprints_to_check)}\n"
            f"Altered fingerprint accuracy: {altered_right / len(fingerprints_to_check)}"
        )
    # Search for fingerprints
    true_positive = 0
    false_positive = 0
    false_negative = 0
    false_positive = 0
    
    iter = -1
    for fingerprint_index in fingerprints_to_check:
        iter += 1
        altered_to_search, real_to_search, person_number_to_search = test_dataset[fingerprint_index]

        altered_to_search = altered_to_search.type(dtype).to(device)
        real_to_search = real_to_search.type(dtype).to(device)

        altered_mean, altered_logvar = variational_autoencoder.encoder(
            altered_to_search.reshape(1, 1, IMG_SHAPE_WITH_PAD[0], IMG_SHAPE_WITH_PAD[1])
        )
    
        results = await collection.query(
            
            query_embeddings=torch.cat(
                (altered_mean.detach().cpu(), altered_logvar.detach().cpu()), dim=1
            ).tolist(),
            n_results=LIMIT_RETURNING_VARIANTS,
            include=["distances"]
        )
        print(person_number_to_search[0])
        print(results["ids"])
        if person_number_to_search[0] in results["ids"]:
            true_positive += 1
        else:
            false_negative += 1
            false_positive += 1
        

        # TODO metrics
        # message()
    print(true_positive / len(fingerprints_to_check))


async def main():
    latent_dim, epochs_amount = validate_input()
    dtype = torch.float32

    collection = await get_database_collection()
    clear_log_file()

    # Defining device (Try to connect cuda, cpu is used otherwise)
    device = define_device()
    train_loader, validation_loader, test_loader = create_dataloaders()

    variational_autoencoder = init_networks(device, latent_dim, 1, [7, 5, 3, 3], 1, 1, dtype)
    loss_function = VAELoss()
    optimizer = torch.optim.Adam(variational_autoencoder.parameters(), lr=1e-4)

    start, best_val_loss_epoch = train_and_validate(
        variational_autoencoder, loss_function, optimizer,
        train_loader, validation_loader,
        epochs_amount, device, dtype
    )

    random_test_fingerprints_to_check = get_indexes_of_fingerprints_to_check(len(test_loader.dataset))

    with torch.no_grad():
        await test_and_save_to_db(
            collection, variational_autoencoder, loss_function, test_loader, device, dtype, start
        )
    
        await test_db_query(
            collection, variational_autoencoder, test_loader.dataset, device, dtype, random_test_fingerprints_to_check
        )
    
    save_models(
        variational_autoencoder,
        "vae_after_last_epoch_encoder_state_dict",
        "vae_after_last_epoch_decoder_state_dict.pt"
    )

    print(f"Best validation loss is counted on epoch {best_val_loss_epoch}")
    logging.info(f"Best validation loss is counted on epoch {best_val_loss_epoch}")
    plot_log(train_log_path)


if __name__ == "__main__":
    asyncio.run(main())






# TODO No dropout and BatchNormalization in VAE  (make additional classes for it)
# TODO Group files
# TODO Group code into modules
# TODO Add chromadb
# TODO Save mu and logvar of z (not onlu mu)   maybe save  mu + exp(z_logvar * 0.5)
# TODO Recall, precision and accuracy
# TODO Check if can make p(x | z) = Gaussian distridution
# TODO metrics
# TODO Check if there is any memmory loss

