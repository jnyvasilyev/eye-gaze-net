#!/usr/bin/env python
# coding: utf-8
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import cv2
import json
import matplotlib.pyplot as plt
from ipdb import set_trace
from glob import glob
from tqdm import tqdm

from model import ECCNet
from warp import WarpImageWithFlowAndBrightness
from warp import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_json_list(json_list):
    ldmks = [eval(s) for s in json_list]
    return np.array([(x, y, z) for (x, y, z) in ldmks])


def get_filename_info(filename):
    info_dict = {}

    # Get ID data
    titles_types = {"ID": int, "T": str, "N": int, "F": int, "V": float, "H": float}
    info_list = os.path.basename(filename[:-5]).split("_")
    for title, info in zip(titles_types, info_list):
        info_dict[title] = titles_types[title](info[len(title) :])
    info_dict["target"] = info_dict["F"] == 1

    # Get json data
    json_data_file = open(filename)
    json_data = json.load(json_data_file)
    info_dict["look_vec"] = np.array(
        eval(json_data["eye_details"]["look_vec"])
    )  # 3D look direction vector
    info_dict["interior_margin"] = process_json_list(
        json_data["interior_margin_2d"]
    )  # eye interior landmarks
    info_dict["caruncle"] = process_json_list(
        json_data["caruncle_2d"]
    )  # caruncle landmarks
    info_dict["iris"] = process_json_list(json_data["iris_2d"])  # iris landmarks
    json_data_file.close()

    # Get image data
    img = cv2.cvtColor(cv2.imread("%s.jpg" % filename[:-5]), cv2.COLOR_BGR2RGB)
    eye_hor_range = (
        np.max(info_dict["interior_margin"], axis=0)[0]
        - np.min(info_dict["interior_margin"], axis=0)[0]
    )
    eye_vert_range = (
        np.max(info_dict["interior_margin"], axis=0)[1]
        - np.min(info_dict["interior_margin"], axis=0)[1]
    )
    if eye_hor_range // 2 > eye_vert_range:
        eye_vert_range = eye_hor_range // 2
    else:
        eye_hor_range = eye_vert_range * 2
    eyeball_center = np.array([300, 400])
    range_coef = 2
    img_cropped = img[
        eyeball_center[0]
        - int(eye_vert_range // 2 * range_coef) : eyeball_center[0]
        + int(eye_vert_range // 2 * range_coef),
        eyeball_center[1]
        - int(eye_hor_range // 2 * range_coef) : eyeball_center[1]
        + int(eye_hor_range // 2 * range_coef),
    ]
    img = cv2.resize(img_cropped, (64, 32))
    img = np.transpose(img, (2, 0, 1))
    info_dict["img"] = img

    return info_dict


# ### Notes on the dataframe info
# - look_vec is a 3D homogeneous vector in the form \[x, y, z, 0\]. For purposes of the model input, only the x and y components are needed. I believe this vector is already normalized, but we can renormalize this vector before extracting the x and y components
# - Feature landmarks (i.e., interior_margin, caruncle, and iris) are the pixel coordinates of feature BEFORE RESIZING. Ideally they are used to determine the crop/resize area
#

# ### Dataloader
# The above dataframe contains info on all images, separated into IDs. As our model input, we would like to take all possible image pairs within an ID. With 40 images per ID, this results in 780 pairs.
#
# Dataloader will output input image, input vector, target image, target vector

# In[18]:


# Read dataset folder
def get_dataset(input_file_path, input_fn):
    """
    Read the image and json data in the specified folder.
    Args:
        input_file_path: the base directory containing the dataset directories
        input_fn: the name of dataset directory
    Return:
        imgs_list (list): List of images, cropped
        vecs_list (list): List of gaze vectors, tiled
    """
    img_infos = []
    json_fns = glob(os.path.join(input_file_path, input_fn, "*.json"))
    for json_fn in json_fns:
        info = get_filename_info(json_fn)
        img_infos.append(info)

    img_df = pd.DataFrame(img_infos)
    img_df.sort_values(["ID", "F"], ignore_index=True, inplace=True)

    # Extract relevant data for dataloader
    n_ids = img_df.iloc[-1]["ID"]

    imgs_list = []
    vecs_list = []

    # For each ID, generate all possible pairs
    for id in range(1, n_ids + 1):
        # Get ID
        df_chunk = img_df.query(f"ID == {id}")
        imgs = np.stack(df_chunk["img"])
        vecs = np.stack(df_chunk["look_vec"].to_numpy())
        vecs = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True))[
            :, :2
        ]  # normalize, then get x and y components
        vecs = np.tile(vecs[:, :, np.newaxis, np.newaxis], (1, 1, 32, 64))

        # Generate pairs
        pairs = np.triu_indices(len(df_chunk), k=1)
        pairs = np.stack(pairs).transpose()

        img_pairs = imgs[pairs]  # (pair_idx, input or output, C, H, W)
        vec_pairs = vecs[pairs]  # (pair_idx, input or output, x or y, ...)

        imgs_list.append(img_pairs)
        vecs_list.append(vec_pairs)

    return imgs_list, vecs_list


def get_dataloader(input_file_path, input_filename_list, batch_size=800):
    imgs_list_full = []
    vecs_list_full = []

    print("Reading dataset")
    for input_fn in tqdm(input_filename_list):
        print("Reading " + input_fn)
        imgs_list_part, vecs_list_part = get_dataset(input_file_path, input_fn)

        imgs_list_full.append(imgs_list_part)
        vecs_list_full.append(vecs_list_part)

    # In[10]:
    imgs_list_full = np.concatenate(np.concatenate(imgs_list_full))
    vecs_list_full = np.concatenate(np.concatenate(vecs_list_full))

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    print("Shuffling data")
    imgs_list_full, vecs_list_full = unison_shuffled_copies(
        imgs_list_full, vecs_list_full
    )

    print(f"Dataset size: {len(imgs_list_full)}")

    # In[11]:

    # Create splits
    num_samples = len(imgs_list_full)
    splits = [0.7, 0.2, 0.1]

    X_img = imgs_list_full[:, 0, ...]
    X_angle = vecs_list_full[:, 0, ...]
    y_img = imgs_list_full[:, 1, ...]
    y_angle = vecs_list_full[:, 1, ...]

    X_img_train, X_img_valid, X_img_test = np.split(
        X_img,
        [int(num_samples * splits[0]), int(num_samples * (splits[0] + splits[1]))],
    )
    X_angle_train, X_angle_valid, X_angle_test = np.split(
        X_angle,
        [int(num_samples * splits[0]), int(num_samples * (splits[0] + splits[1]))],
    )
    y_img_train, y_img_valid, y_img_test = np.split(
        y_img,
        [int(num_samples * splits[0]), int(num_samples * (splits[0] + splits[1]))],
    )
    y_angle_train, y_angle_valid, y_angle_test = np.split(
        y_angle,
        [int(num_samples * splits[0]), int(num_samples * (splits[0] + splits[1]))],
    )

    # In[12]:

    X_img_train = torch.tensor(X_img_train, dtype=torch.int)
    X_angle_train = torch.tensor(X_angle_train, dtype=torch.float32)
    y_img_train = torch.tensor(y_img_train, dtype=torch.int)
    y_angle_train = torch.tensor(y_angle_train, dtype=torch.int)

    X_img_valid = torch.tensor(X_img_valid, dtype=torch.int)
    X_angle_valid = torch.tensor(X_angle_valid, dtype=torch.float32)
    y_img_valid = torch.tensor(y_img_valid, dtype=torch.int)
    y_angle_valid = torch.tensor(y_angle_valid, dtype=torch.int)

    X_img_test = torch.tensor(X_img_test, dtype=torch.int)
    X_angle_test = torch.tensor(X_angle_test, dtype=torch.float32)
    y_img_test = torch.tensor(y_img_test, dtype=torch.int)
    y_angle_test = torch.tensor(y_angle_test, dtype=torch.int)

    train_dataset = TensorDataset(
        X_img_train, X_angle_train, y_img_train, y_angle_train
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(
        X_img_valid, X_angle_valid, y_img_valid, y_angle_valid
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(X_img_test, X_angle_test, y_img_test, y_angle_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def get_model_optimizer(ckpt_iter):
    # Create the model
    model = ECCNet()
    model.to(device)
    # model = nn.DataParallel(model) # Comment out if using only one GPU
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=0.1)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.002, max_lr=0.01)
    start = 0

    # Load checkpoint
    if ckpt_iter:
        ckpt_path = f"./checkpoints/ckpt_{ckpt_iter}.pt"
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start = checkpoint["epoch"] + 1

    return model, optimizer, scheduler, start


def train(
    model,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    warp,
    criterion=nn.MSELoss(),
    start_epoch=0,
    num_epochs=100,
    weight_correction_loss=0.8,
    weight_reconstruction_loss=0.2,
):
    print("Beginning training")
    printout_freq = 1
    save_ckpt_freq = 1

    train_checkpoint_path = "./checkpoints"
    os.makedirs(train_checkpoint_path, exist_ok=True)

    images_directory_path = "./images"
    os.makedirs(images_directory_path, exist_ok=True)

    progress_plot_path = "./progress"
    os.makedirs(progress_plot_path, exist_ok=True)
    start_time = time.time()

    train_losses, valid_losses, epochs = [], [], []

    pbar = tqdm(range(start_epoch, start_epoch + num_epochs))

    # Placeholders for displaying last batch of images
    img_corr_display = None
    target_display = None

    for epoch in pbar:
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        for imgs, angles, targets, target_angles in train_loader:
            imgs, angles, targets, target_angles = (
                imgs.float().to(device),
                angles.float().to(device),
                targets.float().to(device),
                target_angles.float().to(device),
            )

            flow_corr, bright_corr = model(imgs, target_angles)
            img_corr = warp(imgs, flow_corr, bright_corr)
            loss_correction = criterion(img_corr, targets)

            flow_reconstruction, bright_reconstruction = model(img_corr, angles)
            img_reconstruction = warp(
                img_corr, flow_reconstruction, bright_reconstruction
            )
            loss_reconstruction = criterion(img_reconstruction, imgs)

            loss = (weight_correction_loss * loss_correction) + (
                weight_reconstruction_loss * loss_reconstruction
            )
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        train_losses.append(train_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for imgs, angles, targets, target_angles in valid_loader:
                imgs, angles, targets, target_angles = (
                    imgs.float().to(device),
                    angles.float().to(device),
                    targets.float().to(device),
                    target_angles.float().to(device),
                )

                flow_corr, bright_corr = model(imgs, target_angles)
                img_corr = warp(imgs, flow_corr, bright_corr)
                loss_correction = criterion(img_corr, targets)

                flow_reconstruction, bright_reconstruction = model(img_corr, angles)
                img_reconstruction = warp(
                    img_corr, flow_reconstruction, bright_reconstruction
                )
                loss_reconstruction = criterion(img_reconstruction, imgs)

                loss = (weight_correction_loss * loss_correction) + (
                    weight_reconstruction_loss * loss_reconstruction
                )
                valid_loss += loss.item()
            valid_losses.append(valid_loss / len(valid_loader))

            epoch_time = time.time() - epoch_start_time
            overall_time = time.time() - start_time
            num_days = int(overall_time / 86400)
            num_hrs = int((overall_time - (86400 * num_days)) / 3600)
            num_mins = int((overall_time - (86400 * num_days) - (3600 * num_hrs)) / 60)
            num_secs = (
                overall_time - (86400 * num_days) - (3600 * num_hrs) * (60 * num_mins)
            )
            epochs.append(epoch + 1)

            if (epoch + 1) % printout_freq == 0:
                pbar.set_postfix(
                    {"TLoss": f"{train_loss:.3f}", "VLoss": f"{valid_loss:.3f}"}
                )

            if (epoch + 1) % save_ckpt_freq == 0:
                # Save checkpoint
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),  # model.state_dict() if using only one GPU ; model.module.state_dict() if parallel
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "validation_loss": valid_loss,
                    },
                    os.path.join(train_checkpoint_path, f"ckpt_{epoch+1}.pt"),
                )

                # Save example images
                img_corr_display = img_corr.clone()
                target_display = targets.clone()
                save_image(
                    img_corr_display,
                    os.path.join(
                        images_directory_path, f"img_corr_epoch_{epoch+1}.png"
                    ),
                )
                save_image(
                    target_display,
                    os.path.join(images_directory_path, f"targets_epoch_{epoch+1}.png"),
                )

                # Create loss plot
                plt.figure(figsize=(15, 15))
                plt.title("Training Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.plot(
                    epochs,
                    train_losses,
                    color="green",
                    label="Training Loss",
                    marker="o",
                    markerfacecolor="green",
                )
                plt.plot(
                    epochs,
                    valid_losses,
                    color="red",
                    linewidth=1.0,
                    linestyle="--",
                    label="Validation Loss",
                    marker="o",
                    markerfacecolor="red",
                )

                plt.xticks(ticks=epochs)
                plt.legend()
                plt.savefig(
                    os.path.join(progress_plot_path, f"progress_epoch_{epoch + 1}.png")
                )
                plt.clf()


def test(
    model,
    test_loader,
    warp,
    criterion=nn.MSELoss(),
    weight_correction_loss=0.8,
    weight_reconstruction_loss=0.2,
):
    print("Evaluating model")

    images_directory_path = "./test_images"
    os.makedirs(images_directory_path, exist_ok=True)

    # Placeholders for displaying last batch of images
    img_display = None
    img_corr_display = None
    target_display = None

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for imgs, angles, targets, target_angles in test_loader:
            imgs, angles, targets, target_angles = (
                imgs.float().to(device),
                angles.float().to(device),
                targets.float().to(device),
                target_angles.float().to(device),
            )

            flow_corr, bright_corr = model(imgs, target_angles)
            img_corr = warp(imgs, flow_corr, bright_corr)
            loss_correction = criterion(img_corr, targets)

            flow_reconstruction, bright_reconstruction = model(img_corr, angles)
            img_reconstruction = warp(
                img_corr, flow_reconstruction, bright_reconstruction
            )
            loss_reconstruction = criterion(img_reconstruction, imgs)

            loss = (weight_correction_loss * loss_correction) + (
                weight_reconstruction_loss * loss_reconstruction
            )
            test_loss += loss.item()

            img_display = imgs.clone()

        # Save example images
        img_corr_display = img_corr.clone()
        target_display = targets.clone()
        save_image(
            img_display,
            os.path.join(images_directory_path, f"img_og.png"),
        )
        save_image(
            img_corr_display,
            os.path.join(images_directory_path, f"img_corr.png"),
        )
        save_image(
            target_display,
            os.path.join(images_directory_path, f"targets.png"),
        )

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true", default=False)

    args = parser.parse_args()

    print(f"Using device {device}")
    eccmodel, eccoptimizer, eccscheduler, start = get_model_optimizer(args.checkpoint)
    input_filename_list = [
        "imgs_1_cutouts",
        "imgs_2_cutouts",
        "imgs_3_cutouts",
        "imgs_4_cutouts",
        "imgs_5_cutouts",
    ]
    input_file_path = os.path.join(os.getcwd(), "..", "dataset", "UnityEyes_Windows")
    train_loader, valid_loader, test_loader = get_dataloader(
        input_file_path, input_filename_list, 128
    )

    # Train settings
    warp = WarpImageWithFlowAndBrightness(next(iter(train_loader))[0])
    criterion = nn.MSELoss()
    num_epochs = 500
    weight_correction_loss = 0.8
    weight_reconstruction_loss = 0.2

    if not args.eval_only:
        train(
            eccmodel,
            eccoptimizer,
            eccscheduler,
            train_loader,
            valid_loader,
            warp,
            criterion,
            start_epoch=start,
            num_epochs=num_epochs,
            weight_correction_loss=weight_correction_loss,
            weight_reconstruction_loss=weight_reconstruction_loss,
        )
    test(
        eccmodel,
        test_loader,
        warp,
        criterion,
        weight_correction_loss,
        weight_reconstruction_loss,
    )
