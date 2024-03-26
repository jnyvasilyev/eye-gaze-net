import os
import shutil
import json
import pickle
import random
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import time
import cv2
import matplotlib.pyplot as plt
from ipdb import set_trace
from tqdm import tqdm

from warp import plt_show_image


class UnityDataset(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = root
        self.files = os.listdir(root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # imgs, vecs = torch.load(os.path.join(self.root, self.files[idx]))
        with open(os.path.join(self.root, self.files[idx]), "rb") as f:
            fn_in, fn_out = pickle.load(f)
        X_img, X_angle = get_img_vec(fn_in)
        y_img, y_angle = get_img_vec(fn_out)

        # Augment the data
        X_img, y_img = augment_img(X_img, y_img)

        # plt_show_image(X_img, y_img, X_img)

        return X_img, X_angle, y_img, y_angle


def augment_img(X_img, y_img):
    """
    Augment the given images before training.
    Augmentation includes additive noise, color jitter, and Gaussian blur
    Augmentations are applied in random order and magnitude
    """
    ADD_NOISE_STD = 0.1
    BLUR_KERNEL_SIZES = [3, 5, 7]

    # Additive noise
    noise = torch.randn_like(X_img) * ADD_NOISE_STD

    # Gaussian blur
    blur_kernel = random.choice(BLUR_KERNEL_SIZES)

    # Color jitter
    brightness = 1 + random.uniform(-0.2, 0.2)
    contrast = 1 + random.uniform(-0.2, 0.2)
    saturation = 1 + random.uniform(-0.2, 0.2)
    hue = random.uniform(-0.1, 0.1)

    augmentation_functions = [
        lambda x: torch.clamp(x + noise, 0, 1),
        lambda x: tvf.gaussian_blur(x, kernel_size=blur_kernel),
        lambda x: tvf.adjust_brightness(x, brightness),
        lambda x: tvf.adjust_contrast(x, contrast),
        lambda x: tvf.adjust_saturation(x, saturation),
        lambda x: tvf.adjust_hue(x, hue),
    ]

    random.shuffle(augmentation_functions)

    # Apply augmentation techniques with random magnitudes
    for augmentation_func in augmentation_functions:
        X_img = augmentation_func(X_img)
        y_img = augmentation_func(y_img)

    return X_img, y_img


def get_img_vec(filename):
    """
    Read the jpg and json of the given filename.
    Preprocess and image and return the cropped img and angle vector
    """
    filename = filename[:-5]

    img = torch.load(filename + "_img.pt")
    vec = torch.load(filename + "_vec.pt")

    return img, vec


def get_filename_info(filename):
    info_dict = {}

    # Get ID data
    titles_types = {"ID": int, "T": str, "N": int, "F": int, "V": float, "H": float}
    info_list = os.path.basename(filename[:-5]).split("_")
    for title, info in zip(titles_types, info_list):
        info_dict[title] = titles_types[title](info[len(title) :])
    info_dict["target"] = info_dict["F"] == 1
    info_dict["filename"] = filename

    # Get image data
    img = cv2.cvtColor(cv2.imread("%s.jpg" % filename[:-5]), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    info_dict["img"] = img

    # Get json data
    def process_json_list(json_list):
        ldmks = [eval(s) for s in json_list]
        return np.array([(x, img.shape[0] - y, z) for (x, y, z) in ldmks])

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

    return info_dict


# ### Notes on the dataframe info
# - look_vec is a 3D homogeneous vector in the form \[x, y, z, 0\]. For purposes of the model input, only the x and y components are needed. I believe this vector is already normalized, but we can renormalize this vector before extracting the x and y components
# - Feature landmarks (i.e., interior_margin, caruncle, and iris) are the pixel coordinates of feature BEFORE RESIZING. Ideally they are used to determine the crop/resize area
#

# ### Dataloader
# The above dataframe contains info on all images, separated into IDs. As our model input, we would like to take all possible image pairs within an ID. With 40 images per ID, this results in 780 pairs.
#
# Dataloader will output input image, input vector, target image, target vector


def process_image(info_dict, output_file_path):
    """
    Read the jpg and json of the given filename.
    Preprocess and image and return the cropped img and angle vector
    """
    # Get image data
    img = info_dict["img"]

    # Crop the eye
    # Calculate bounding box
    min_x = np.min(info_dict["interior_margin"], axis=0)[0]
    min_y = np.min(info_dict["interior_margin"], axis=0)[1]
    max_x = np.max(info_dict["interior_margin"], axis=0)[0]
    max_y = np.max(info_dict["interior_margin"], axis=0)[1]

    # Get bbox length
    l = max_x - min_x
    aspect_ratio = 2
    width = 1.5 * l
    height = max_y - min_y

    # Calculate center of bbox
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Correct aspect ratio
    if (width) / (height) > aspect_ratio:
        # too wide. Expand height
        height = width / aspect_ratio
    else:
        # too tall. Expand width
        width = height * aspect_ratio

    # Resize to 64x32
    img_cropped = img[
        int(center_y - (height / 2)) : int(center_y + (height / 2)),
        int(center_x - (width / 2)) : int(center_x + (width / 2)),
    ]

    img = cv2.resize(img_cropped, (64, 32))
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.tensor(img)

    # Preprocess angle vector
    vec = info_dict["look_vec"]
    vec = (vec / np.linalg.norm(vec, keepdims=True))[
        :2
    ]  # normalize, then get x and y components
    vec = np.tile(vec[:, np.newaxis, np.newaxis], (1, 32, 64))
    vec_tensor = torch.tensor(vec)

    # Save the image and vector
    os.makedirs(os.path.join(output_file_path), exist_ok=True)
    filename = os.path.basename(info_dict["filename"][:-5])
    torch.save(
        img_tensor,
        os.path.join(output_file_path, filename + "_img.pt"),
    )
    torch.save(
        vec_tensor,
        os.path.join(output_file_path, filename + "_vec.pt"),
    )

    return img, vec


# Read dataset folder
def process_dataset(input_file_path, input_fn, output_file_path):
    """
    Read the image and json data in the specified folder.
    Args:
        input_file_path: the base directory containing the dataset directories
        input_fn: the name of dataset directory
        output_file_path: the directory in which process tensors are stored
    """
    img_infos = []
    json_fns = glob(os.path.join(input_file_path, input_fn, "*.json"))
    for json_fn in json_fns:
        info = get_filename_info(json_fn)
        process_image(info, os.path.join(input_file_path, input_fn + "_cutouts"))
        img_infos.append(info)

    img_df = pd.DataFrame(img_infos)
    img_df.sort_values(["ID", "F"], ignore_index=True, inplace=True)

    # Extract relevant data for dataloader
    n_ids = img_df.iloc[-1]["ID"]

    # For each ID, generate all possible pairs
    for id in tqdm(range(1, n_ids + 1)):
        # Get ID
        df_chunk = img_df.query(f"ID == {id}")
        fns = np.stack(df_chunk["filename"])

        # Generate pairs
        pairs = np.triu_indices(len(df_chunk), k=1)
        pairs = np.stack(pairs).transpose()

        fn_pairs = fns[pairs]

        # Store the filenames for every possible pair
        # This is what the dataloader will read
        # os.makedirs(os.path.join(output_file_path, input_fn), exist_ok=True)
        for pair_idx in range(len(fn_pairs)):
            with open(
                os.path.join(output_file_path, f"{input_fn}_{id}_p{pair_idx}.pkl"),
                "wb",
            ) as f:
                pickle.dump((fn_pairs[pair_idx][0], fn_pairs[pair_idx][1]), f)


def get_dataloader(
    input_file_path,
    input_filename_list,
    output_file_path,
    batch_size=512,
    num_workers=8,
):
    """
    Process and return a dataloader for the UnityEyes dataset.
    """
    train_path = os.path.join(output_file_path, "train")
    valid_path = os.path.join(output_file_path, "valid")

    if os.path.exists(output_file_path):
        print("Preprocessed dataset found. Loading...")
    else:
        print("Preprocessed tensors not available. Reading dataset")
        os.makedirs(output_file_path, exist_ok=True)

        for input_fn in tqdm(input_filename_list):
            print("Reading " + input_fn)
            process_dataset(input_file_path, input_fn, output_file_path)

        # Create splits
        files = os.listdir(output_file_path)
        split_ratio = 0.8
        split_idx = int(len(files) * split_ratio)

        train_files = files[:split_idx]
        valid_files = files[split_idx:]

        # Move splits into new directories
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(valid_path, exist_ok=True)

        for file in train_files:
            src = os.path.join(output_file_path, file)
            dst = os.path.join(train_path, file)
            shutil.move(src, dst)

        for file in valid_files:
            src = os.path.join(output_file_path, file)
            dst = os.path.join(valid_path, file)
            shutil.move(src, dst)

    train_dataset = UnityDataset(train_path)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_dataset = UnityDataset(valid_path)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, valid_loader


if __name__ == "__main__":
    # Preprocess the dataset
    input_filename_list = [
        "imgs_1",
        "imgs_2",
        "imgs_3",
        "imgs_4",
        "imgs_5",
        "imgs_6",
        "imgs_7",
        "imgs_8",
        "imgs_9",
        "imgs_10",
        "imgs_11",
        "imgs_12",
        "imgs_13",
        "imgs_14",
        "imgs_15",
        "imgs_16",
        "imgs_17",
        "imgs_18",
        "imgs_19",
        "imgs_20",
    ]
    input_file_path = os.path.join(os.getcwd(), "..", "dataset", "UnityEyes_Windows")
    output_file_path = "./dataset"
    train_loader, valid_loader = get_dataloader(
        input_file_path, input_filename_list, output_file_path, 1
    )

    # Test the output
    device = "cuda"
    for imgs, angles, targets, target_angles in train_loader:
        imgs, angles, targets, target_angles = (
            imgs.float().to(device),
            angles.float().to(device),
            targets.float().to(device),
            target_angles.float().to(device),
        )

        print(imgs.shape)
        print(imgs.dtype)
        plt.imshow(imgs[0].permute(1, 2, 0).detach().cpu().numpy())
        print(angles.shape)
        print(angles.dtype)
        print(angles[0, 0])
        print(angles[0, 1])
        print(targets.shape)
        print(targets.dtype)
        plt.imshow(targets[0].permute(1, 2, 0).detach().cpu().numpy())
        print(target_angles.shape)
        print(target_angles.dtype)
        print(target_angles[0, 0])
        print(target_angles[0, 1])

        plt.show()

        exit()
