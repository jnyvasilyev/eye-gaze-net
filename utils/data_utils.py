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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
import mediapipe as mp
import time
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.vcam_utils import get_eye_patch

# Profiling
from torch.profiler import profile, record_function, ProfilerActivity

DEGREES_TO_RADIANS = np.pi / 180


def plt_show_image(im1, im2):
    im1 = im1.permute(1, 2, 0).detach().cpu().numpy()
    im2 = im2.permute(1, 2, 0).detach().cpu().numpy()

    full_im = np.concatenate((im1, im2), axis=1)

    plt.imshow(full_im)
    plt.show()


class UnityDataset(Dataset):
    def __init__(self, pair_list, augment=True) -> None:
        super().__init__()
        self.pairs = pair_list
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fn_in, fn_out = self.pairs[idx]

        X_img, X_angle = get_img_vec(fn_in)
        y_img, y_angle = get_img_vec(fn_out)

        # Augment the data
        if self.augment:
            X_img, X_angle, y_img, y_angle = augment_img(X_img, X_angle, y_img, y_angle)

        # plt_show_image(X_img, y_img)

        return X_img, X_angle, y_img, y_angle


def augment_img(X_img, X_angle, y_img, y_angle):
    """
    Augment the given images before training.
    Augmentation includes additive noise, color jitter, and Gaussian blur
    Augmentations are applied in random order and magnitude
    """
    ADD_NOISE_STD = [0.0, 0.05, 0.075]
    BLUR_KERNEL_SIZES = [1, 3, 5]

    # Additive noise
    noise = torch.randn_like(X_img) * random.choice(ADD_NOISE_STD)

    # Gaussian blur
    blur_kernel = random.choice(BLUR_KERNEL_SIZES)

    # Color jitter
    brightness = 1 + random.uniform(-0.5, 0.5)
    contrast = 1 + random.uniform(-0.5, 0.5)
    saturation = 1 + random.uniform(-0.5, 0.5)
    hue = random.uniform(-0.05, 0.05)

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

    # # Rotation in degrees, counter-clockwise
    # rot_angle = random.gauss(0, 7)
    # rot_angle = max(min(rot_angle, 15), -15)  # Range [-15, 15]
    # X_img = tvf.rotate(X_img, angle=rot_angle)
    # y_img = tvf.rotate(y_img, angle=rot_angle)
    # theta = torch.tensor([rot_angle * DEGREES_TO_RADIANS])
    #
    # print(rot_angle)
    #
    # def rotate_angle(X, theta):
    #     """rotate a 2x32x64 look vec"""
    #     new_x = X[0, :, :] * torch.cos(theta) - X[1, :, :] * torch.sin(theta)
    #     new_y = X[0, :, :] * torch.sin(theta) + X[1, :, :] * torch.cos(theta)
    #     return torch.stack((new_x, new_y))
    #
    # X_angle = rotate_angle(X_angle, theta)
    # y_angle = rotate_angle(y_angle, theta)

    return X_img, X_angle, y_img, y_angle


def pitchyaw_to_vector(angles, is_degrees=True):
    r"""Convert given gaze pitch and yaw to vector.
    Args:
        angles (:obj:`numpy.array`): gaze pitch (column 0) and yaw (column 1) :math:`(n\times 2)`.
        is_degrees (bool): specifies whether pitch and yaw are given in degrees. If False, angles are given in radians.
    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` specifying (X, Y, Z) vector where positive X direction is to right of head, positive Y is straight up from head, and positive Z is going into the face.
    """
    n = angles.shape[0]
    out = np.empty((n, 3))
    if is_degrees:
        angles = angles * DEGREES_TO_RADIANS
    out[:, 0] = np.sin(angles[:, 1])  # Yaw
    out[:, 1] = np.sin(
        angles[:, 0]
    )  # Pitch, assuming pitch is up/down and should be applied to Y
    out[:, 2] = np.sqrt(1 - out[:, 0] * out[:, 0] - out[:, 1] * out[:, 1])
    out[:, 2] = -out[:, 2]  # Negate z direction
    # out = np.apply_along_axis(lambda vec: vec / np.linalg.norm(vec), 1, out)
    return out


def get_img_vec(filename):
    """
    Read the jpg and json of the given filename.
    Preprocess and image and return the cropped img and angle vector
    """
    # print(f"Reading {filename}")

    img = torch.load(filename + "_img.pt")
    vec = torch.load(filename + "_vec.pt")

    # print(f"Done reading {filename}")

    return img, vec


def get_filename_info(filename):
    info_dict = {}

    # Get ID data
    titles_types = {"ID": int, "T": str, "N": int, "F": int, "V": float, "H": float}
    info_list = os.path.basename(filename[:-5]).split("_")
    for title, info in zip(titles_types, info_list):
        info_dict[title] = titles_types[title](info[len(title) :])
    info_dict["target"] = info_dict["F"] == 1
    info_dict["filename"] = os.path.basename(filename[:-5])

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


def get_filename_info_columbia(filename):
    info_dict = {}

    # Get ID data
    titles_types = {"folder": int, "m": str, "P": int, "V": float, "H": float}
    info_list = os.path.basename(filename[:-4]).split("_")
    for title, info in zip(titles_types, info_list):
        if title == "folder":
            info_dict[title] = titles_types[title](info)
        elif title == "P":
            info_dict["ID"] = titles_types[title](info[: -len(title)])
        else:
            info_dict[title] = titles_types[title](info[: -len(title)])
    info_dict["target"] = info_dict["V"] == 0 and info_dict["H"] == 0
    info_dict["filename"] = os.path.basename(filename[:-4])

    # Get image data
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    # img = img.astype(np.float32) / 255.0
    info_dict["img"] = img

    pitchyaw = np.array([[info_dict["V"], info_dict["H"]]])
    info_dict["look_vec"] = np.squeeze(pitchyaw_to_vector(pitchyaw))

    return info_dict


# ### Notes on the dataframe info
# - look_vec is a 3D homogeneous vector in the form \[x, y, z, 0\]. For
# purposes of the model input, only the x and y components are needed. I
# believe this vector is already normalized, but we can renormalize this vector
# before extracting the x and y components
# - Feature landmarks (i.e., interior_margin, caruncle, and iris) are the pixel
# coordinates of feature BEFORE RESIZING. Ideally they are used to determine
# the crop/resize area
#

# ### Dataloader
# The above dataframe contains info on all images, separated into IDs. As our
# model input, we would like to take all possible image pairs within an ID.
# With 40 images per ID, this results in 780 pairs.
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
    landmarks = np.concatenate(
        (info_dict["interior_margin"], info_dict["caruncle"]), axis=0
    )
    min_x = np.min(landmarks, axis=0)[0]
    min_y = np.min(landmarks, axis=0)[1]
    max_x = np.max(landmarks, axis=0)[0]
    max_y = np.max(landmarks, axis=0)[1]

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
    img_tensor_right = torch.flip(img_tensor, [2])

    # Preprocess angle vector
    vec = info_dict["look_vec"]
    vec = (vec / np.linalg.norm(vec, keepdims=True))[
        :2
    ]  # normalize, then get x and y components
    vec_right = vec.copy()
    vec_right[0] = -vec_right[0]
    vec = np.tile(vec[:, np.newaxis, np.newaxis], (1, 32, 64))
    vec_right = np.tile(vec_right[:, np.newaxis, np.newaxis], (1, 32, 64))
    vec_tensor = torch.tensor(vec)
    vec_tensor_right = torch.tensor(vec_right)

    # Save the image and vector
    # TODO: make these one file
    os.makedirs(output_file_path, exist_ok=True)
    filename = info_dict["filename"]
    torch.save(
        img_tensor,
        os.path.join(output_file_path, filename + "_img.pt"),
    )
    torch.save(
        vec_tensor,
        os.path.join(output_file_path, filename + "_vec.pt"),
    )

    # Save right eye image and vector
    os.makedirs(output_file_path + "_right", exist_ok=True)
    torch.save(
        img_tensor_right,
        os.path.join(output_file_path + "_right", filename + "_img.pt"),
    )
    torch.save(
        vec_tensor_right,
        os.path.join(output_file_path + "_right", filename + "_vec.pt"),
    )


def process_image_columbia(info_dict, output_file_path):
    """
    Read the jpg and json of the given filename.
    Preprocess and image and return the cropped img and angle vector
    """
    # Get image data
    img = info_dict["img"]

    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True
    ) as face_mesh:
        results = face_mesh.process(img)
        for face_landmarks in results.multi_face_landmarks:
            face = face_landmarks.landmark
            # draw(face, image)

            # Apply ECCNet to image
            with torch.no_grad():
                for left in [True, False]:
                    # Get eye image patch
                    og_eye_patch, og_size, cut_coord = get_eye_patch(face, img, left)
                    og_eye_patch = og_eye_patch.astype(np.float32) / 255.0
                    og_eye_patch = np.transpose(og_eye_patch, (2, 0, 1))
                    img_tensor = torch.tensor(og_eye_patch).float()

                    # Preprocess angle vector
                    vec = info_dict["look_vec"]
                    vec = (vec / np.linalg.norm(vec, keepdims=True))[
                        :2
                    ]  # normalize, then get x and y components
                    vec = np.tile(vec[:, np.newaxis, np.newaxis], (1, 32, 64))
                    vec_tensor = torch.tensor(vec)

                    # Save the image and vector
                    # TODO: make these one file
                    filename = info_dict["filename"]
                    if left:
                        os.makedirs(os.path.join(output_file_path), exist_ok=True)
                        torch.save(
                            img_tensor,
                            os.path.join(output_file_path, filename + "_img.pt"),
                        )
                        torch.save(
                            vec_tensor,
                            os.path.join(output_file_path, filename + "_vec.pt"),
                        )
                    else:
                        os.makedirs(output_file_path + "_right", exist_ok=True)
                        torch.save(
                            img_tensor,
                            os.path.join(
                                output_file_path + "_right", filename + "_img.pt"
                            ),
                        )
                        torch.save(
                            vec_tensor,
                            os.path.join(
                                output_file_path + "_right", filename + "_vec.pt"
                            ),
                        )


# Read dataset folder
def process_dataset(input_file_path, input_fn, output_file_path, dtype="UnityEyes"):
    """
    Read the image and json data in the specified folder.
    Args:
        input_file_path: the base directory containing the dataset directories
        input_fn: the name of dataset directory
        output_file_path: the directory in which process tensors are stored
    """
    img_infos = []
    if dtype == "UnityEyes":
        json_fns = glob(os.path.join(input_file_path, input_fn, "*.json"))
        for json_fn in json_fns:
            info = get_filename_info(json_fn)
            process_image(info, os.path.join(input_file_path, input_fn + "_cutouts"))
            img_infos.append(info)
    elif dtype == "Columbia":
        jpg_fns = glob(os.path.join(input_file_path, input_fn, "*.jpg"))
        for jpg_fn in jpg_fns:
            info = get_filename_info_columbia(jpg_fn)
            process_image_columbia(
                info, os.path.join(input_file_path, input_fn + "_cutouts")
            )
            img_infos.append(info)

    else:
        raise ValueError(f"Unrecognized dataset type {dtype}")

    img_df = pd.DataFrame(img_infos)
    img_df.sort_values(["ID"], ignore_index=True, inplace=True)

    # Extract relevant data for dataloader
    # n_ids = img_df.iloc[-1]["ID"]

    # For each ID, generate all possible pairs
    # for id in tqdm(range(1, n_ids + 1)):
    for id in tqdm(img_df["ID"].unique().tolist()):
        # Get ID
        df_chunk = img_df.query(f"ID == {id}")
        fns = np.stack(df_chunk["filename"])

        # Generate pairs
        pairs = np.triu_indices(len(df_chunk), k=1)
        pairs = np.stack(pairs).transpose()

        fn_pairs = fns[pairs]

        # Store the filenames for every possible pair
        # This is what the dataloader will read
        os.makedirs(output_file_path, exist_ok=True)
        os.makedirs(output_file_path + "_right", exist_ok=True)
        for pair_idx in range(len(fn_pairs)):
            # Left eye pairs
            with open(
                os.path.join(output_file_path, f"{input_fn}_{id}_p{pair_idx}.pkl"),
                "wb",
            ) as f:
                pickle.dump(
                    (
                        os.path.join(
                            input_file_path,
                            input_fn + "_cutouts",
                            fn_pairs[pair_idx][0],
                        ),
                        os.path.join(
                            input_file_path,
                            input_fn + "_cutouts",
                            fn_pairs[pair_idx][1],
                        ),
                    ),
                    f,
                )

            # Right eye pairs
            with open(
                os.path.join(
                    output_file_path + "_right", f"{input_fn}_{id}_p{pair_idx}.pkl"
                ),
                "wb",
            ) as f:
                pickle.dump(
                    (
                        os.path.join(
                            input_file_path,
                            input_fn + "_cutouts_right",
                            fn_pairs[pair_idx][0],
                        ),
                        os.path.join(
                            input_file_path,
                            input_fn + "_cutouts_right",
                            fn_pairs[pair_idx][1],
                        ),
                    ),
                    f,
                )


def create_pairs(input_file_path, input_filename_list, left):
    train_paths = [
        os.path.join(input_file_path, input_fn + "_pairs")
        for input_fn in input_filename_list
    ]

    train_pairs = []

    for root in train_paths:
        for file in os.listdir(root):
            with open(os.path.join(root, file), "rb") as f:
                fn_in, fn_out = pickle.load(f)
                train_pairs += [(fn_in, fn_out)]

    with open(
        os.path.join(input_file_path, "train_pairs.pkl"),
        "wb",
    ) as f:
        pickle.dump(
            train_pairs,
            f,
        )

    train_paths_right = [
        os.path.join(input_file_path, input_fn + "_pairs_right")
        for input_fn in input_filename_list
    ]

    train_pairs_right = []

    for root in train_paths_right:
        for file in os.listdir(root):
            with open(os.path.join(root, file), "rb") as f:
                fn_in, fn_out = pickle.load(f)
                train_pairs_right += [(fn_in, fn_out)]

    with open(
        os.path.join(input_file_path, "train_pairs_right.pkl"),
        "wb",
    ) as f:
        pickle.dump(
            train_pairs_right,
            f,
        )

    return train_pairs if left else train_pairs_right


def get_dataloader(
    input_file_path,
    input_filename_list,
    # output_file_path,
    batch_size=512,
    num_workers=16,
    dtype="UnityEyes",
    left=True,
):
    """
    Process and return a dataloader for the UnityEyes dataset.
    """
    if left:
        pairs_path = os.path.join(input_file_path, f"train_pairs.pkl")
    else:
        pairs_path = os.path.join(input_file_path, f"train_pairs_right.pkl")
    if os.path.exists(pairs_path):
        print(f"Preprocessed {dtype} dataset found. Loading...")
        with open(pairs_path, "rb") as f:
            train_pairs = pickle.load(f)
    else:
        print(f"Preprocessed {dtype} tensors not available. Reading dataset")
        for input_fn in tqdm(input_filename_list):
            output_file_path = os.path.join(input_file_path, input_fn + "_pairs")
            print("Reading " + input_fn)

            process_dataset(input_file_path, input_fn, output_file_path, dtype=dtype)

            print("Creating pairs file")
            train_pairs = create_pairs(input_file_path, input_filename_list, left)

    augment = False
    if dtype == "UnityEyes":
        augment = True

    train_dataset = UnityDataset(train_pairs, augment)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    print("Dataset loaded")

    return train_loader


if __name__ == "__main__":
    import data.unityeyes
    import data.columbia

    # Preprocess the dataset
    dataset_dir = os.path.join(os.getcwd(), "..", "dataset")
    train_filename_list = data.unityeyes.filename_list
    train_file_path = os.path.join(dataset_dir, data.unityeyes.dir_name)
    valid_filename_list = data.columbia.filename_list
    valid_file_path = os.path.join(dataset_dir, data.columbia.dir_name)

    train_loader = get_dataloader(
        train_file_path,
        train_filename_list,
        batch_size=1,
        num_workers=1,
        dtype=data.unityeyes.name,
        # left=False,
    )
    valid_loader = get_dataloader(
        valid_file_path,
        valid_filename_list,
        batch_size=1,
        num_workers=1,
        dtype=data.columbia.name,
        # left=False,
    )

    print(len(train_loader))
    print(len(valid_loader))

    # exit()

    # Test the output
    device = "cuda"

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        with record_function("load batch"):
            for imgs, angles, targets, target_angles in tqdm(valid_loader):
                imgs, angles, targets, target_angles = (
                    imgs.float().to(device),
                    angles.float().to(device),
                    targets.float().to(device),
                    target_angles.float().to(device),
                )
                plt_show_image(imgs[0], targets[0])

                # break
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # print(imgs.shape)
    # print(imgs.dtype)
    # # plt.imshow(imgs[0].permute(1, 2, 0).detach().cpu().numpy())
    # print(angles.shape)
    # print(angles.dtype)
    # print(angles[0, 0])
    # print(angles[0, 1])
    # print(targets.shape)
    # print(targets.dtype)
    # # plt.imshow(targets[0].permute(1, 2, 0).detach().cpu().numpy())
    # print(target_angles.shape)
    # print(target_angles.dtype)
    # print(target_angles[0, 0])
    # print(target_angles[0, 1])

    # exit()
