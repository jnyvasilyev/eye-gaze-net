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
from utils.loss_utils import (
    misalignment_tolerant_mse_loss,
    misalignment_tolerant_ssim_loss,
)
from utils.data_utils import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "./output"


def get_model_optimizer(ckpt_iter):
    # Create the model
    model = ECCNet()
    model.to(device)
    # model = nn.DataParallel(model) # Comment out if using only one GPU
    optimizer = optim.Adam(model.parameters(), lr=0.002, eps=0.1)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=0.002, max_lr=0.01, cycle_momentum=False
    )
    start = 0
    epochs = []
    train_losses = []
    valid_losses = []

    # Load checkpoint
    if ckpt_iter:
        ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoints/ckpt_{ckpt_iter}.pt")
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start = checkpoint["epoch"] + 1
        epochs = checkpoint["epochs"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

    return model, optimizer, scheduler, start, epochs, train_losses, valid_losses


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
    mse_weight=0.8,
    ssim_weight=0.2,
    epochs=[],
    train_losses=[],
    valid_losses=[],
):
    print("Beginning training")
    printout_freq = 1
    save_ckpt_freq = 1

    train_checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(train_checkpoint_path, exist_ok=True)

    images_directory_path = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(images_directory_path, exist_ok=True)

    progress_plot_path = os.path.join(OUTPUT_DIR, "progress")
    os.makedirs(progress_plot_path, exist_ok=True)
    start_time = time.time()

    pbar = tqdm(range(start_epoch, start_epoch + num_epochs))

    # Placeholders for displaying last batch of images
    img_corr_display = None
    target_display = None

    for epoch in pbar:
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        for imgs, angles, targets, target_angles in tqdm(train_loader):
            imgs, angles, targets, target_angles = (
                imgs.float().to(device),
                angles.float().to(device),
                targets.float().to(device),
                target_angles.float().to(device),
            )

            flow_corr, bright_corr = model(imgs, target_angles)
            img_corr = warp(imgs, flow_corr, bright_corr)

            # print(img_corr.shape)
            # print(targets.shape)
            # plt.imshow(img_corr[0].permute(1, 2, 0).detach().cpu().numpy())
            # plt.show()
            # plt.imshow(targets[0].permute(1, 2, 0).detach().cpu().numpy())
            # plt.show()
            #
            loss_correction_mse = misalignment_tolerant_mse_loss(
                img_corr, targets, criterion
            )
            loss_correction_ssim = misalignment_tolerant_ssim_loss(img_corr, targets)

            flow_reconstruction, bright_reconstruction = model(img_corr, angles)
            img_reconstruction = warp(
                img_corr, flow_reconstruction, bright_reconstruction
            )

            loss_reconstruction_mse = misalignment_tolerant_mse_loss(
                img_reconstruction, imgs, criterion
            )
            loss_reconstruction_ssim = misalignment_tolerant_ssim_loss(
                img_reconstruction, imgs
            )

            mse_loss = (weight_correction_loss * loss_correction_mse) + (
                weight_reconstruction_loss * loss_reconstruction_mse
            )

            ssim_loss = (weight_correction_loss * loss_correction_ssim) + (
                weight_reconstruction_loss * loss_reconstruction_ssim
            )

            loss = mse_loss * mse_weight + ssim_loss * ssim_weight

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

                loss_correction_mse = misalignment_tolerant_mse_loss(
                    img_corr, targets, criterion
                )
                loss_correction_ssim = misalignment_tolerant_ssim_loss(
                    img_corr, targets
                )

                flow_reconstruction, bright_reconstruction = model(img_corr, angles)
                img_reconstruction = warp(
                    img_corr, flow_reconstruction, bright_reconstruction
                )

                loss_reconstruction_mse = misalignment_tolerant_mse_loss(
                    img_reconstruction, imgs, criterion
                )
                loss_reconstruction_ssim = misalignment_tolerant_ssim_loss(
                    img_reconstruction, imgs
                )

                mse_loss = (weight_correction_loss * loss_correction_mse) + (
                    weight_reconstruction_loss * loss_reconstruction_mse
                )

                ssim_loss = (weight_correction_loss * loss_correction_ssim) + (
                    weight_reconstruction_loss * loss_reconstruction_ssim
                )

                loss = mse_loss * mse_weight + ssim_loss * ssim_weight

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
                        "epochs": epochs,
                        "train_losses": train_losses,
                        "validation_losses": valid_losses,
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
    mse_weight=0.8,
    ssim_weight=0.2,
):
    print("Evaluating model")

    images_directory_path = os.path.join(OUTPUT_DIR, "test_images")
    os.makedirs(images_directory_path, exist_ok=True)

    # Misc tensors to save for debugging
    misc_path = os.path.join(OUTPUT_DIR, "misc")
    os.makedirs(misc_path, exist_ok=True)
    imgs_copy, flow_copy, brightness_copy, targets_copy = None, None, None, None

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
            loss_correction_mse = misalignment_tolerant_mse_loss(
                img_corr, targets, criterion
            )
            loss_correction_ssim = misalignment_tolerant_ssim_loss(img_corr, targets)

            flow_reconstruction, bright_reconstruction = model(img_corr, angles)
            img_reconstruction = warp(
                img_corr, flow_reconstruction, bright_reconstruction
            )

            loss_reconstruction_mse = misalignment_tolerant_mse_loss(
                img_reconstruction, imgs, criterion
            )
            loss_reconstruction_ssim = misalignment_tolerant_ssim_loss(
                img_reconstruction, imgs
            )

            mse_loss = (weight_correction_loss * loss_correction_mse) + (
                weight_reconstruction_loss * loss_reconstruction_mse
            )

            ssim_loss = (weight_correction_loss * loss_correction_ssim) + (
                weight_reconstruction_loss * loss_reconstruction_ssim
            )

            loss = mse_loss * mse_weight + ssim_loss * ssim_weight

            test_loss += loss.item()

            img_display = imgs.clone()

            imgs_copy = imgs.clone()
            flow_copy = flow_corr.clone()
            brightness_copy = bright_corr.clone()
            targets_copy = targets.clone()

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

        # Save misc tensors
        torch.save(imgs_copy, os.path.join(misc_path, "imgs.pt"))
        torch.save(flow_copy, os.path.join(misc_path, "flow.pt"))
        torch.save(brightness_copy, os.path.join(misc_path, "brightness.pt"))
        torch.save(targets_copy, os.path.join(misc_path, "targets.pt"))

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true", default=False)

    args = parser.parse_args()

    print(f"Using device {device}")
    (
        eccmodel,
        eccoptimizer,
        eccscheduler,
        start,
        epochs,
        train_losses,
        valid_losses,
    ) = get_model_optimizer(args.checkpoint)
    input_filename_list = [
        "imgs_1_cutouts",
        "imgs_2_cutouts",
        "imgs_3_cutouts",
        "imgs_4_cutouts",
        "imgs_5_cutouts",
        "imgs_6_cutouts",
        "imgs_7_cutouts",
        "imgs_8_cutouts",
        "imgs_9_cutouts",
        "imgs_10_cutouts",
        "imgs_11_cutouts",
        "imgs_12_cutouts",
        "imgs_13_cutouts",
        "imgs_14_cutouts",
        "imgs_15_cutouts",
        "imgs_16_cutouts",
        "imgs_17_cutouts",
        "imgs_18_cutouts",
        "imgs_19_cutouts",
        "imgs_20_cutouts",
    ]
    input_file_path = os.path.join(os.getcwd(), "..", "dataset", "UnityEyes_Windows")
    dataset_file_path = "./dataset"
    train_loader, valid_loader = get_dataloader(
        input_file_path, input_filename_list, dataset_file_path, 512, 8
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
            mse_weight=1.0,
            ssim_weight=0.0,
            epochs=epochs,
            train_losses=train_losses,
            valid_losses=valid_losses,
        )
    # TODO: for now, reusing validation set as test set. Test set should be a natural dataset
    test(
        eccmodel,
        valid_loader,
        warp,
        criterion,
        weight_correction_loss,
        weight_reconstruction_loss,
    )
