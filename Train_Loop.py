#!/usr/bin/env python
# coding: utf-8
import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cv2
import json
import matplotlib.pyplot as plt
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

import utils.data.unityeyes
import utils.data.columbia

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "./output3_head"

# os.environ["TORCH_BOTTLENECK"] = "1"


def get_model_optimizer(ckpt_iter, fine_tune=False):
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
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start = checkpoint["epoch"] + 1
        epochs = checkpoint["epochs"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["validation_losses"]

        print("Checkpoint loaded.")

    if fine_tune:
        # Freeze all layers after first skip connection
        # i.e. everything but conv_block1 and conv_block2
        for name, param in model.named_parameters():
            lname = name.split(".")[0]
            if not any(
                layer_name == lname for layer_name in ["conv_block1", "conv_block2"]
            ):
                param.requires_grad = False

        # Verify frozen
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

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

    torch.cuda.empty_cache()

    for epoch in pbar:
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        for imgs, angles, heads, targets, target_angles, target_heads in tqdm(
            train_loader
        ):
            imgs, angles, heads, targets, target_angles, target_heads = (
                imgs.float().to(device),
                angles.float().to(device),
                heads.float().to(device),
                targets.float().to(device),
                target_angles.float().to(device),
                target_heads.float().to(device),
            )

            flow_corr, bright_corr = model(imgs, target_angles, heads)
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

            flow_reconstruction, bright_reconstruction = model(img_corr, angles, heads)
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

            # exit()

        train_losses.append(train_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for (
                imgs,
                angles,
                heads,
                targets,
                target_angles,
                target_heads,
            ) in valid_loader:
                imgs, angles, heads, targets, target_angles, target_heads = (
                    imgs.float().to(device),
                    angles.float().to(device),
                    heads.float().to(device),
                    targets.float().to(device),
                    target_angles.float().to(device),
                    target_heads.float().to(device),
                )

                flow_corr, bright_corr = model(imgs, target_angles, heads)
                img_corr = warp(imgs, flow_corr, bright_corr)

                loss_correction_mse = misalignment_tolerant_mse_loss(
                    img_corr, targets, criterion
                )
                loss_correction_ssim = misalignment_tolerant_ssim_loss(
                    img_corr, targets
                )

                flow_reconstruction, bright_reconstruction = model(
                    img_corr, angles, heads
                )
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
                        "scheduler_state_dict": scheduler.state_dict(),
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
                # print(img_corr_display.shape)
                # print(target_display.shape)
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
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--fine_tune", action="store_true", default=False)

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
    ) = get_model_optimizer(args.checkpoint, args.fine_tune)

    dataset_dir = os.path.join(os.getcwd(), "..", "dataset")
    if args.fine_tune:
        train_filename_list = utils.data.columbia.filename_list
        train_file_path = os.path.join(dataset_dir, utils.data.columbia.dir_name)
        train_loader = get_dataloader(
            train_file_path,
            train_filename_list,
            batch_size=512,
            num_workers=8,
            dtype=utils.data.columbia.name,
        )
    else:
        train_filename_list = utils.data.unityeyes.filename_list
        train_file_path = os.path.join(dataset_dir, utils.data.unityeyes.dir_name)
        train_loader = get_dataloader(
            train_file_path,
            train_filename_list,
            batch_size=512,
            num_workers=8,
            dtype=utils.data.unityeyes.name,
        )

    valid_filename_list = utils.data.columbia.filename_list
    valid_file_path = os.path.join(dataset_dir, utils.data.columbia.dir_name)
    valid_loader = get_dataloader(
        valid_file_path,
        valid_filename_list,
        batch_size=512,
        num_workers=8,
        dtype=utils.data.columbia.name,
    )

    # Train settings
    warp = WarpImageWithFlowAndBrightness(next(iter(train_loader))[0])
    criterion = nn.MSELoss()
    num_epochs = 2000
    weight_correction_loss = 0.8
    weight_reconstruction_loss = 0.2

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
        mse_weight=0.8,
        ssim_weight=0.2,
        epochs=epochs,
        train_losses=train_losses,
        valid_losses=valid_losses,
    )
