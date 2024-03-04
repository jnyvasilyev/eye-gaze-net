import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as V
import torchvision
import matplotlib.pyplot as plt
import numpy as np


import torch
import torch.nn.functional as F


class WarpImageWithFlowAndBrightness:
    def __init__(self, images):
        """
        Initialize the WarpImage class.

        Args:
            images (torch.Tensor): Input images of shape [N, C, H, W].
        """
        self.H, self.W = images.shape[2], images.shape[3]
        self.grid_x, self.grid_y = torch.meshgrid(torch.arange(self.W), torch.arange(self.H))
        self.grid_x = self.grid_x.to(torch.float32)
        self.grid_y = self.grid_y.to(torch.float32)

    def __call__(self, images, flow_map, brightness_map):
        """
        Warps input images based on flow and brightness maps.

        Args:
            images (torch.Tensor): Input images of shape [N, C, H, W].
            flow_map (torch.Tensor): Flow map of shape [N, 2, H, W].
            brightness_map (torch.Tensor): Brightness map of shape [N, 1, H, W].

        Returns:
            torch.Tensor: Warped images.
        """
        # Apply flow map to grid coordinates
        flow_x = self.grid_x + flow_map[:, 0]
        flow_y = self.grid_y + flow_map[:, 1]

        # Normalize flow coordinates to [-1, 1]
        flow_x_normalized = (2 * flow_x / (self.W - 1)) - 1
        flow_y_normalized = (2 * flow_y / (self.H - 1)) - 1

        # Sample pixels from input images using bilinear interpolation
        warped_images = F.grid_sample(images, torch.stack((flow_x_normalized, flow_y_normalized), dim=-1), mode='bilinear', padding_mode='border')

        # Apply brightness correction
        warped_images *= brightness_map

        warped_images = V.adjust_sharpness(warped_images, 3)

        return warped_images

def display_images(original_images, warped_images):
    """
    Display original and warped images side by side.

    Args:
        original_images (torch.Tensor): Original input images.
        warped_images (torch.Tensor): Warped images.

    Returns:
        None
    """
    N = original_images.shape[0]  # Number of images
    fig, axs = plt.subplots(N, 2, figsize=(10, 5 * N))

    for i in range(N):
        axs[i, 0].imshow(original_images[i].permute(1, 2, 0))
        axs[i, 0].set_title(f"Original Image {i + 1}")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(warped_images[i].permute(1, 2, 0))
        axs[i, 1].set_title(f"Warped Image {i + 1}")
        axs[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


# Use to make sure batch of images is in proper input format or warped correctly
def save_image(tensor, filename="image_grid.png"):
    """
    Save a tensor as an image grid to a PNG file.

    Parameters:
    - tensor: A PyTorch tensor of shape [N, C, H, W]
    - filename: String, the filename to save the image as
    """
    # Check if tensor needs to be detached and moved to CPU
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()

    tensor = tensor / 255.0

    # Make grid (2 rows and 5 columns) to display image batch
    grid_img = torchvision.utils.make_grid(tensor, nrow=5)

    # Convert the tensor to a numpy array and change data layout from C, H, W to H, W, C for displaying
    np_grid_img = grid_img.permute(1, 2, 0).numpy()

    # Display the image grid
    plt.imshow(np_grid_img)
    plt.axis("off")  # Turn off axis numbers and ticks

    # Save the image grid to a file
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.0)
    plt.close()  # Close the plot to prevent it from displaying in notebooks or environments


# Example usage:
# save_image(your_tensor, "your_filename.png")

"""
warp = WarpImageWithFlowAndBrightness(initial_images)

for epoch in range(num_epochs):
    for images, labels in data_loader:
        # Generate flow_map and brightness_map from the model
        flow_map, brightness_map = model(images)
        
        # Warp and adjust the images using the pre-initialized class instance
        adjusted_images = warp(images, flow_map, brightness_map)
        
        # Calculate the loss between adjusted_images and the ground truth labels
        loss = loss_function(adjusted_images, labels)
        
        # Backpropagation, optimizer steps, etc.

"""
