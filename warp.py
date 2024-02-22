import torch
import torch.nn.functional as F
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import numpy as np



class WarpImageWithFlowAndBrightness:
    # Use before the training loop to save compute
    # See bottom of file for example of where to put in training loop
    def __init__(self, images):
        """
        Initialize warp class with torch grid
        
        Parameters:
        - images: A PyTorch tensor of shape [N, C, H, W]
        """
        N, C, H, W = images.size()
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        self.grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0)  # Pre-compute grid
    
    # 
    def __call__(self, images, flow_map, brightness_map):
        """
        Warp batch of images using model-outputted flow_map and brightness_map.
        
        Parameters:
        - images: A PyTorch tensor of shape [N, C, H, W]
        - flow_map: A PyTorch tensor of shape [N, H, W, 2]
            note: model.py now outputs flow_map with above shape, so no change needed
        - brightness_map: brightness map of shape [N, C, H, W]
        
        Returns:
        - adjusted_images: A PyTorch tensor of shape [N, C, H, W]
        """
        warped_grid = self.grid.to(images.device) + flow_map
        warped_images = F.grid_sample(images, warped_grid, mode='bilinear', padding_mode='border')
        adjusted_images = warped_images * brightness_map
        return adjusted_images

# Use to make sure batch of images is in proper input format or warped correctly
def display_image(tensor):
    """
    Display a tensor as an image.
    
    Parameters:
    - tensor: A PyTorch tensor of shape [N, C, H, W]
    """
    # Check if tensor needs to be detached and moved to CPU
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # make grid (2 rows and 5 columns) to display our 10 images
    grid_img = torchvision.utils.make_grid(tensor, nrow=5)

    plt.imshow(grid_img.permute(1, 2, 0))

'''
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
'''