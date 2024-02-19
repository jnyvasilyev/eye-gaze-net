import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def warp_image_with_flow_and_brightness(image, flow_map, brightness_map):
    # Assuming image, flow_map, and brightness_map are PyTorch tensors
    # and have been properly normalized and prepared for processing

    # Create a coordinate grid for the original image
    N, C, H, W = image.size()
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2)  # Shape: [H, W, 2]
    grid = grid.unsqueeze(0).repeat(N, 1, 1, 1).to(image.device)  # Repeat grid for each batch item

    # Apply flow map
    flow_map = flow_map.permute(0, 2, 3, 1)  # Change flow_map to [N, H, W, 2] to match grid
    warped_grid = grid + flow_map

    # Warp the image using the grid sample
    warped_image = F.grid_sample(image, warped_grid, mode='bilinear', padding_mode='border')

    # Adjust brightness
    adjusted_image = warped_image * brightness_map

    return adjusted_image

def display_image(tensor):
    """
    Display a tensor as an image.
    
    Parameters:
    - tensor: A PyTorch tensor of shape [1, C, H, W]
    """
    # Check if tensor needs to be detached and moved to CPU
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert to NumPy and adjust dimensions
    np_image = tensor.squeeze().numpy()  # Remove batch dimension, [C, H, W]
    np_image = np.transpose(np_image, (1, 2, 0))  # Change to [H, W, C]
    
    # Clip values to ensure they are in the [0, 1] range
    np_image = np.clip(np_image, 0, 1)
    
    # Display the image
    plt.imshow(np_image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

# Image dimensions
image_height, image_width = 100, 100

# Create a flow map for horizontal and vertical shifts
flow_map = torch.zeros(1, 2, image_height, image_width)
# Horizontal shift: increasing shift to the right
#flow_map[0, 0, :, :] = torch.linspace(-0.5, 0.5, image_width).unsqueeze(0).repeat(image_height, 1)
# Vertical shift: increasing shift downward
#flow_map[0, 1, :, :] = torch.linspace(-0.5, 0.5, image_height).unsqueeze(1).repeat(1, image_width)

# Create a brightness map with a horizontal gradient
brightness_map = torch.ones(1, 1, image_height, image_width) * torch.linspace(1, 0, image_width).unsqueeze(0).repeat(image_height, 1)

# Path to your saved image
image_path = 'eye-gaze-net/gafdASDFas.png'

# Load the image
image = Image.open(image_path)

# Convert the image to a tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])

image_tensor = transform(image)

# Adjust the tensor dimensions to match [1, 3, H, W]
# Assuming the image is grayscale, repeat it across 3 channels to simulate an RGB image
if image_tensor.size(0) == 1:  # Grayscale image
    image_tensor = image_tensor.repeat(3, 1, 1)

image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension

warped_and_adjusted_image = warp_image_with_flow_and_brightness(image_tensor, flow_map, brightness_map)
# Assuming 'warped_and_adjusted_image' is the output from your warp function
#display_image(warped_and_adjusted_image)