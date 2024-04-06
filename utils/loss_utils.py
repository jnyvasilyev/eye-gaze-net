import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def misalignment_tolerant_mse_loss(output, target, criterion, slack=1):
    """
    Calculate a misalignment-tolerant mean squared error by considering a slack of 1 pixel,
    where 'output' is shifted and both 'output' and 'target' are cropped to ensure
    padding is not included in the loss calculation.

    Args:
        output (torch.Tensor): The shifted output images from the model of shape [N, C, H, W].
        target (torch.Tensor): The target images of shape [N, C, H, W].
        criterion: The loss function to use, typically torch.nn.functional.mse_loss.
        slack (int): The number of pixels to consider for the slack in each direction.

    Returns:
        torch.Tensor: The misalignment-tolerant MSE loss for valid regions.
    """
    N, C, H, W = output.shape
    min_mse_loss = torch.inf

    for dy in range(-slack, slack + 1):
        for dx in range(-slack, slack + 1):
            # Shift 'output' and crop both 'output' and 'target' to exclude padding
            shifted_output = F.pad(output, (slack, slack, slack, slack), "constant", 0)
            shifted_output = shifted_output[
                :, :, slack + dy : slack + dy + H, slack + dx : slack + dx + W
            ]
            shifted_loss = shifted_output[:, :, slack : H - slack, slack : W - slack]

            # Crop 'target' to match the dimensions of 'shifted_output'
            valid_start_y, valid_end_y = max(0, -dy), H + min(0, -dy)
            valid_start_x, valid_end_x = max(0, -dx), W + min(0, -dx)
            valid_target = target[
                :, :, valid_start_y:valid_end_y, valid_start_x:valid_end_x
            ]
            target_loss = valid_target[:, :, slack : H - slack, slack : W - slack]

            # Calculate MSE loss for the current shift
            mse_loss = criterion(shifted_loss, target_loss)

            # Track the minimum MSE loss across all shifts
            if mse_loss < min_mse_loss:
                min_mse_loss = mse_loss

    return min_mse_loss


def misalignment_tolerant_ssim_loss(output, target, slack=1):
    """
    Calculate a misalignment-tolerant SSIM by considering a slack of 1 pixel,
    where 'output' is shifted and both 'output' and 'target' are cropped to ensure
    padding is not included in the loss calculation.

    Args:
        output (torch.Tensor): The shifted output images from the model of shape [N, C, H, W].
        target (torch.Tensor): The target images of shape [N, C, H, W].
        criterion: The loss function to use, ssim.
        slack (int): The number of pixels to consider for the slack in each direction.

    Returns:
        torch.Tensor: The misalignment-tolerant SSIM loss for valid regions.
    """
    N, C, H, W = output.shape
    min_ssim_loss = torch.inf

    for dy in range(-slack, slack + 1):
        for dx in range(-slack, slack + 1):
            # Shift 'output' and crop both 'output' and 'target' to exclude padding
            shifted_output = F.pad(output, (slack, slack, slack, slack), "constant", 0)
            shifted_output = shifted_output[
                :, :, slack + dy : slack + dy + H, slack + dx : slack + dx + W
            ]
            shifted_loss = shifted_output[:, :, slack : H - slack, slack : W - slack]

            # Crop 'target' to match the dimensions of 'shifted_output'
            valid_start_y, valid_end_y = max(0, -dy), H + min(0, -dy)
            valid_start_x, valid_end_x = max(0, -dx), W + min(0, -dx)
            valid_target = target[
                :, :, valid_start_y:valid_end_y, valid_start_x:valid_end_x
            ]
            target_loss = valid_target[:, :, slack : H - slack, slack : W - slack]

            # Calculate SSIM loss for the current shift
            ssim_loss = ssim(shifted_loss, target_loss, data_range=1, size_average=True)
            ssim_loss = 1 - ssim_loss
            # Track the minimum SSIM loss across all shifts
            if ssim_loss < min_ssim_loss:
                min_ssim_loss = ssim_loss

    return min_ssim_loss
