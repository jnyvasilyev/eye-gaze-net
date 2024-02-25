import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, 3)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, 3)
        self.conv3 = DepthwiseSeparableConv(out_channels, out_channels, 3)

    def forward(self, x):
        # Residual skip connection, might need to add downsample depending on input and output channels
        residual = self.conv1(x)
        out = self.conv2(residual)
        out = torch.add(out, residual)
        out = self.conv3(out)
        return out

class ECCNet(nn.Module):
    def __init__(self):
        super(ECCNet, self).__init__()

        self.conv_block1 = ResidualConvBlock(5, 32)
        self.conv_block2 = ResidualConvBlock(32, 64)
        self.conv_block3 = ResidualConvBlock(64, 128)
        self.conv_block4 = ResidualConvBlock(128, 256)

        self.upconv_block3 = ResidualConvBlock(256 + 128, 128)
        self.upconv_block2 = ResidualConvBlock(128 + 64, 64)
        self.upconv_block1 = ResidualConvBlock(64 + 32, 32)

        self.out = nn.Conv2d(32, 3, kernel_size = 3, padding=1)

    def forward(self, img, angle):
        x = torch.cat([img, angle], dim=1)

        # Encoder
        x1 = self.conv_block1(x)
        x2 = F.max_pool2d(x1, 2)
        x2 = self.conv_block2(x2)
        x3 = F.max_pool2d(x2, 2)
        x3 = self.conv_block3(x3)
        x4 = F.max_pool2d(x3, 2)
        x4 = self.conv_block4(x4)

        # Decoder with skip connections
        x = F.interpolate(x4, scale_factor=2, mode='nearest')
        x = torch.cat([x, x3], dim=1)
        x = self.upconv_block3(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, x2], dim=1)
        x = self.upconv_block2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, x1], dim=1)
        x = self.upconv_block1(x)

        output = self.out(x)
        flow = output[:, :2, :, :]
        brightness_map = torch.sigmoid(output[:, 2, :, :].unsqueeze(1))

        return flow.permute(0, 2, 3, 1), brightness_map

# Create the model
model = ECCNet()
