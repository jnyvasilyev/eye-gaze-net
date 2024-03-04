import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels
        )
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
    
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, upsample=2):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.kernel_size = kernel_size
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)  
        if(self.kernel_size != 2):
            x = self.reflection_pad(x)
        else:
            x = F.pad(x, (0, 1, 0, 1), mode = 'replicate')
        out = self.conv2d(x)
        return out


class ECCNet(nn.Module):
    def __init__(self):
        super(ECCNet, self).__init__()

        self.conv_block1 = ResidualConvBlock(5, 32)
        self.conv_block2 = ResidualConvBlock(32, 64)
        self.conv_block3 = ResidualConvBlock(64, 128)
        self.conv_block4 = ResidualConvBlock(128, 256)

        self.upconv_block3 = ResidualConvBlock(256, 256)
        self.upconv_block2 = ResidualConvBlock(128 + 128, 128)
        self.upconv_block1 = ResidualConvBlock(64 + 64, 32)

        self.upconv_3 = UpsampleConvLayer(256, 128, 3)
        self.upconv_2 = UpsampleConvLayer(128, 64, 3)

        self.final_upconv_2 = UpsampleConvLayer(32, 16, 2)
        self.final_upconv_1 = UpsampleConvLayer(16, 8, 2)

        self.out = nn.Conv2d(8, 3, 3, padding=1)

    def forward(self, img, angle):
        x = torch.cat([img, angle], dim=1)

        # Encoder
        x1 = self.conv_block1(x)
        x1 = F.max_pool2d(x1, 2)
        x2 = self.conv_block2(x1)
        x2 = F.max_pool2d(x2, 2)
        x3 = self.conv_block3(x2)
        x3 = F.max_pool2d(x3, 2)
        x4 = self.conv_block4(x3)
        x4 = F.max_pool2d(x4, 2)
        x5 = self.upconv_block3(x4)
        x5 = self.upconv_3(x5)


        # Decoder with skip connections
        x = torch.cat([x5, x3], dim=1)
        x = self.upconv_block2(x)
        x = self.upconv_2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv_block1(x)
        x = self.final_upconv_2(x)
        x = self.final_upconv_1(x)

        output = self.out(x)
        flow = output[:, :2, :, :]
        brightness_map = torch.sigmoid(output[:, 2, :, :].unsqueeze(1))

        return flow.permute(0, 2, 3, 1), brightness_map
    
model = ECCNet()
