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


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, final=False):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, 3)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, 3)
        self.conv3 = DepthwiseSeparableConv(out_channels, out_channels, 3)
        self.final = final

        self.se = SEBlock(out_channels)
        self.convstride = DepthwiseSeparableConv(
            out_channels, out_channels, 3, stride=2, padding=1
        )

    def forward(self, x):
        # Residual skip connection, might need to add downsample depending on input and output channels
        residual = self.conv1(x)
        out = self.conv2(residual)
        out = torch.add(out, residual)
        out = self.conv3(out)
        out = self.se(out)
        if not self.final:
            out = self.convstride(out)
        return out


class ECCNet(nn.Module):
    def __init__(self):
        super(ECCNet, self).__init__()

        self.conv_block1 = ResidualConvBlock(7, 32)
        self.conv_block2 = ResidualConvBlock(32, 64)
        self.conv_block3 = ResidualConvBlock(64, 128)
        self.conv_block4 = ResidualConvBlock(128, 256)

        self.upconv_block3 = ResidualConvBlock(256, 256, final=True)
        self.upconv_block2 = ResidualConvBlock(128 + 128, 128, final=True)
        self.upconv_block1 = ResidualConvBlock(64 + 64, 32, final=True)

        self.upconv_3 = nn.ConvTranspose2d(
            256, 128, 3, stride=2, padding=1, output_padding=1
        )
        self.upconv_2 = nn.ConvTranspose2d(
            128, 64, 3, stride=2, padding=1, output_padding=1
        )

        self.final_upconv_2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.final_upconv_1 = nn.ConvTranspose2d(16, 8, 2, stride=2)

        self.out = nn.Conv2d(8, 3, 1, padding=0)

    def forward(self, img, angle, head_pose):
        x = torch.cat([img, angle, head_pose], dim=1)

        # Encoder
        x1 = self.conv_block1(x)
        # x1 = F.max_pool2d(x1, 2)
        x2 = self.conv_block2(x1)
        # x2 = F.max_pool2d(x2, 2)
        x3 = self.conv_block3(x2)
        # x3 = F.max_pool2d(x3, 2)
        x4 = self.conv_block4(x3)
        # x4 = F.max_pool2d(x4, 2)
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
