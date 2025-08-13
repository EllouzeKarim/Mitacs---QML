import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """[Conv3D + ReLU] x 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(UNet3D, self).__init__()

        self.down1 = DoubleConv3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = DoubleConv3D(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.down3 = DoubleConv3D(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.down4 = DoubleConv3D(128, 256)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv3D(256, 512)

        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.conv4 = DoubleConv3D(512, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv3D(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv3D(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv3D(64, 32)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # -> [B, 32, 1, 1, 1]
            nn.Flatten(),             # -> [B, 32]
            nn.Linear(32, num_classes)  # -> [B, 1]
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        bn = self.bottleneck(self.pool4(d4))

        up4 = self.up4(bn)
        up4 = torch.cat([up4, d4], dim=1)
        up4 = self.conv4(up4)

        up3 = self.up3(up4)
        up3 = torch.cat([up3, d3], dim=1)
        up3 = self.conv3(up3)

        up2 = self.up2(up3)
        up2 = torch.cat([up2, d2], dim=1)
        up2 = self.conv2(up2)

        up1 = self.up1(up2)
        up1 = torch.cat([up1, d1], dim=1)
        up1 = self.conv1(up1)

        out = self.classifier(up1)  
        return out
