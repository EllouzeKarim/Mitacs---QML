import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True, dropout=0.3):
        super(ConvBlock3D, self).__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Efficient3DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(Efficient3DCNN, self).__init__()

        self.conv1 = ConvBlock3D(in_channels, 32, pool=True, dropout=0.1)
        self.conv2 = ConvBlock3D(32, 64, pool=True, dropout=0.1)
        self.conv3 = ConvBlock3D(64, 128, pool=True, dropout=0.2)
        self.conv4 = ConvBlock3D(128, 256, pool=True, dropout=0.3)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)  # [B, 32, D/2, H/2, W/2]
        x = self.conv2(x)  # [B, 64, D/4, H/4, W/4]
        x = self.conv3(x)  # [B, 128, D/8, H/8, W/8]
        x = self.conv4(x)  # [B, 256, D/16, H/16, W/16]
        x = self.global_pool(x)  # -> [B, 256, 1, 1, 1]
        out = self.classifier(x)  # -> [B, 1]
        return out
