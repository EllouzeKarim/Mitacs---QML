import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

class InceptionV3(nn.Module):
    """
    An adapted InceptionV3 model for 3D medical imaging.
    
    This model uses a 3D "stem" to process the input volume and extract
    feature maps that are then reshaped and fed into a pre-trained 2D
    InceptionV3 model. This allows leveraging the power of ImageNet
    pre-training for a 3D classification task.
    """
    def __init__(self, in_channels=1, num_classes=1, pretrained=True):
        super(InceptionV3, self).__init__()

        # --- Part 1: 3D Stem ---
        # This part of the network processes the 3D volume. Its goal is to
        # learn to extract relevant 2D-like feature maps from the 3D data.
        self.stem_3d = nn.Sequential(
            # Start with a convolution to increase channel depth
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            # Reduce spatial dimensions
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        )

        # --- Part 2: Bridge from 3D to 2D ---
        # This layer takes the feature map from the 3D stem and projects it
        # into the 3 channels that the 2D InceptionV3 expects.
        # The input channels to this layer will be (16 channels * Depth of feature map).
        # We assume the depth dimension will be around 64 after the stem.
        self.bridge_conv = nn.Conv2d(16 * 64, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bridge_bn = nn.BatchNorm2d(3)
        self.bridge_relu = nn.ReLU(inplace=True)

        # --- Part 3: Pre-trained 2D InceptionV3 Backbone ---
        # Load the standard InceptionV3 model
        self.inception_2d = inception_v3(weights=None, aux_logits=False)
        
        if pretrained:
            # Manually load pre-trained weights to avoid conflicts with our modified stem
            state_dict = Inception_V3_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
            # We only load weights for layers that match. The first conv layer `Conv2d_1a_3x3`
            # and the `fc` layer will not match and will be ignored due to `strict=False`.
            self.inception_2d.load_state_dict(state_dict, strict=False)

        # We are using our own stem, so we replace the original InceptionV3 stem.
        # This new first convolution takes our 3-channel bridge output.
        self.inception_2d.Conv2d_1a_3x3 = nn.Conv2d(3, 32, kernel_size=3, stride=2, bias=False)

        # Replace the final fully connected layer for our binary classification task.
        self.inception_2d.fc = nn.Linear(self.inception_2d.fc.in_features, num_classes)

    def forward(self, x):
        # x has shape [B, 1, D, H, W], e.g., [32, 1, 64, 64, 64]

        # 1. Pass through the 3D stem
        # Output shape: [B, 16, D, H/2, W/2], e.g., [32, 16, 64, 32, 32]
        x_3d = self.stem_3d(x)
        
        # 2. Bridge the gap from 3D to 2D
        # We "flatten" the depth and channel dimensions together to create
        # a single, very deep channel dimension for a 2D convolution.
        b, c, d, h, w = x_3d.shape
        x_2d_like = x_3d.view(b, c * d, h, w) # Shape: [B, 16*64, 32, 32]

        # Project the deep channels down to the 3 channels InceptionV3 expects
        x_bridge = self.bridge_conv(x_2d_like) # Shape: [B, 3, 32, 32]
        x_bridge = self.bridge_relu(self.bridge_bn(x_bridge))

        # 3. Pre-process for the InceptionV3 backbone
        # InceptionV3 expects 299x299 images. We upsample our feature map.
        x_upsampled = F.interpolate(x_bridge, size=(299, 299), mode='bilinear', align_corners=False)

        # 4. Pass through the pre-trained 2D InceptionV3 model
        # The output will be our final logit for classification
        out = self.inception_2d(x_upsampled)

        # Ensure the output shape is consistent with other models ([B] or [B, 1])
        return out.squeeze(-1) if out.ndim > 1 else out