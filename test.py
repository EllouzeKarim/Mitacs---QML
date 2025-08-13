import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane.qnn.torch import TorchLayer
from models.model_unet3d import UNet3D  # assume your UNet3D is importable

# ----------------------
# Helper: find patch grid
# ----------------------
def find_patch_grid(n_qubits):
    """
    Find integer (pd, ph, pw) with product = n_qubits.
    This picks factors close to a cube root for balanced patches.
    Raises ValueError if no exact factorization with 3 integers found.
    """
    # Try to find factors by brute force up to cube root range
    for pd in range(1, int(round(n_qubits ** (1/3))) + 5):
        if n_qubits % pd != 0:
            continue
        rem = n_qubits // pd
        for ph in range(1, int(math.sqrt(rem)) + 5):
            if rem % ph != 0:
                continue
            pw = rem // ph
            return pd, ph, pw
    # fallback: try pairs (1,1,n)
    if n_qubits > 0:
        return 1, 1, n_qubits
    raise ValueError(f"Cannot factor n_qubits={n_qubits} into 3 integers")

# ----------------------
# Quantum filter layer
# ----------------------
class PatchBasedQuantumFilter(nn.Module):
    def __init__(self, n_qubits=8, n_ent_layers=1, dev_name="default.qubit"):
        """
        n_qubits: number of qubits == number of patches
        n_ent_layers: how many layers of (entangling + rotations) to use
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_ent_layers = n_ent_layers

        dev = qml.device(dev_name, wires=n_qubits)

        # Build a qnode with inputs (length n_qubits) and a trainable weight tensor
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def qnode(inputs, weights):
            # inputs: shape (n_qubits,)
            # weights: shape (n_ent_layers, n_qubits, 2)  (two trainable params per qubit per layer here)
            # Layer 0: encode inputs
            for i in range(self.n_qubits):
                qml.RY(inputs[i] + weights[0, i, 0], wires=i)

            # Additional entangling + rotation layers
            for layer in range(self.n_ent_layers):
                # simple ring entanglement (CNOT chain)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])  # close the ring

                # rotations after entanglement (use weights[layer, i, 1])
                for i in range(self.n_qubits):
                    # stagger parameters so every layer has some trainable freedom
                    qml.RZ(weights[layer, i, 1], wires=i)
                    qml.RY(weights[layer, i, 0], wires=i)

            # measure Z on each qubit -> returns n_qubits scalars in [-1,1]
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # weight shape: (n_ent_layers, n_qubits, 2)
        weight_shapes = {"weights": (self.n_ent_layers, self.n_qubits, 2)}
        self.q_layer = TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        """
        x: (B, n_qubits) - a batch of patch-encoded scalars
        returns: (B, n_qubits) quantum-processed scalars
        """
        # TorchLayer supports batching: pass the whole (B, n_qubits) in one call
        return self.q_layer(x)

# ----------------------
# The UNet wrapper with patch-based quantum encoding
# ----------------------
class UNet3DWithPatchQPF(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, n_qubits=8, n_ent_layers=1, unet_kwargs=None):
        """
        in_channels: channels of input volume
        n_qubits: how many qubits (== patches)
        n_ent_layers: depth of entangling layers in the qnode
        unet_kwargs: dict forwarded to UNet3D (optional)
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.pd, self.ph, self.pw = find_patch_grid(n_qubits)

        # 1) Local patch summarizer:
        # Use a 1-channel conv to produce a single feature map, then AdaptiveAvgPool to grid
        self.patch_projector = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),              # -> (B,1,D,H,W)
            nn.AdaptiveAvgPool3d((self.pd, self.ph, self.pw)),     # -> (B,1,pd,ph,pw)
            nn.Flatten(1)                                         # -> (B, pd*ph*pw) == (B, n_qubits)
        )

        # 2) Quantum filter operating on patch scalars
        self.qpf = PatchBasedQuantumFilter(n_qubits=n_qubits, n_ent_layers=n_ent_layers)

        # 3) Expand quantum outputs back to a patch-grid and a spatial map (1 channel)
        small_spatial = (self.pd, self.ph, self.pw)
        self.quantum_feature_expander = nn.Sequential(
            nn.Linear(n_qubits, 1 * self.pd * self.ph * self.pw),
            nn.ReLU()
        )

        # 4) UNet: we will concatenate the quantum map as an extra channel
        if unet_kwargs is None:
            unet_kwargs = {}
        # Ensure UNet3D constructor exists and accepts in_channels and num_classes
        self.unet = UNet3D(in_channels=in_channels + 1, num_classes=num_classes, **unet_kwargs)

        self._small_spatial = small_spatial

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        returns: UNet output
        """
        B, C, D, H, W = x.shape

        # 1) Project to per-patch scalars
        patch_scalars = self.patch_projector(x)  # (B, n_qubits)

        # 2) Quantum processing (batched)
        q_out = self.qpf(patch_scalars)          # (B, n_qubits)

        # 3) Expand back to a small patch-grid and channel
        q_features = self.quantum_feature_expander(q_out)  # (B, pd*ph*pw)
        q_features = q_features.view(B, 1, *self._small_spatial)  # (B,1,pd,ph,pw)

        # 4) Upsample to original resolution
        q_features = F.interpolate(q_features, size=(D, H, W), mode='trilinear', align_corners=True)

        # 5) Concatenate as an extra channel and feed UNet
        x_aug = torch.cat([x, q_features], dim=1)  # (B, in_channels+1, D, H, W)
        return self.unet(x_aug)
