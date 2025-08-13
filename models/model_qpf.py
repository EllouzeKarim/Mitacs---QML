import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_cnn3d import Efficient3DCNN
from models.model_inception import InceptionV3
from models.model_resnet3d import ResNet3D
from models.model_unet3d import UNet3D
import pennylane as qml
from pennylane import numpy as np
from pennylane.qnn.torch import TorchLayer


class QuantumFilterLayer(nn.Module):
    def __init__(self, n_qubits=24, n_layers=1):
        super().__init__()
        self.n_qubits = n_qubits

        dev = qml.device("default.qubit", wires=n_qubits)

        def quantum_circuit(inputs, weights):
            for i in range(n_qubits):
                # Combine input and weight in a single rotation to reduce gates
                qml.RY(inputs[i] + weights[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_qubits,)}  # Flatten weights to 1D 
        qnode = qml.QNode(quantum_circuit, dev, interface="torch", diff_method="backprop")
        self.q_layer = TorchLayer(qnode, weight_shapes)


    def forward(self, x):  # x: (B, n_qubits)
        B = x.shape[0]
        results = []
        for i in range(B):
            results.append(self.q_layer(x[i]))
        return torch.stack(results)  # (B, n_qubits)

class UNet3DWithQPF(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, n_qubits=4):
        super().__init__()
        self.n_qubits = n_qubits

        self.qpf = QuantumFilterLayer(n_qubits=n_qubits)

        # Project input to quantum-friendly vector
        self.quantum_input_projector = nn.Sequential(
            nn.Conv3d(in_channels, 4, kernel_size=1),
            nn.AdaptiveAvgPool3d(1),  # → (B, 4, 1, 1, 1)
            nn.Flatten(1)             # → (B, 4)
        )

        # Project quantum output to feature map shape and concat with input
        self.quantum_feature_expander = nn.Sequential(
            nn.Linear(n_qubits, in_channels * 4 * 4 * 4),
            nn.ReLU()
        )

        self.attention_mask_generator = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 4 * 4 * 4),
            nn.Sigmoid() # Sigmoid ensures mask values are between 0 and 1.
        )

        # self.unet = UNet3D(in_channels=in_channels, num_classes=num_classes)
        self.unet = ResNet3D(depth=34, in_channels=in_channels, num_classes=num_classes)
        # self.unet = Efficient3DCNN(in_channels=in_channels, num_classes=num_classes)
        self.unet = InceptionV3(in_channels=in_channels, num_classes=num_classes)

    
    def forward(self, x):  # x: (B, 1, D, H, W)
        B, C, D, H, W = x.shape

        # 1. Get a vector summary of the input image.
        q_input = self.quantum_input_projector(x)

        # 2. Process the summary through the quantum filter.
        q_output = self.qpf(q_input) # Shape: [B, n_qubits]

        # --- THIS IS THE CORRECT "ATTENTION MASK" FORWARD PASS ---
        # 3. Generate the attention mask using the correct module name.
        attention_mask = self.attention_mask_generator(q_output)
        attention_mask = attention_mask.view(B, 1, 4, 4, 4) # Reshape to a small 3D volume.

        # 4. Upsample the mask to match the original input's spatial dimensions.
        attention_mask = F.interpolate(attention_mask, size=(D, H, W), mode='trilinear', align_corners=False)

        # 5. Apply the quantum filter as a multiplicative mask.
        # This re-weights the input features without changing the number of channels.
        x_filtered = x * attention_mask

        # 6. Pass the filtered, single-channel tensor to the UNMODIFIED backbone.
        return self.unet(x_filtered)
 