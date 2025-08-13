import torch
import torch.nn as nn
import torch.nn.functional as F
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
                qml.RY(inputs[i] + weights[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_qubits,)}
        qnode = qml.QNode(
            quantum_circuit,
            dev,
            interface="torch",
            diff_method="backprop",
            cache=True,       
            cachesize=10000   
        )

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

        self.unet = UNet3D(in_channels=in_channels + 1, num_classes=num_classes)

    def forward(self, x):  # x: (B, 1, D, H, W)
        B, C, D, H, W = x.shape

        q_input = self.quantum_input_projector(x)   # (B, 4)
        q_output = self.qpf(q_input)                # (B, n_qubits)

        # Expand and reshape quantum output to match input spatially
        q_features = self.quantum_feature_expander(q_output)  # (B, C*4*4*4)
        q_features = q_features.view(B, 1, 4, 4, 4)            # small spatial map

        # Upsample to match x
        q_features = F.interpolate(q_features, size=(D, H, W), mode='trilinear', align_corners=True)

        # Concatenate quantum features as an additional channel
        x_aug = torch.cat([x, q_features], dim=1)  # (B, C+1, D, H, W)

        return self.unet(x_aug)

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

        self.unet = UNet3D(in_channels=in_channels + 1, num_classes=num_classes)

    def forward(self, x):  # x: (B, 1, D, H, W)
        B, C, D, H, W = x.shape

        q_input = self.quantum_input_projector(x)   # (B, 4)
        q_output = self.qpf(q_input)                # (B, n_qubits)

        # Expand and reshape quantum output to match input spatially
        q_features = self.quantum_feature_expander(q_output)  # (B, C*4*4*4)
        q_features = q_features.view(B, 1, 4, 4, 4)            # small spatial map

        # Upsample to match x
        q_features = F.interpolate(q_features, size=(D, H, W), mode='trilinear', align_corners=True)

        # Concatenate quantum features as an additional channel
        x_aug = torch.cat([x, q_features], dim=1)  # (B, C+1, D, H, W)

        return self.unet(x_aug)
