import torch
import torch.nn as nn
from models.model_resnet3d import BasicBlock3D, ResNet3D
import pennylane as qml
from pennylane.qnn import TorchLayer

import logging
logger = logging.getLogger("pennylane")
logger.setLevel(logging.WARNING)

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits=4, dev_name="default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits

        # 1. Classical feature extractor (CNN 3D)
        self.cnn = ResNet3D(depth=34)

        # 2. Classical → Quantum projection
        self.classical_to_quantum = nn.Sequential(
            nn.Flatten(),              # → (B, 1)
            nn.Linear(1, n_qubits),    # match CNN output (1) to n_qubits
            nn.Tanh()
        )

        # 3. Quantum circuit
        dev = qml.device(dev_name, wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(inputs[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.q_layer = TorchLayer(circuit, weight_shapes={})  # no output_dim

        # 4. Final classification
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x shape: (B, 3, D, H, W)
        x = self.cnn(x)                         # → (B, 1, 1, 1, 1)
        x = x.unsqueeze(1) 
        x = self.classical_to_quantum(x)        # → (B, n_qubits)
        B = x.shape[0]
        results = []
        for i in range(B):
            results.append(self.q_layer(x[i]))  # Each → (n_qubits,)
        x = torch.stack(results)                # → (B, n_qubits)
        return self.classifier(x)               # → (B, 1)
