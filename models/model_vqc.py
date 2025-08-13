import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np # Use PennyLane's numpy for compatibility

class VariationalQuantumClassifier(nn.Module):
    """
    A variational quantum classifier model integrated with PyTorch.
    It wraps a PennyLane QNode as a torch.nn.Module.
    """
    def __init__(self, n_qubits, n_layers, device_name="default.qubit"):
        """
        Args:
            n_qubits (int): Number of qubits in the circuit. This will also be the
                            number of features used from the input data.
            n_layers (int): Number of layers in the variational circuit.
            device_name (str): PennyLane device to run the simulation on.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Define the quantum device
        dev = qml.device(device_name, wires=self.n_qubits)

        # Define the QNode that will be wrapped by the TorchLayer
        @qml.qnode(dev, interface='torch')
        def circuit(inputs, weights):
            # State preparation (feature encoding)
            # We use AngleEmbedding, which is equivalent to your RX loop
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='X')
            
            # Variational layers
            for i in range(self.n_layers):
                # A layer of rotations
                for j in range(self.n_qubits):
                    qml.Rot(*weights[i, j], wires=j)
                # A layer of entangling CNOTs
                for j in range(self.n_qubits):
                    qml.CNOT(wires=[j, (j + 1) % self.n_qubits])
            
            # Return the expectation value of PauliZ on the first qubit
            return qml.expval(qml.PauliZ(0))
        
        # Define the shape of the trainable weights for the TorchLayer
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        
        # Create the TorchLayer
        self.qml_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        
        # Add a classical bias term
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
        The forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).
        
        Returns:
            torch.Tensor: The model's output (logits) of shape (batch_size, 1).
        """
        # 1. Flatten the input image data
        # Input 'x' has shape [batch_size, channels, H, W] or similar
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        # 2. Select the first `n_qubits` features for encoding.
        # This is a simple feature selection strategy.
        # Ensure the input data has at least `n_qubits` features.
        if x.shape[1] < self.n_qubits:
            raise ValueError(
                f"Input features ({x.shape[1]}) are less than n_qubits ({self.n_qubits}). "
                "Please increase image size or reduce n_qubits."
            )
        x = x[:, :self.n_qubits]
        
        # 3. Pass the features through the quantum layer
        q_out = self.qml_layer(x)
        
        # 4. Add the classical bias and return
        # The output shape will be (batch_size,), which we reshape for BCEWithLogitsLoss
        return (q_out + self.bias).view(batch_size, 1)