
import torch
import torch.nn as nn
import pennylane as qml

class AdvancedQuantumNeuralNetwork(nn.Module):
    """
    An advanced variational quantum classifier with more qubits and layers,
    integrated with PyTorch using PennyLane's TorchLayer.
    """
    def __init__(self, n_qubits, n_layers, device_name="default.qubit"):
        """
        Args:
            n_qubits (int): Number of qubits in the circuit.
            n_layers (int): Number of layers in the variational circuit.
            device_name (str): PennyLane device for simulation.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Define the quantum device
        dev = qml.device(device_name, wires=self.n_qubits)

        # Define the QNode, which contains the quantum circuit logic
        @qml.qnode(dev, interface='torch')
        def circuit(inputs, weights):
            # State preparation layer: encodes input features into the quantum state
            # qml.AngleEmbedding is a standard way to do this, equivalent to your RX loop
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='X')
            
            # Sequence of variational layers
            for i in range(self.n_layers):
                # A layer of general single-qubit rotations
                for j in range(self.n_qubits):
                    qml.Rot(*weights[i, j], wires=j)
                # A layer of entangling CNOTs in a ring-like topology
                for j in range(self.n_qubits):
                    qml.CNOT(wires=[j, (j + 1) % self.n_qubits])
            
            # Return the expectation value of a single observable (PauliZ on the first qubit)
            return qml.expval(qml.PauliZ(0))
        
        # Define the shape of the trainable weights for the TorchLayer
        # Shape: (num_layers, num_qubits, 3 parameters per Rot gate)
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        
        # Create the TorchLayer that makes the QNode behave like a PyTorch layer
        self.qml_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        
        # Add a classical bias parameter
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
        Defines the data flow through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: The model's output logits of shape (batch_size, 1).
        """
        # 1. Flatten the input image data from [B, C, H, W] to [B, Features]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        # 2. Select the first `n_qubits` features.
        # This is a simple feature selection method. The number of features must
        # be at least the number of qubits.
        if x.shape[1] < self.n_qubits:
            raise ValueError(
                f"Input features ({x.shape[1]}) are less than n_qubits ({self.n_qubits}). "
                "Please increase image size or reduce n_qubits."
            )
        x_selected_features = x[:, :self.n_qubits]
        
        # 3. Pass the selected features through the quantum layer
        q_out = self.qml_layer(x_selected_features)
        
        # 4. Add the classical bias and reshape for compatibility with BCEWithLogitsLoss
        return (q_out + self.bias).view(batch_size, 1)