import torch
import torch.nn as nn
import pennylane as qml
from models.model_3d import I3D 

class QuantumKernelClassifier(nn.Module):
    """
    A hybrid model using a pre-trained I3D backbone for feature extraction
    and a quantum kernel-inspired classification layer.
    """
    def __init__(self, n_qubits=4, **kwargs):
        super().__init__()
        self.n_qubits = n_qubits

        # --- Part 1: Classical Feature Extractor ---
        # Load a pre-trained I3D model as the backbone.
        # We will use its output as features, so we need to access the layer before the final classifier.
        i3d_backbone = I3D(
            num_classes=400,         # Original number of classes for pre-trained weights
            input_channels=3,        
            pre_trained=True,
            freeze_bn=True,
        )
        # We are hijacking the I3D model. The feature vector is the output of the avg_pool layer.
        # We replace the final classification layer with an identity layer.
        i3d_backbone.conv3d_0c_1x1 = nn.Identity()
        self.i3d_features = i3d_backbone
        
        # Freeze the backbone to prevent it from being re-trained
        for param in self.i3d_features.parameters():
            param.requires_grad = False
        
        # This layer projects the 1024 features from I3D down to n_qubits
        # for the quantum circuit. It is a trainable classical layer.
        self.feature_projector = nn.Sequential(
            nn.Linear(1024, n_qubits),
            nn.Tanh() # Tanh scales outputs to [-1, 1], suitable for angle encoding
        )

        # --- Part 2: Quantum Kernel Layer ---
        # FIX 1: Provide a valid device name string
        dev_name = "default.qubit"
        self.dev = qml.device(dev_name, wires=n_qubits)

        # This is the quantum circuit that embeds classical data (features).
        # It defines our quantum feature map.
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def quantum_state_circuit(x):
            # A common feature map: AngleEmbedding
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation='X')
            return qml.state()

        self.quantum_state = quantum_state_circuit

        # FIX 2: Create a trainable reference vector.
        # This vector represents the "ideal" or "center" point for the positive class.
        # The model will learn the best reference point during training.
        # Initializing with small random values.
        ref_init = torch.randn(n_qubits)
        self.reference_vector = nn.Parameter(ref_init)
        
        # --- Part 3: Final Classifier ---
        # This final layer is not strictly necessary as the kernel output can be used
        # as a logit, but it adds a final learnable bias and scale.
        self.final_classifier = nn.Linear(1, 1)

    def forward(self, x):  # Input x has shape: (Batch, Channels, Depth, Height, Width)
        B = x.shape[0]

        # Step 1: Extract features using the frozen I3D backbone
        # The I3D forward pass needs to be traced carefully to get the features.
        # Based on the I3D code, the feature vector is the output of avg_pool.
        # We run the forward pass up to the point before the final classification layer.
        
        # Since we replaced the last layer with Identity, this gives us the 1024-dim features
        features = self.i3d_features(x)  # Shape: (B, 1024)
        features = features.view(B, -1)  # Ensure it's flat

        # Step 2: Project classical features to the quantum input dimension
        q_inputs = self.feature_projector(features)  # Shape: (B, n_qubits)

        # Step 3: Compute the quantum kernel (fidelity)
        # Get the quantum state for the learned reference vector.
        # We normalize it before using it in the embedding.
        ref_state = self.quantum_state(torch.tanh(self.reference_vector))

        outputs = []
        for i in range(B):
            # Get the quantum state for the current data sample
            psi = self.quantum_state(q_inputs[i])
            # Calculate the fidelity (squared inner product)
            fidelity = torch.abs(torch.vdot(psi, ref_state)) ** 2
            outputs.append(fidelity.unsqueeze(0))

        kernel_outputs = torch.stack(outputs)  # Shape: (B, 1)
        
        # Step 4: Pass the kernel value through a final linear layer to get the logit
        # This allows the model to scale and shift the fidelity value appropriately.
        return self.final_classifier(kernel_outputs)