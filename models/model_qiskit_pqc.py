import torch
import torch.nn as nn
from qiskit.circuit import QuantumCircuit, ParameterVector, ClassicalRegister, QuantumRegister
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

def create_pqc_for_connector(num_qubits, num_layers):
    """
    Creates a PQC compatible with SamplerQNN for binary classification.
    It now includes a single classical bit for a single measurement.
    """
    # Use separate ParameterVectors for inputs and weights
    input_params = ParameterVector("x", num_qubits)
    weight_params = ParameterVector("w", num_qubits * num_layers)
    
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(1) # A single classical bit is sufficient
    qc = QuantumCircuit(qr, cr)
    
    param_idx = 0
    for layer in range(num_layers):
        # Data encoding layer
        for i in range(num_qubits):
            qc.ry(input_params[i], qr[i])
            
        # Variational/Weight layer
        for i in range(num_qubits):
            qc.rz(weight_params[param_idx], qr[i])
            param_idx += 1
            
        # Entanglement layer
        if num_qubits > 1:
            for i in range(num_qubits - 1):
                qc.cx(qr[i], qr[i+1])
            qc.cx(qr[num_qubits - 1], qr[0])
        
        if layer < num_layers - 1:
            qc.barrier()
    
    # --- CRITICAL FIX: Measure only the first qubit into the classical bit ---
    qc.measure(qr[0], cr[0])
            
    return qc, input_params, weight_params


class QiskitPQCClassifier(nn.Module):
    """
    A Qiskit PQC that handles its own data preprocessing (PCA, Scaling)
    inside the forward pass, using pre-fitted transformers.
    """
    def __init__(self, n_features, n_layers, pca_transformer, standard_scaler):
        super().__init__()
        self.pca = pca_transformer
        self.scaler = standard_scaler
        
        # --- Define the Quantum Part ---
        self.qc, input_params, weight_params = create_pqc_for_connector(
            num_qubits=n_features, 
            num_layers=n_layers
        )
        
        # --- CRITICAL FIX 2: Simplify the SamplerQNN ---
        # By removing the `interpret` function, SamplerQNN will return the
        # raw quasi-distribution of probabilities for the classical bits.
        # Since we have one classical bit, this will be {0: prob_0, 1: prob_1}.
        sampler_qnn = SamplerQNN(
            circuit=self.qc,
            input_params=input_params,
            weight_params=weight_params,
        )
        self.qnn = TorchConnector(sampler_qnn)

    def _preprocess_batch(self, x_raw_batch):
        """Handles the CPU/GPU transfer and preprocessing for a single batch."""
        x_np = x_raw_batch.detach().cpu().numpy()
        batch_size = x_np.shape[0]
        x_flat = x_np.reshape(batch_size, -1)
        x_pca = self.pca.transform(x_flat)
        x_scaled = self.scaler.transform(x_pca)
        return torch.from_numpy(x_scaled).float().to(x_raw_batch.device)

    def forward(self, x):
        """ The forward pass for the model. """
        x_preprocessed = self._preprocess_batch(x)
        
        # --- CRITICAL FIX 3: Process the QNN output correctly ---
        # The output `q_out` will have shape [B, 2], where B is the batch size.
        # Column 0 is the probability of measuring '0', Column 1 is for '1'.
        q_out = self.qnn(x_preprocessed)
        
        # We only need the probability of class 1 for binary cross-entropy.
        prob_class_1 = q_out[:, 1]
        
        # Convert this probability to a logit for BCEWithLogitsLoss.
        epsilon = 1e-6
        prob = torch.clamp(prob_class_1, epsilon, 1 - epsilon)
        logits = torch.log(prob / (1 - prob))
        
        # Ensure the output has shape [B, 1] for the loss function.
        return logits.view(-1, 1)