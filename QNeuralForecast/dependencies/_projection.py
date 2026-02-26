from torch import nn
import torch
import pennylane as qml

# Define the quantum device
n_qubits = 4  # Adjust based on simulation speed (4-8 is usually fast)
# dev = qml.device("default.qubit", wires=n_qubits)

def get_quantum_circuit(device_name="default.qubit"):
    dev = qml.device(device_name, wires=n_qubits)
    
    @qml.qnode(dev)
    def quantum_projection_circuit(inputs, weights):
        # 1. State Preparation / Data Encoding
        # H -> Ry(arctan(x)) -> Rz(arctan(x^2))
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            # Use accessing column i for all batches: inputs[..., i]
            # This handles both batched inputs (batch, n_qubits) and single inputs (n_qubits)
            feature = inputs[..., i]
            qml.RY(torch.arctan(feature), wires=i)
            qml.RZ(torch.arctan(feature**2), wires=i)
        
        # 2. Variational Layer (Entanglement + Rotation)
        # Get number of layers from weights shape
        n_layers = weights.shape[0] if len(weights.shape) > 2 else 1
        
        for l in range(n_layers):
            # CNOT chain: 0->1, 1->2, ..., (n-1)->0
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
                
            # Rotations: R(alpha, beta, gamma)
            # weights shape: (n_layers, n_qubits, 3)
            for i in range(n_qubits):
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
        
        # 3. Measurement
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    
    return quantum_projection_circuit


class QuantumProjection(nn.Module):
    def __init__(self, input_size, output_size, n_qubits=4, n_layers=1, circuit_device="default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        
        # 1. Compress input to fit on quantum chip
        self.pre_net = nn.Linear(input_size, n_qubits)
        
        # 2. Quantum Layer
        # Circuit uses n_layers blocks, each with one rotation per qubit (3 params)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        # Get circuit for the specific device
        self.circuit = get_quantum_circuit(circuit_device)
        self.q_layer = qml.qnn.TorchLayer(self.circuit, weight_shapes)
        
        # 3. Expand output to forecast horizon
        self.post_net = nn.Linear(n_qubits, output_size)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Force q_layer to stay on CPU if target is MPS
        try:
            device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
            if device and device.type == "mps":
                self.q_layer.to("cpu")
        except:
            pass
        return self

    def forward(self, x):
        device = x.device
        x = torch.tanh(self.pre_net(x)) * torch.pi # Scale to [-pi, pi] for AngleEmbedding

        # Handle MPS device issues with Pennylane
        if device.type == "mps":
            # Execute on CPU (q_layer is already on CPU due to .to() override)
            x = self.q_layer(x.cpu())
            # Move result back to device
            x = x.to(device)
        else:
            x = self.q_layer(x)

        # Scale and shift the output to break symmetry and allow larger values
        # Quantum output is in [-1, 1], but we need real-valued forecasts
        x = self.post_net(x.float()) 
        return x