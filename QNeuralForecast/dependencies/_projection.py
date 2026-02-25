from torch import nn
import torch
import pennylane as qml

# Define the quantum device
n_qubits = 4  # Adjust based on simulation speed (4-8 is usually fast)
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_projection_circuit(inputs, weights):
    # 1. Data Encoding: Map classical features to quantum state
    # We assume inputs are pre-processed to match n_qubits
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # 2. Variational Layers: Trainable quantum gates
    # weights shape: (n_layers, n_qubits, 3) for StronglyEntangling
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # 3. Measurement: Return expectation values for each qubit
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


class QuantumProjection(nn.Module):
    def __init__(self, input_size, output_size, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        
        # 1. Compress input to fit on quantum chip
        self.pre_net = nn.Linear(input_size, n_qubits)
        
        # 2. Quantum Layer
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_projection_circuit, weight_shapes)
        
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