from torch import nn
import torch
import pennylane as qml
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the quantum device
n_qubits = 4  # Adjust based on simulation speed (4-8 is usually fast)

# Hardware-inspired noise parameters for realistic NISQ simulation
# Based on typical quantum hardware characteristics:
# - IBM Quantum devices: 0.1-2% single-qubit, 1-5% two-qubit gate errors
# - Reference: Qiskit Aer noise model tutorials (qiskit.github.io/qiskit-aer/)
# - T1/T2 relaxation times: ~50-70μs typical for superconducting qubits
#
# The values below are calibrated to approximate mid-range NISQ device performance:
HARDWARE_NOISE_1Q = 0.015  # 1.5% single-qubit depolarizing error
HARDWARE_NOISE_2Q = 0.035  # 3.5% two-qubit depolarizing error
HARDWARE_NOISE_T1 = 0.01   # 1% amplitude damping (T1 relaxation)
SHOTS = 1024

def get_quantum_circuit(device_name="default.qubit"):
    if device_name == "qiskit.remote":
        from qiskit_ibm_runtime import QiskitRuntimeService

        IBM_API_TOKEN = os.environ.get("IBM_API_TOKEN")
        IBM_CRN = os.environ.get("IBM_CRN")

        if not IBM_API_TOKEN or not IBM_CRN:
            raise ValueError(
                "IBM_API_TOKEN and IBM_CRN environment variables must be set "
                "to use qiskit.remote device"
            )

        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=IBM_API_TOKEN,
            instance=IBM_CRN,
        )
        backend = service.least_busy(operational=True, simulator=False)
        dev = qml.device(
            'qiskit.remote',
            wires=n_qubits,
            backend=backend,
            optimization_level=3,
            resilience_level=0,
            seed_transpiler=42,
        )
    else:
        dev = qml.device(device_name, wires=n_qubits)
    noisy = (device_name == "default.mixed")

    #@qml.qnode(dev)
    @qml.qnode(dev, interface="torch", diff_method="best", shots=SHOTS)
    def quantum_projection_circuit(inputs, weights):
        # Define noise parameters based on simulation type
        if noisy:
            # Hardware-inspired noise: realistic error rates from NISQ devices
            depol_1q = HARDWARE_NOISE_1Q
            depol_2q = HARDWARE_NOISE_2Q
            amp_damp = HARDWARE_NOISE_T1
        else:
            # No noise (ideal simulation)
            depol_1q = depol_2q = amp_damp = 0.0

        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            feature = inputs[..., i]
            qml.RY(torch.arctan(feature), wires=i)
            qml.RZ(torch.arctan(feature**2), wires=i)
            if noisy:
                qml.DepolarizingChannel(depol_1q, wires=i)
                qml.AmplitudeDamping(amp_damp, wires=i)

        # 2. Variational Layer (Entanglement + Rotation)
        # weights shape: (n_layers, n_qubits, 3)
        n_layers = weights.shape[0] if len(weights.shape) > 2 else 1

        for l in range(n_layers):

            cnot_pairs = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (0, 2), (1, 3), (2, 0), (3, 1)
            ]

            for ctrl, tgt in cnot_pairs:
                qml.CNOT(wires=[ctrl, tgt])
                if noisy:
                    qml.DepolarizingChannel(depol_2q, wires=ctrl)
                    qml.DepolarizingChannel(depol_2q, wires=tgt)

            for i in range(n_qubits):
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
                if noisy:
                    qml.DepolarizingChannel(depol_1q, wires=i)
                    qml.AmplitudeDamping(amp_damp, wires=i)

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