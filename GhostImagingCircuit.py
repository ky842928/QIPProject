import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector

# --- Setting Parameter
theta = Parameter('θ')

# --- Building Ghost Imaging Demonstration Circuit (Two Qubit)
def build_circuit():
    qc = QuantumCircuit(2)

    # --- Entanglement/Prepping Bell State
    qc.h(0)
    qc.cx(0, 1)

    # --- Simulated Interaction with Object
    qc.rz(theta, 0)

    # --- Recovery/Inverse of Bell State Prep
    qc.cx(0, 1)
    qc.h(0)

    return qc

qc_param = build_circuit()

# --- Quantum Circuit Visualization
qc_param.draw(output='mpl', style='iqp')
plt.show()

# --- Bloch Sphere Visualization
state = Statevector.from_instruction(
    qc_param.assign_parameters({theta: np.pi/2})
)

plot_bloch_multivector(state)
plt.show()

# --- Let's go, Phase Sweep!
theta_vals = np.linspace(0, 2*np.pi, 50)
correlations = []

for t in theta_vals:
    qc = qc_param.assign_parameters({theta: t})
    state = Statevector.from_instruction(qc)

    probs = state.probabilities_dict()

    corr = (
        probs.get('00', 0)
        + probs.get('11', 0)
        - probs.get('01', 0)
        - probs.get('10', 0)
    )

    correlations.append(corr)

# --- Correlation Values Form Cosine Interference Pattern
plt.plot(theta_vals, correlations, marker='o')
plt.xlabel("Object Phase (θ)")

# --- Positive Correlation Signal Value = Constructive Interference
# --- Negative Correlation Signal Value = Destructive Interference
plt.ylabel("Correlation Signal")
plt.title("Ghost Imaging-Style Correlation Plot")
plt.grid()
plt.show()