import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit import transpile

# --- Settings For Normal Hardware (Keep nx/ny small!)
nx, ny = 3, 3
n_pairs = nx * ny
shots = 2000
grover_steps = 1

thetas = ParameterVector('θ', n_pairs)

# --- Object Interaction
def generate_object(nx, ny):
    img = np.zeros((nx, ny))

    for x in range(nx):
        for y in range(ny):

            # --- Normalize Coordinates
            X = x / (nx - 1)
            Y = y / (ny - 1)

            # --- Make L shape ---
            vertical_bar = X < 0.25
            horizontal_bar = Y > 0.75

            if vertical_bar or horizontal_bar:
                img[x, y] = np.pi / 2
            else:
                img[x, y] = 0

    return img

# --- Building Ghost Imaging Grover's Algorithm Circuit
def build_circuit(n_pairs, grover_steps):
    qc = QuantumCircuit(2 * n_pairs, 2 * n_pairs)

    for i in range(n_pairs):
        q0, q1 = 2*i, 2*i + 1

        # --- Initialize Bell State
        qc.h(q0)
        qc.cx(q0, q1)

        # --- Object Interaction
        qc.rz(thetas[i], q0)

        # --- Interference Readout
        qc.cx(q0, q1)
        qc.h(q0)

        # --- Local Amplification
        for _ in range(grover_steps):
            qc.cz(q0, q1)
            qc.x([q0, q1])
            qc.cz(q0, q1)
            qc.x([q0, q1])

            qc.h([q0, q1])
            qc.x([q0, q1])

            qc.h(q1)
            qc.cx(q0, q1)
            qc.h(q1)

            qc.x([q0, q1])
            qc.h([q0, q1])

    # --- Measurement Time!
    qc.measure(range(2*n_pairs), range(2*n_pairs))

    return qc

# --- Correlation Image Definition
def reconstruct_image(probs, n_pairs):
    pixels = np.zeros(n_pairs)

    for bitstring, p in probs.items():

        # --- DO NOT REVERSE
        for i in range(n_pairs):
            b0 = int(bitstring[2*n_pairs - 2*i - 2])
            b1 = int(bitstring[2*n_pairs - 2*i - 1])

            parity = 1 if (b0 + b1) % 2 == 0 else -1
            pixels[i] += parity * p

    return pixels

# --- Building Image
image = generate_object(nx, ny)
theta_vals = image.flatten()

qc_param = build_circuit(n_pairs, grover_steps)

# --- Visualizing Circuit
qc_param.draw(output='mpl', style='iqp')
plt.title("Ghost Imaging Circuit (Grover's Version)")
plt.show()

# --- Parameters
param_map = {thetas[i]: theta_vals[i] for i in range(n_pairs)}
qc = qc_param.assign_parameters(param_map)

# --- Activating Simulator
sim = AerSimulator()
compiled = transpile(qc, sim)

result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# --- Reconstructing Image
recon = reconstruct_image(counts, n_pairs)
recon_img = recon.reshape(nx, ny)

# --- Plot of Imaging Done
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(image, cmap='viridis')
plt.title("Original Object")
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(recon_img, cmap='viridis')
plt.title("Reconstructed (k=1)")
plt.colorbar()

plt.tight_layout()
plt.show()