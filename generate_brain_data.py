import numpy as np
import matplotlib.pyplot as plt

# Simulate synthetic brainwave signals
def generate_brain_data(n_samples=1000, n_channels=8, seed=42):
    np.random.seed(seed)
    time = np.linspace(0, 10, n_samples)
    data = []
    for ch in range(n_channels):
        alpha = np.sin(2 * np.pi * 10 * time)  # 10 Hz alpha waves
        beta = np.sin(2 * np.pi * 20 * time)   # 20 Hz beta waves
        noise = 0.3 * np.random.randn(n_samples)
        signal = alpha * np.random.uniform(0.5, 1.0) + beta * np.random.uniform(0.2, 0.6) + noise
        data.append(signal)
    return np.array(data), time

# Generate data
brain_data, time = generate_brain_data()

# Plot a sample of 3 channels
plt.figure(figsize=(10,5))
for i in range(3):
    plt.plot(time, brain_data[i] + i*3, label=f"Channel {i+1}")
plt.title("Synthetic Brainwave Signals")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
