import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Set random seed for reproducibility
np.random.seed(42)

# Generate 100 non-anomalous curves
def generate_normal_curve():
    x = np.linspace(0, 10, 200)  # Domain
    base_curve = np.sin(x)  # Base sinusoidal curve
    noise = np.random.normal(0, 0.2, size=x.shape)  # Add some noise
    return x, base_curve + noise

x, _ = generate_normal_curve()
normal_curves = [generate_normal_curve()[1] for _ in range(100)]

# Create a magnitude outlier (red curve)
magnitude_outlier = np.sin(x) + 3  # Shift the curve upward

# Create a shape outlier (blue curve)
shape_outlier = np.sin(2 * x)  # Change frequency to create a different shape

# Create an amplitude outlier (green curve)
amplitude_outlier = 3 * np.sin(x)  # Scale the amplitude

# Plot the curves
plt.figure(figsize=(10, 6))

# Plot non-anomalous curves in gray
for curve in normal_curves:
    plt.plot(x, curve, color="gray", alpha=0.5, linewidth=1.5)

# Plot outliers
plt.plot(x, magnitude_outlier, color="red", label="Magnitude outlier", linewidth=2)
plt.plot(x, shape_outlier, color="blue", label="Shape outlier", linewidth=2)
plt.plot(x, amplitude_outlier, color="green", label="Amplitude outlier", linewidth=2)

# Add labels and legend
plt.title("Functional data with outliers")
plt.xlabel("Domain")
plt.ylabel("Value")
plt.legend()
# plt.grid(True)

# plt.show()

plt.savefig('plots/fda_sample.pdf', format='pdf', dpi=300, bbox_inches='tight')
