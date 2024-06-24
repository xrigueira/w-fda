import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from skfda import datasets
from skfda import FDataGrid

# Load the weather dataset
X, y = datasets.fetch_weather(return_X_y=True, as_frame=True)
fd = X.iloc[:, 0].values
fd_temperatures = fd.coordinates[0]

# Calculate the average temperature along the first axis (axis=0)
mean_values = np.mean(fd_temperatures.data_matrix, axis=0)

# Create a new 3D numpy array with just one element
mean_values = mean_values.reshape(1, *mean_values.shape)

# Load the average values in a FDataGrid object.
fd_mean = FDataGrid(mean_values, fd_temperatures.grid_points,
                argument_names=['t'],
                coordinate_names=['x(t)'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

fd_temperatures.scatter(s=0.5, marker='.', edgecolor=None, axes=ax1) # Plot the measured values
# fd_mean.scatter(s=0.5, marker='.', color='black', label='Mean values', axes=ax1) # Plot the mean values
ax1.set_xlabel('t', fontsize=16)
ax1.set_ylabel('x(t)', fontsize=16)
ax1.set_title('Discrete values', fontsize=18)

fd_temperatures.plot(axes=ax2) # Plot functions
fd_mean.plot(color='black', label='Mean function', axes=ax2)
ax2.set_xlabel('t', fontsize=16)
ax2.set_ylabel('x(t)', fontsize=16)
ax2.set_title('Functional data', fontsize=18)
fig.suptitle('')
plt.legend()


plt.tight_layout()
# plt.show()

plt.savefig(f'plots/fda_sample.pdf', format='pdf', dpi=300, bbox_inches='tight')