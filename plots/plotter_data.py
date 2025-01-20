import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

"""This file is used to plot the univariate data
and display the anomalous curves."""

def single_plot(data, label, variable_index, color, ax):
    
    for window, label in zip(data, label): # zip(data[:1000], label[:1000])
        
        window_data = window.reshape(-1, 6)
    
        # Plot the first variable (column 0)
        if label == 1:
            ax.plot(window_data[:, variable_index], color=color, linewidth=1.5)
        else:
            ax.plot(window_data[:, variable_index], color='grey', alpha=0.32, linewidth=0.5)

# Define the station
station = 907

# Define the variables and methods
var_names = ['am', 'co', 'do', 'ph', 'tu', 'wt']
methods = ['Ground truth', 'MOUT', 'MUOD', 'MS', 'MMSA', 'SVM', 'LR', 'RF']

# Read data
data = np.load(f'results/X_{station}.npy')

# Read labels
y_gt = np.load(f'results/y_gt_ml_{station}.npy')
y_msa = np.load(f'results/y_msa_{station}.npy')
y_ms = np.load(f'results/y_ms_{station}.npy')
y_mout = np.load(f'results/y_mout_{station}.npy')
y_muod = np.load(f'results/y_muod_{station}.npy')
y_rf = np.load(f'results/y_rf_{station}.npy')
y_svm = np.load(f'results/y_svm_{station}.npy')
y_lr = np.load(f'results/y_lr_{station}.npy')

fig, axes = plt.subplots(nrows=6, ncols=8, sharex=True, figsize=(16, 6))

# Ground truth
single_plot(data, y_gt, 0, 'red', axes[0, 0])
single_plot(data, y_gt, 1, 'blue', axes[1, 0])
single_plot(data, y_gt, 2, 'purple', axes[2, 0])
single_plot(data, y_gt, 3, 'darkgray', axes[3, 0])
single_plot(data, y_gt, 4, 'goldenrod', axes[4, 0])
single_plot(data, y_gt, 5, 'green', axes[5, 0])

# MOUT
single_plot(data, y_mout, 0, 'red', axes[0, 1])
single_plot(data, y_mout, 1, 'blue', axes[1, 1])
single_plot(data, y_mout, 2, 'purple', axes[2, 1])
single_plot(data, y_mout, 3, 'darkgray', axes[3, 1])
single_plot(data, y_mout, 4, 'goldenrod', axes[4, 1])
single_plot(data, y_mout, 5, 'green', axes[5, 1])

# MUOD
single_plot(data, y_muod, 0, 'red', axes[0, 2])
single_plot(data, y_muod, 1, 'blue', axes[1, 2])
single_plot(data, y_muod, 2, 'purple', axes[2, 2])
single_plot(data, y_muod, 3, 'darkgray', axes[3, 2])
single_plot(data, y_muod, 4, 'goldenrod', axes[4, 2])
single_plot(data, y_muod, 5, 'green', axes[5, 2])

# MS
single_plot(data, y_ms, 0, 'red', axes[0, 3])
single_plot(data, y_ms, 1, 'blue', axes[1, 3])
single_plot(data, y_ms, 2, 'purple', axes[2, 3])
single_plot(data, y_ms, 3, 'darkgray', axes[3, 3])
single_plot(data, y_ms, 4, 'goldenrod', axes[4, 3])
single_plot(data, y_ms, 5, 'green', axes[5, 3])

# MMSA
single_plot(data, y_msa, 0, 'red', axes[0, 4])
single_plot(data, y_msa, 1, 'blue', axes[1, 4])
single_plot(data, y_msa, 2, 'purple', axes[2, 4])
single_plot(data, y_msa, 3, 'darkgray', axes[3, 4])
single_plot(data, y_msa, 4, 'goldenrod', axes[4, 4])
single_plot(data, y_msa, 5, 'green', axes[5, 4])

# SVM
single_plot(data, y_svm, 0, 'red', axes[0, 5])
single_plot(data, y_svm, 1, 'blue', axes[1, 5])
single_plot(data, y_svm, 2, 'purple', axes[2, 5])
single_plot(data, y_svm, 3, 'darkgray', axes[3, 5])
single_plot(data, y_svm, 4, 'goldenrod', axes[4, 5])
single_plot(data, y_svm, 5, 'green', axes[5, 5])

# LR
single_plot(data, y_lr, 0, 'red', axes[0, 6])
single_plot(data, y_lr, 1, 'blue', axes[1, 6])
single_plot(data, y_lr, 2, 'purple', axes[2, 6])
single_plot(data, y_lr, 3, 'darkgray', axes[3, 6])
single_plot(data, y_lr, 4, 'goldenrod', axes[4, 6])
single_plot(data, y_lr, 5, 'green', axes[5, 6])

# RF
single_plot(data, y_rf, 0, 'red', axes[0, 7])
single_plot(data, y_rf, 1, 'blue', axes[1, 7])
single_plot(data, y_rf, 2, 'purple', axes[2, 7])
single_plot(data, y_rf, 3, 'darkgray', axes[3, 7])
single_plot(data, y_rf, 4, 'goldenrod', axes[4, 7])
single_plot(data, y_rf, 5, 'green', axes[5, 7])

# Clean defaul y label and reduce font size for all axes
for ax in axes.flat:
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=6)

# Set the title for each model
for i, ax in enumerate(axes[0]):
    if methods[i] == 'Ground truth':
        ax.set_title(methods[i], fontname='Arial', fontsize=14, fontweight='bold')
    else:
        ax.set_title(methods[i], fontname='Arial', fontsize=14)

# Set the y label for each variable
for i, ax in enumerate(axes):
    ax[0].set_ylabel(var_names[i])

fig.suptitle(f'Anomalous time series for station {station}', fontname='Arial', fontsize=16)

# plt.show()

# Save the plot
plt.savefig(f'plots/time_series_{station}.pdf', format='pdf', dpi=300, bbox_inches='tight')