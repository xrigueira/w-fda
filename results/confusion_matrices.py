import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

"""This file is used to generate the confusion
matrices"""

# Define statio number
station = 907

# Read the data for the corresponding station
y_indices = np.load(f'results/indices_y_real_outliers_{station}.npy', allow_pickle=False, fix_imports=False) # Ground truth
y_indices_msa = np.load(f'results/indices_y_msa_{station}.npy', allow_pickle=False, fix_imports=False) # MSA
y_indices_mout = np.load(f'results/outliers_MOUT_{station}.npy', allow_pickle=False, fix_imports=False) # MOUT
y_indices_muod = np.load(f'results/outliers_MUOD_{station}.npy', allow_pickle=False, fix_imports=False) # MUOD
y_rf = np.load(f'results/y_rf_{station}.npy', allow_pickle=False, fix_imports=False)

# Binarize the results from ground_truth, msa, mout, muod
y = np.zeros(len(y_rf))
y[y_indices] = 1

y_msa = np.zeros(len(y_rf))
y_msa[y_indices_msa] = 1

y_mout = np.zeros(len(y_rf))
y_mout[y_indices_mout] = 1

y_muod = np.zeros(len(y_rf))
y_muod[y_indices_muod] = 1

from sklearn.metrics import confusion_matrix

print('Confusion matrix MOUT:\n', confusion_matrix(y, y_mout))
print('Confusion matrix MUOD:\n', confusion_matrix(y, y_muod))
print('Confusion matrix MSA:\n', confusion_matrix(y, y_msa))
print('Confusion matrix RF:\n', confusion_matrix(y, y_rf))

# # Plots
# fig, ax = plt.subplots(figsize=(10, 9))

# # Create a plot with the specified colors and alpha values
# ax.plot(range(len(y)), y, color='blue', alpha=0.2, label='Ground truth label')
# ax.plot(range(len(y)), y_msa, color='red', alpha=0.2, label='Result MSA')
# ax.plot(range(len(y)), y_rf, color='green', alpha=0.2, label='Result RF')

# # Add a title and labels for the x-axis and y-axis
# plt.title('Classification resutls')
# plt.xlabel('4-hour group')
# plt.ylabel('Label')

# # Add a legend to the plot
# plt.legend()

# # Show the plot
# plt.show()
