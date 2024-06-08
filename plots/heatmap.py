import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

"""This file is used to generate the heatmap to compare results
across the different methods."""

# Read the data and get the mean for each variable
station = 901
data = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8', parse_dates=['date'])

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data.iloc[:, 1:-2] = scaler.fit_transform(data.iloc[:, 1:-2])

# Get the timestamps
nhours = 8

# Create a new column 'time_block' to group dates into 4-hour intervals
data['time_block'] = data['date'].dt.floor(f'{nhours}H')

# Group the data by 'time_block' and apply some function (e.g., 'first') to turn it back into a DataFrame
time_stamps = data.groupby('time_block', sort=False)['week'].first().index

stats_dict = {}
var_names = ['am', 'co', 'do', 'ph', 'tu', 'wt']
for i, e in enumerate(data.iloc[:, 1:7]):
    stats_dict[var_names[i]] = data[e].mean()

# Read windowed data
X = np.load(f'results/X.npy', allow_pickle=True, fix_imports=True)

# Get the mean of each column of the items in X
X_mean = []
for window in X:
    X_mean.append(window.reshape(-1, 6).mean(axis=0))
X_mean = np.array(X_mean)

# Get the distance between the mean of each variable in the window and the mean of the variable in the dataset (stats_dict)
distances = []
for window in X_mean:
    distance = []
    for i, e in enumerate(window):
        distance.append(e - stats_dict[var_names[i]])
    distances.append(distance)
distances = np.array(distances)

# Calculate the mean difference between consecutive values in each column of the items in X
differences = []
for window in X:
    diff = np.diff(window.reshape(-1, 6), axis=0)
    differences.append(diff.mean(axis=0))
differences = np.array(differences)

# Get the indices where y_gt, y_msa, y_mout, y_muod, y_rf, y_svm and y_lr are different from 0
y_gt = np.load(f'results/y_gt_fda.npy', allow_pickle=False, fix_imports=False) # Ground truth
y_msa = np.load(f'results/y_msa.npy', allow_pickle=False, fix_imports=False) # MSA
y_mout = np.load(f'results/y_mout.npy', allow_pickle=False, fix_imports=False) # MOUT
y_muod = np.load(f'results/y_muod.npy', allow_pickle=False, fix_imports=False) # MUOD
y_rf = np.load(f'results/y_rf.npy', allow_pickle=False, fix_imports=False) # RF
y_svm = np.load(f'results/y_svm.npy', allow_pickle=False, fix_imports=False) # SVM
y_lr = np.load(f'results/y_lr.npy', allow_pickle=False, fix_imports=False) # LR

# Put the indices in an 2D array
ys = np.stack([y_gt, y_msa, y_mout, y_muod, y_rf, y_svm, y_lr], axis=1)

# Get the indices where any subelement is not 0 and turn it into a 1D array
indices = np.array(np.where(np.any(ys != 0, axis=1))[0])

# Get the distances for the indices
distances = distances[indices]

# Get the differences for the indices (not used)
differences = differences[indices]

# Get the y values for the indices
ys = ys[indices]

# Get the x labels for the indices
x_labels = time_stamps[indices]

max_index = 300
num_batches = len(distances) // max_index

for i in range(num_batches):
    start_index = i * max_index
    end_index = start_index + max_index

    distances_batch = distances[start_index:end_index]
    differences_batch = differences[start_index:end_index]
    ys_batch = ys[start_index:end_index]
    x_labels_batch = x_labels[start_index:end_index]

    # Define the labels for the method's names
    method_names = ['GT', 'MSA', 'MOUT', 'MUOD', 'RF', 'SVM', 'LR']

    # Reverse the order of the variables and the method names for plotting purposes
    var_names = var_names[::-1]
    method_names = method_names[::-1]
    distances_batch = np.fliplr(distances_batch)
    ys_batch = np.fliplr(ys_batch)

    # Convert x_labels to a sequence of integers
    x_labels_int = np.arange(len(x_labels_batch))

    # Create the heatmap with the distances and the heatmap with the y values
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    heatmap_1 = ax1.pcolormesh(x_labels_int, var_names, distances_batch.T, cmap='jet')
    heatmap_2 = ax2.pcolormesh(x_labels_int, method_names, ys_batch.T, cmap='jet')

    # Format x_labels to only display year and month
    x_labels_formatted = [date.strftime('%Y') for date in x_labels_batch]

    # Create new lists for the ticks and labels
    x_ticks = []
    x_ticklabels = []

    # Add the first label
    x_ticks.append(x_labels_int[0])
    x_ticklabels.append(x_labels_formatted[0])

    # Add a label only when there is a change every 2 months
    for j in range(1, len(x_labels_formatted)):
        if x_labels_formatted[j] != x_labels_formatted[j-1]:
            x_ticks.append(x_labels_int[j])
            x_ticklabels.append(x_labels_formatted[j])

    # Set the x-tick labels to the corresponding dates
    ax1.set_xticks([])
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_ticklabels, rotation=45)

    fig.colorbar(heatmap_1, ax=ax1)
    fig.colorbar(heatmap_2, ax=ax2)
    plt.show()

# # # Save the figure
# # # plt.savefig(f'plots/heatmap_{station}.png', dpi=300, bbox_inches='tight')
