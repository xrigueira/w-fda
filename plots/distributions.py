import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

"""This file plots the data distribution (using kde) for each model and variable,
when the their prediction is 1 (anomaly) or 0 (normal)."""

# Define the station
station = 901

# Read windowed data
X = np.load(f'results/X_{station}.npy', allow_pickle=True, fix_imports=True)

# Get the ground truth and results of each model
y_gt = np.load(f'results/y_gt_fda_{station}.npy', allow_pickle=False, fix_imports=False) # Ground truth
y_mout = np.load(f'results/y_mout_{station}.npy', allow_pickle=False, fix_imports=False) # MOUT
y_muod = np.load(f'results/y_muod_{station}.npy', allow_pickle=False, fix_imports=False) # MUOD
y_ms = np.load(f'results/y_ms_{station}.npy', allow_pickle=False, fix_imports=False) # MS
y_msa = np.load(f'results/y_msa_{station}.npy', allow_pickle=False, fix_imports=False) # MMSA
y_svm = np.load(f'results/y_svm_{station}.npy', allow_pickle=False, fix_imports=False) # SVM
y_lr = np.load(f'results/y_lr_{station}.npy', allow_pickle=False, fix_imports=False) # LR
y_rf = np.load(f'results/y_rf_{station}.npy', allow_pickle=False, fix_imports=False) # RF

# Reshape X for easier access to each variables via columns
X_reshaped = []
for window in X:
    X_reshaped.append(window.reshape(-1, 6))
X_reshaped = np.array(X_reshaped)

# Get the indices where the ground truth and each model's predictions is 0 and 1
y_gt_0, y_gt_1 = np.where(y_gt == 0)[0], np.where(y_gt == 1)[0]
y_mout_0, y_mout_1 = np.where(y_mout == 0)[0], np.where(y_mout == 1)[0]
y_muod_0, y_muod_1 = np.where(y_muod == 0)[0], np.where(y_muod == 1)[0]
y_ms_0, y_ms_1 = np.where(y_ms == 0)[0], np.where(y_ms == 1)[0]
y_msa_0, y_msa_1 = np.where(y_msa == 0)[0], np.where(y_msa == 1)[0]
y_svm_0, y_svm_1 = np.where(y_svm == 0)[0], np.where(y_svm == 1)[0]
y_lr_0, y_lr_1 = np.where(y_lr == 0)[0], np.where(y_lr == 1)[0]

# Get the values of each variable when the ground truth and each model's prediction is 0 or 1
var_names = ['am', 'co', 'do', 'ph', 'tu', 'wt']

# Ground truth
am_gt_0, am_gt_1 = X_reshaped[y_gt_0, :, 0].flatten(), X_reshaped[y_gt_1, :, 0].flatten()
co_gt_0, co_gt_1 = X_reshaped[y_gt_0, :, 1].flatten(), X_reshaped[y_gt_1, :, 1].flatten()
do_gt_0, do_gt_1 = X_reshaped[y_gt_0, :, 2].flatten(), X_reshaped[y_gt_1, :, 2].flatten()
ph_gt_0, ph_gt_1 = X_reshaped[y_gt_0, :, 3].flatten(), X_reshaped[y_gt_1, :, 3].flatten()
tu_gt_0, tu_gt_1 = X_reshaped[y_gt_0, :, 4].flatten(), X_reshaped[y_gt_1, :, 4].flatten()
wt_gt_0, wt_gt_1 = X_reshaped[y_gt_0, :, 5].flatten(), X_reshaped[y_gt_1, :, 5].flatten()

# MUOD

# MOUT

# MS

# MMSA

# SVM

# LR

# RF


# Plot the distribution of the first variable when y_gt is 0
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
sns.kdeplot(am_gt_0, bw_adjust=0.65, fill=True, color='blue', ax=ax)
plt.ylabel('')
plt.show()
