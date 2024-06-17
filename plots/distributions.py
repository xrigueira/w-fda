import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

"""This file plots the data distribution (using kde) for each model and variable,
when the their prediction is 1 (anomaly) or 0 (normal)."""

# Define the station
station = 907

# Read windowed data
X = np.load(f'results/X_{station}.npy', allow_pickle=True, fix_imports=True)

# Get the ground truth and results of each model
y_gt = np.load(f'results/y_gt_fda_{station}.npy', allow_pickle=False, fix_imports=False) # Ground truth
y_mout = np.load(f'results/y_mout_{station}.npy', allow_pickle=False, fix_imports=False) # MOUT
y_muod = np.load(f'results/y_muod_{station}.npy', allow_pickle=False, fix_imports=False) # MUOD
y_ms = np.load(f'results/y_ms_{station}.npy', allow_pickle=False, fix_imports=False) # MS
y_mmsa = np.load(f'results/y_msa_{station}.npy', allow_pickle=False, fix_imports=False) # MMSA
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
y_msa_0, y_msa_1 = np.where(y_mmsa == 0)[0], np.where(y_mmsa == 1)[0]
y_svm_0, y_svm_1 = np.where(y_svm == 0)[0], np.where(y_svm == 1)[0]
y_lr_0, y_lr_1 = np.where(y_lr == 0)[0], np.where(y_lr == 1)[0]
y_rf_0, y_rf_1 = np.where(y_rf == 0)[0], np.where(y_rf == 1)[0]

# Get the values of each variable when the ground truth and each model's prediction is 0 or 1
var_names = ['am', 'co', 'do', 'ph', 'tu', 'wt']
methods = ['gt', 'mout', 'muod', 'ms', 'mmsa', 'svm', 'lr', 'rf']

# Ground truth
am_gt_0, am_gt_1 = X_reshaped[y_gt_0, :, 0].flatten(), X_reshaped[y_gt_1, :, 0].flatten()
co_gt_0, co_gt_1 = X_reshaped[y_gt_0, :, 1].flatten(), X_reshaped[y_gt_1, :, 1].flatten()
do_gt_0, do_gt_1 = X_reshaped[y_gt_0, :, 2].flatten(), X_reshaped[y_gt_1, :, 2].flatten()
ph_gt_0, ph_gt_1 = X_reshaped[y_gt_0, :, 3].flatten(), X_reshaped[y_gt_1, :, 3].flatten()
tu_gt_0, tu_gt_1 = X_reshaped[y_gt_0, :, 4].flatten(), X_reshaped[y_gt_1, :, 4].flatten()
wt_gt_0, wt_gt_1 = X_reshaped[y_gt_0, :, 5].flatten(), X_reshaped[y_gt_1, :, 5].flatten()

# MUOD
am_muod_0, am_muod_1 = X_reshaped[y_muod_0, :, 0].flatten(), X_reshaped[y_muod_1, :, 0].flatten()
co_muod_0, co_muod_1 = X_reshaped[y_muod_0, :, 1].flatten(), X_reshaped[y_muod_1, :, 1].flatten()
do_muod_0, do_muod_1 = X_reshaped[y_muod_0, :, 2].flatten(), X_reshaped[y_muod_1, :, 2].flatten()
ph_muod_0, ph_muod_1 = X_reshaped[y_muod_0, :, 3].flatten(), X_reshaped[y_muod_1, :, 3].flatten()
tu_muod_0, tu_muod_1 = X_reshaped[y_muod_0, :, 4].flatten(), X_reshaped[y_muod_1, :, 4].flatten()
wt_muod_0, wt_muod_1 = X_reshaped[y_muod_0, :, 5].flatten(), X_reshaped[y_muod_1, :, 5].flatten()

# MOUT
am_mout_0, am_mout_1 = X_reshaped[y_mout_0, :, 0].flatten(), X_reshaped[y_mout_1, :, 0].flatten()
co_mout_0, co_mout_1 = X_reshaped[y_mout_0, :, 1].flatten(), X_reshaped[y_mout_1, :, 1].flatten()
do_mout_0, do_mout_1 = X_reshaped[y_mout_0, :, 2].flatten(), X_reshaped[y_mout_1, :, 2].flatten()
ph_mout_0, ph_mout_1 = X_reshaped[y_mout_0, :, 3].flatten(), X_reshaped[y_mout_1, :, 3].flatten()
tu_mout_0, tu_mout_1 = X_reshaped[y_mout_0, :, 4].flatten(), X_reshaped[y_mout_1, :, 4].flatten()
wt_mout_0, wt_mout_1 = X_reshaped[y_mout_0, :, 5].flatten(), X_reshaped[y_mout_1, :, 5].flatten()

# MS
am_ms_0, am_ms_1 = X_reshaped[y_ms_0, :, 0].flatten(), X_reshaped[y_ms_1, :, 0].flatten()
co_ms_0, co_ms_1 = X_reshaped[y_ms_0, :, 1].flatten(), X_reshaped[y_ms_1, :, 1].flatten()
do_ms_0, do_ms_1 = X_reshaped[y_ms_0, :, 2].flatten(), X_reshaped[y_ms_1, :, 2].flatten()
ph_ms_0, ph_ms_1 = X_reshaped[y_ms_0, :, 3].flatten(), X_reshaped[y_ms_1, :, 3].flatten()
tu_ms_0, tu_ms_1 = X_reshaped[y_ms_0, :, 4].flatten(), X_reshaped[y_ms_1, :, 4].flatten()
wt_ms_0, wt_ms_1 = X_reshaped[y_ms_0, :, 5].flatten(), X_reshaped[y_ms_1, :, 5].flatten()

# MMSA
am_mmsa_0, am_mmsa_1 = X_reshaped[y_msa_0, :, 0].flatten(), X_reshaped[y_msa_1, :, 0].flatten()
co_mmsa_0, co_mmsa_1 = X_reshaped[y_msa_0, :, 1].flatten(), X_reshaped[y_msa_1, :, 1].flatten()
do_mmsa_0, do_mmsa_1 = X_reshaped[y_msa_0, :, 2].flatten(), X_reshaped[y_msa_1, :, 2].flatten()
ph_mmsa_0, ph_mmsa_1 = X_reshaped[y_msa_0, :, 3].flatten(), X_reshaped[y_msa_1, :, 3].flatten()
tu_mmsa_0, tu_mmsa_1 = X_reshaped[y_msa_0, :, 4].flatten(), X_reshaped[y_msa_1, :, 4].flatten()
wt_mmsa_0, wt_mmsa_1 = X_reshaped[y_msa_0, :, 5].flatten(), X_reshaped[y_msa_1, :, 5].flatten()

# SVM
am_svm_0, am_svm_1 = X_reshaped[y_svm_0, :, 0].flatten(), X_reshaped[y_svm_1, :, 0].flatten()
co_svm_0, co_svm_1 = X_reshaped[y_svm_0, :, 1].flatten(), X_reshaped[y_svm_1, :, 1].flatten()
do_svm_0, do_svm_1 = X_reshaped[y_svm_0, :, 2].flatten(), X_reshaped[y_svm_1, :, 2].flatten()
ph_svm_0, ph_svm_1 = X_reshaped[y_svm_0, :, 3].flatten(), X_reshaped[y_svm_1, :, 3].flatten()
tu_svm_0, tu_svm_1 = X_reshaped[y_svm_0, :, 4].flatten(), X_reshaped[y_svm_1, :, 4].flatten()
wt_svm_0, wt_svm_1 = X_reshaped[y_svm_0, :, 5].flatten(), X_reshaped[y_svm_1, :, 5].flatten()

# LR
am_lr_0, am_lr_1 = X_reshaped[y_lr_0, :, 0].flatten(), X_reshaped[y_lr_1, :, 0].flatten()
co_lr_0, co_lr_1 = X_reshaped[y_lr_0, :, 1].flatten(), X_reshaped[y_lr_1, :, 1].flatten()
do_lr_0, do_lr_1 = X_reshaped[y_lr_0, :, 2].flatten(), X_reshaped[y_lr_1, :, 2].flatten()
ph_lr_0, ph_lr_1 = X_reshaped[y_lr_0, :, 3].flatten(), X_reshaped[y_lr_1, :, 3].flatten()
tu_lr_0, tu_lr_1 = X_reshaped[y_lr_0, :, 4].flatten(), X_reshaped[y_lr_1, :, 4].flatten()
wt_lr_0, wt_lr_1 = X_reshaped[y_lr_0, :, 5].flatten(), X_reshaped[y_lr_1, :, 5].flatten()

# RF
am_rf_0, am_rf_1 = X_reshaped[y_rf_0, :, 0].flatten(), X_reshaped[y_rf_1, :, 0].flatten()
co_rf_0, co_rf_1 = X_reshaped[y_rf_0, :, 1].flatten(), X_reshaped[y_rf_1, :, 1].flatten()
do_rf_0, do_rf_1 = X_reshaped[y_rf_0, :, 2].flatten(), X_reshaped[y_rf_1, :, 2].flatten()
ph_rf_0, ph_rf_1 = X_reshaped[y_rf_0, :, 3].flatten(), X_reshaped[y_rf_1, :, 3].flatten()
tu_rf_0, tu_rf_1 = X_reshaped[y_rf_0, :, 4].flatten(), X_reshaped[y_rf_1, :, 4].flatten()
wt_rf_0, wt_rf_1 = X_reshaped[y_rf_0, :, 5].flatten(), X_reshaped[y_rf_1, :, 5].flatten()

# Plot the distribution of the each variable for the ground truth and each model's prediction
fig, axes = plt.subplots(nrows=6, ncols=8, sharex=True, figsize=(16, 6))

# Ammonium
sns.kdeplot(am_gt_0, bw_adjust=0.65, fill=True, color='lightcoral', ax=axes[0, 0])
sns.kdeplot(am_gt_1, bw_adjust=0.65, fill=True, color='red', ax=axes[0, 0])
sns.kdeplot(am_mout_0, bw_adjust=0.65, fill=True, color='lightcoral', ax=axes[0, 1])
sns.kdeplot(am_mout_1, bw_adjust=0.65, fill=True, color='red', ax=axes[0, 1])
sns.kdeplot(am_muod_0, bw_adjust=0.65, fill=True, color='lightcoral', ax=axes[0, 2])
sns.kdeplot(am_muod_1, bw_adjust=0.65, fill=True, color='red', ax=axes[0, 2])
sns.kdeplot(am_ms_0, bw_adjust=0.65, fill=True, color='lightcoral', ax=axes[0, 3])
sns.kdeplot(am_ms_1, bw_adjust=0.65, fill=True, color='red', ax=axes[0, 3])
sns.kdeplot(am_mmsa_0, bw_adjust=0.65, fill=True, color='lightcoral', ax=axes[0, 4])
sns.kdeplot(am_mmsa_1, bw_adjust=0.65, fill=True, color='red', ax=axes[0, 4])
sns.kdeplot(am_svm_0, bw_adjust=0.65, fill=True, color='lightcoral', ax=axes[0, 5])
sns.kdeplot(am_svm_1, bw_adjust=0.65, fill=True, color='red', ax=axes[0, 5])
sns.kdeplot(am_lr_0, bw_adjust=0.65, fill=True, color='lightcoral', ax=axes[0, 6])
sns.kdeplot(am_lr_1, bw_adjust=0.65, fill=True, color='red', ax=axes[0, 6])
sns.kdeplot(am_rf_0, bw_adjust=0.65, fill=True, color='lightcoral', ax=axes[0, 7])
sns.kdeplot(am_rf_1, bw_adjust=0.65, fill=True, color='red', ax=axes[0, 7])

# Conductivity
sns.kdeplot(co_gt_0, bw_adjust=0.65, fill=True, color='cornflowerblue', ax=axes[1, 0])
sns.kdeplot(co_gt_1, bw_adjust=0.65, fill=True, color='blue', ax=axes[1, 0])
sns.kdeplot(co_mout_0, bw_adjust=0.65, fill=True, color='cornflowerblue', ax=axes[1, 1])
sns.kdeplot(co_mout_1, bw_adjust=0.65, fill=True, color='blue', ax=axes[1, 1])
sns.kdeplot(co_muod_0, bw_adjust=0.65, fill=True, color='cornflowerblue', ax=axes[1, 2])
sns.kdeplot(co_muod_1, bw_adjust=0.65, fill=True, color='blue', ax=axes[1, 2])
sns.kdeplot(co_ms_0, bw_adjust=0.65, fill=True, color='cornflowerblue', ax=axes[1, 3])
sns.kdeplot(co_ms_1, bw_adjust=0.65, fill=True, color='blue', ax=axes[1, 3])
sns.kdeplot(co_mmsa_0, bw_adjust=0.65, fill=True, color='cornflowerblue', ax=axes[1, 4])
sns.kdeplot(co_mmsa_1, bw_adjust=0.65, fill=True, color='blue', ax=axes[1, 4])
sns.kdeplot(co_svm_0, bw_adjust=0.65, fill=True, color='cornflowerblue', ax=axes[1, 5])
sns.kdeplot(co_svm_1, bw_adjust=0.65, fill=True, color='blue', ax=axes[1, 5])
sns.kdeplot(co_lr_0, bw_adjust=0.65, fill=True, color='cornflowerblue', ax=axes[1, 6])
sns.kdeplot(co_lr_1, bw_adjust=0.65, fill=True, color='blue', ax=axes[1, 6])
sns.kdeplot(co_rf_0, bw_adjust=0.65, fill=True, color='cornflowerblue', ax=axes[1, 7])
sns.kdeplot(co_rf_1, bw_adjust=0.65, fill=True, color='blue', ax=axes[1, 7])

# Dissolved oxygen
sns.kdeplot(do_gt_0, bw_adjust=0.65, fill=True, color='mediumpurple', ax=axes[2, 0])
sns.kdeplot(do_gt_1, bw_adjust=0.65, fill=True, color='purple', ax=axes[2, 0])
sns.kdeplot(do_mout_0, bw_adjust=0.65, fill=True, color='mediumpurple', ax=axes[2, 1])
sns.kdeplot(do_mout_1, bw_adjust=0.65, fill=True, color='purple', ax=axes[2, 1])
sns.kdeplot(do_muod_0, bw_adjust=0.65, fill=True, color='mediumpurple', ax=axes[2, 2])
sns.kdeplot(do_muod_1, bw_adjust=0.65, fill=True, color='purple', ax=axes[2, 2])
sns.kdeplot(do_ms_0, bw_adjust=0.65, fill=True, color='mediumpurple', ax=axes[2, 3])
sns.kdeplot(do_ms_1, bw_adjust=0.65, fill=True, color='purple', ax=axes[2, 3])
sns.kdeplot(do_mmsa_0, bw_adjust=0.65, fill=True, color='mediumpurple', ax=axes[2, 4])
sns.kdeplot(do_mmsa_1, bw_adjust=0.65, fill=True, color='purple', ax=axes[2, 4])
sns.kdeplot(do_svm_0, bw_adjust=0.65, fill=True, color='mediumpurple', ax=axes[2, 5])
sns.kdeplot(do_svm_1, bw_adjust=0.65, fill=True, color='purple', ax=axes[2, 5])
sns.kdeplot(do_lr_0, bw_adjust=0.65, fill=True, color='mediumpurple', ax=axes[2, 6])
sns.kdeplot(do_lr_1, bw_adjust=0.65, fill=True, color='purple', ax=axes[2, 6])
sns.kdeplot(do_rf_0, bw_adjust=0.65, fill=True, color='mediumpurple', ax=axes[2, 7])
sns.kdeplot(do_rf_1, bw_adjust=0.65, fill=True, color='purple', ax=axes[2, 7])

# pH
sns.kdeplot(ph_gt_0, bw_adjust=0.65, fill=True, color='dimgray', ax=axes[3, 0])
sns.kdeplot(ph_gt_1, bw_adjust=0.65, fill=True, color='darkgray', ax=axes[3, 0])
sns.kdeplot(ph_mout_0, bw_adjust=0.65, fill=True, color='dimgray', ax=axes[3, 1])
sns.kdeplot(ph_mout_1, bw_adjust=0.65, fill=True, color='darkgray', ax=axes[3, 1])
sns.kdeplot(ph_muod_0, bw_adjust=0.65, fill=True, color='dimgray', ax=axes[3, 2])
sns.kdeplot(ph_muod_1, bw_adjust=0.65, fill=True, color='darkgray', ax=axes[3, 2])
sns.kdeplot(ph_ms_0, bw_adjust=0.65, fill=True, color='dimgray', ax=axes[3, 3])
sns.kdeplot(ph_ms_1, bw_adjust=0.65, fill=True, color='darkgray', ax=axes[3, 3])
sns.kdeplot(ph_mmsa_0, bw_adjust=0.65, fill=True, color='dimgray', ax=axes[3, 4])
sns.kdeplot(ph_mmsa_1, bw_adjust=0.65, fill=True, color='darkgray', ax=axes[3, 4])
sns.kdeplot(ph_svm_0, bw_adjust=0.65, fill=True, color='dimgray', ax=axes[3, 5])
sns.kdeplot(ph_svm_1, bw_adjust=0.65, fill=True, color='darkgray', ax=axes[3, 5])
sns.kdeplot(ph_lr_0, bw_adjust=0.65, fill=True, color='dimgray', ax=axes[3, 6])
sns.kdeplot(ph_lr_1, bw_adjust=0.65, fill=True, color='darkgray', ax=axes[3, 6])
sns.kdeplot(ph_rf_0, bw_adjust=0.65, fill=True, color='dimgray', ax=axes[3, 7])
sns.kdeplot(ph_rf_1, bw_adjust=0.65, fill=True, color='darkgray', ax=axes[3, 7])

# Turbidity
sns.kdeplot(tu_gt_0, bw_adjust=0.65, fill=True, color='gold', ax=axes[4, 0])
sns.kdeplot(tu_gt_1, bw_adjust=0.65, fill=True, color='goldenrod', ax=axes[4, 0])
sns.kdeplot(tu_mout_0, bw_adjust=0.65, fill=True, color='gold', ax=axes[4, 1])
sns.kdeplot(tu_mout_1, bw_adjust=0.65, fill=True, color='goldenrod', ax=axes[4, 1])
sns.kdeplot(tu_muod_0, bw_adjust=0.65, fill=True, color='gold', ax=axes[4, 2])
sns.kdeplot(tu_muod_1, bw_adjust=0.65, fill=True, color='goldenrod', ax=axes[4, 2])
sns.kdeplot(tu_ms_0, bw_adjust=0.65, fill=True, color='gold', ax=axes[4, 3])
sns.kdeplot(tu_ms_1, bw_adjust=0.65, fill=True, color='goldenrod', ax=axes[4, 3])
sns.kdeplot(tu_mmsa_0, bw_adjust=0.65, fill=True, color='gold', ax=axes[4, 4])
sns.kdeplot(tu_mmsa_1, bw_adjust=0.65, fill=True, color='goldenrod', ax=axes[4, 4])
sns.kdeplot(tu_svm_0, bw_adjust=0.65, fill=True, color='gold', ax=axes[4, 5])
sns.kdeplot(tu_svm_1, bw_adjust=0.65, fill=True, color='goldenrod', ax=axes[4, 5])
sns.kdeplot(tu_lr_0, bw_adjust=0.65, fill=True, color='gold', ax=axes[4, 6])
sns.kdeplot(tu_lr_1, bw_adjust=0.65, fill=True, color='goldenrod', ax=axes[4, 6])
sns.kdeplot(tu_rf_0, bw_adjust=0.65, fill=True, color='gold', ax=axes[4, 7])
sns.kdeplot(tu_rf_1, bw_adjust=0.65, fill=True, color='goldenrod', ax=axes[4, 7])

# Water temperature
sns.kdeplot(wt_gt_0, bw_adjust=0.65, fill=True, color='limegreen', ax=axes[5, 0])
sns.kdeplot(wt_gt_1, bw_adjust=0.65, fill=True, color='green', ax=axes[5, 0])
sns.kdeplot(wt_mout_0, bw_adjust=0.65, fill=True, color='limegreen', ax=axes[5, 1])
sns.kdeplot(wt_mout_1, bw_adjust=0.65, fill=True, color='green', ax=axes[5, 1])
sns.kdeplot(wt_muod_0, bw_adjust=0.65, fill=True, color='limegreen', ax=axes[5, 2])
sns.kdeplot(wt_muod_1, bw_adjust=0.65, fill=True, color='green', ax=axes[5, 2])
sns.kdeplot(wt_ms_0, bw_adjust=0.65, fill=True, color='limegreen', ax=axes[5, 3])
sns.kdeplot(wt_ms_1, bw_adjust=0.65, fill=True, color='green', ax=axes[5, 3])
sns.kdeplot(wt_mmsa_0, bw_adjust=0.65, fill=True, color='limegreen', ax=axes[5, 4])
sns.kdeplot(wt_mmsa_1, bw_adjust=0.65, fill=True, color='green', ax=axes[5, 4])
sns.kdeplot(wt_svm_0, bw_adjust=0.65, fill=True, color='limegreen', ax=axes[5, 5])
sns.kdeplot(wt_svm_1, bw_adjust=0.65, fill=True, color='green', ax=axes[5, 5])
sns.kdeplot(wt_lr_0, bw_adjust=0.65, fill=True, color='limegreen', ax=axes[5, 6])
sns.kdeplot(wt_lr_1, bw_adjust=0.65, fill=True, color='green', ax=axes[5, 6])
sns.kdeplot(wt_rf_0, bw_adjust=0.65, fill=True, color='limegreen', ax=axes[5, 7])
sns.kdeplot(wt_rf_1, bw_adjust=0.65, fill=True, color='green', ax=axes[5, 7])

# Clean defaul y label and reduce font size for all axes
for ax in axes.flat:
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=6)

# Set the title for each model
for i, ax in enumerate(axes[0]):
    ax.set_title(methods[i])

# Set the y label for each variable
for i, ax in enumerate(axes):
    ax[0].set_ylabel(var_names[i])

fig.suptitle(f'Data distribution of ground truth and model predictions for station {station}', fontsize=16)

# plt.show()

# Save the plot
plt.savefig(f'plots/distributions_{station}.pdf', format="pdf", dpi=300, bbox_inches='tight')
