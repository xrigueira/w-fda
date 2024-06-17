import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

"""This file is used to generate the confusion matrices and calculate the
accuracy, precision, recall, F1-score, error rate and ROC-AUC curves
for each model."""

# Define the station
station = 907

# Read the data for the corresponding station
y_gt = np.load(f'results/y_gt_fda_{station}.npy', allow_pickle=False, fix_imports=False) # Ground truth
y_mmsa = np.load(f'results/y_msa_{station}.npy', allow_pickle=False, fix_imports=False) # MMSA
y_mout = np.load(f'results/y_mout_{station}.npy', allow_pickle=False, fix_imports=False) # MOUT
y_muod = np.load(f'results/y_muod_{station}.npy', allow_pickle=False, fix_imports=False) # MUOD
y_ms = np.load(f'results/y_ms_{station}.npy', allow_pickle=False, fix_imports=False) # MS
y_rf = np.load(f'results/y_rf_{station}.npy', allow_pickle=False, fix_imports=False) # RF
y_svm = np.load(f'results/y_svm_{station}.npy', allow_pickle=False, fix_imports=False) # SVM
y_lr = np.load(f'results/y_lr_{station}.npy', allow_pickle=False, fix_imports=False) # LR

# Create an empyt dataframe to store the accuracies, precisions, recalls, F1-scores and error_rates
import pandas as pd
data = {'Method': ['MOUT', 'MUOD', 'MS', 'MMSA', 'SVM', 'LR', 'RF'],
        'Accuracy': [0, 0, 0, 0, 0, 0, 0],
        'Precision': [0, 0, 0, 0, 0, 0, 0],
        'Recall': [0, 0, 0, 0, 0, 0, 0],
        'F1-score': [0, 0, 0, 0, 0, 0, 0],
        'Error rate': [0, 0, 0, 0, 0, 0, 0]}

df = pd.DataFrame(data)

# Pupulate the dataframe with the corresponding values
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df.loc[0, 'Accuracy'] = accuracy_score(y_gt, y_mout)
df.loc[0, 'Precision'] = precision_score(y_gt, y_mout)
df.loc[0, 'Recall'] = recall_score(y_gt, y_mout)
df.loc[0, 'F1-score'] = f1_score(y_gt, y_mout)
df.loc[0, 'Error rate'] = 1 - accuracy_score(y_gt, y_mout)

df.loc[1, 'Accuracy'] = accuracy_score(y_gt, y_muod)
df.loc[1, 'Precision'] = precision_score(y_gt, y_muod)
df.loc[1, 'Recall'] = recall_score(y_gt, y_muod)
df.loc[1, 'F1-score'] = f1_score(y_gt, y_muod)
df.loc[1, 'Error rate'] = 1 - accuracy_score(y_gt, y_muod)

df.loc[2, 'Accuracy'] = accuracy_score(y_gt, y_ms)
df.loc[2, 'Precision'] = precision_score(y_gt, y_ms)
df.loc[2, 'Recall'] = recall_score(y_gt, y_ms)
df.loc[2, 'F1-score'] = f1_score(y_gt, y_ms)
df.loc[2, 'Error rate'] = 1 - accuracy_score(y_gt, y_ms)

df.loc[3, 'Accuracy'] = accuracy_score(y_gt, y_mmsa)
df.loc[3, 'Precision'] = precision_score(y_gt, y_mmsa)
df.loc[3, 'Recall'] = recall_score(y_gt, y_mmsa)
df.loc[3, 'F1-score'] = f1_score(y_gt, y_mmsa)
df.loc[3, 'Error rate'] = 1 - accuracy_score(y_gt, y_mmsa)

df.loc[4, 'Accuracy'] = accuracy_score(y_gt, y_svm)
df.loc[4, 'Precision'] = precision_score(y_gt, y_svm)
df.loc[4, 'Recall'] = recall_score(y_gt, y_svm)
df.loc[4, 'F1-score'] = f1_score(y_gt, y_svm)
df.loc[4, 'Error rate'] = 1 - accuracy_score(y_gt, y_svm)

df.loc[5, 'Accuracy'] = accuracy_score(y_gt, y_lr)
df.loc[5, 'Precision'] = precision_score(y_gt, y_lr)
df.loc[5, 'Recall'] = recall_score(y_gt, y_lr)
df.loc[5, 'F1-score'] = f1_score(y_gt, y_lr)
df.loc[5, 'Error rate'] = 1 - accuracy_score(y_gt, y_lr)

df.loc[6, 'Accuracy'] = accuracy_score(y_gt, y_rf)
df.loc[6, 'Precision'] = precision_score(y_gt, y_rf)
df.loc[6, 'Recall'] = recall_score(y_gt, y_rf)
df.loc[6, 'F1-score'] = f1_score(y_gt, y_rf)
df.loc[6, 'Error rate'] = 1 - accuracy_score(y_gt, y_rf)

print(df)

from sklearn.metrics import confusion_matrix

print('Confusion matrix MOUT:\n', confusion_matrix(y_gt, y_mout))
print('Confusion matrix MUOD:\n', confusion_matrix(y_gt, y_muod))
print('Confusion matrix MS:\n', confusion_matrix(y_gt, y_ms))
print('Confusion matrix MSA:\n', confusion_matrix(y_gt, y_mmsa))
print('Confusion matrix SVM:\n', confusion_matrix(y_gt, y_svm))
print('Confusion matrix LR:\n', confusion_matrix(y_gt, y_lr))
print('Confusion matrix RF:\n', confusion_matrix(y_gt, y_rf))

# Plot the ROC-AUC curves
from sklearn.metrics import roc_curve, roc_auc_score

fpr_mout, tpr_mout, _ = roc_curve(y_gt, y_mout)
roc_auc_score_mout = roc_auc_score(y_gt, y_mout)
fpr_muod, tpr_muod, _ = roc_curve(y_gt, y_muod)
roc_auc_score_muod = roc_auc_score(y_gt, y_muod)
fpr_ms, tpr_ms, _ = roc_curve(y_gt, y_ms)
roc_auc_score_ms = roc_auc_score(y_gt, y_ms)
fpr_mmsa, tpr_mmsa, _ = roc_curve(y_gt, y_mmsa)
roc_auc_score_msa = roc_auc_score(y_gt, y_mmsa)
fpr_svm, tpr_svm, _ = roc_curve(y_gt, y_svm)
roc_auc_score_svm = roc_auc_score(y_gt, y_svm)
fpr_lr, tpr_lr, _ = roc_curve(y_gt, y_lr)
roc_auc_score_lr = roc_auc_score(y_gt, y_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_gt, y_rf)
roc_auc_score_rf = roc_auc_score(y_gt, y_rf)

# basic_colors = ['red', 'blue', 'yellow', 'green', 'purple', 'orange', 'gray']
# dark_colors = ['red', 'dodgerblue', 'mediumpurple', 'dimgrey', 'chocolate', 'goldenrod', 'green']
# light_colors = ['salmon', 'lightskyblue', 'plum', 'darkgrey', 'orange', 'gold', 'yellowgreen']

plt.figure(figsize=(10, 6))
plt.plot(fpr_mout, tpr_mout, color='salmon', label=f'MOUT: {roc_auc_score_mout:.2f}')
plt.plot(fpr_muod, tpr_muod, color='lightskyblue', label=f'MUOD: {roc_auc_score_muod:.2f}')
plt.plot(fpr_ms, tpr_ms, color='gold', label=f'MS: {roc_auc_score_ms:.2f}')
plt.plot(fpr_mmsa, tpr_mmsa, color='orange', label=f'MSA: {roc_auc_score_msa:.2f}')
plt.plot(fpr_svm, tpr_svm, color='gray', label=f'SVM: {roc_auc_score_svm:.2f}')
plt.plot(fpr_lr, tpr_lr, color='plum', label=f'LR: {roc_auc_score_lr:.2f}')
plt.plot(fpr_rf, tpr_rf, color='yellowgreen', label=f'RF: {roc_auc_score_rf:.2f}')
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title(f'ROC curve for station {station}')
plt.legend()

# plt.show()

# Save the image
plt.savefig(f'results/roc_{station}.png', dpi=300, bbox_inches='tight')
