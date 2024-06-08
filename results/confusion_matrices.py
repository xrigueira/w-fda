import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

"""This file is used to generate the confusion matrices for all methods."""

# Read the data for the corresponding station
y_gt = np.load(f'results/y_gt_fda.npy', allow_pickle=False, fix_imports=False) # Ground truth
y_msa = np.load(f'results/y_msa.npy', allow_pickle=False, fix_imports=False) # MSA
y_mout = np.load(f'results/y_mout.npy', allow_pickle=False, fix_imports=False) # MOUT
y_muod = np.load(f'results/y_muod.npy', allow_pickle=False, fix_imports=False) # MUOD
y_rf = np.load(f'results/y_rf.npy', allow_pickle=False, fix_imports=False) # RF
y_svm = np.load(f'results/y_svm.npy', allow_pickle=False, fix_imports=False) # SVM
y_lr = np.load(f'results/y_lr.npy', allow_pickle=False, fix_imports=False) # LR

# Create an empyt dataframe to store the accuracies, precisions, recalls, F1-scores and error_rates
import pandas as pd
data = {'Method': ['MOUT', 'MUOD', 'MSA', 'SVM', 'LR', 'RF'],
        'Accuracy': [0, 0, 0, 0, 0, 0],
        'Precision': [0, 0, 0, 0, 0, 0],
        'Recall': [0, 0, 0, 0, 0, 0],
        'F1-score': [0, 0, 0, 0, 0, 0],
        'Error rate': [0, 0, 0, 0, 0, 0]}

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

df.loc[2, 'Accuracy'] = accuracy_score(y_gt, y_msa)
df.loc[2, 'Precision'] = precision_score(y_gt, y_msa)
df.loc[2, 'Recall'] = recall_score(y_gt, y_msa)
df.loc[2, 'F1-score'] = f1_score(y_gt, y_msa)
df.loc[2, 'Error rate'] = 1 - accuracy_score(y_gt, y_msa)

df.loc[3, 'Accuracy'] = accuracy_score(y_gt, y_svm)
df.loc[3, 'Precision'] = precision_score(y_gt, y_svm)
df.loc[3, 'Recall'] = recall_score(y_gt, y_svm)
df.loc[3, 'F1-score'] = f1_score(y_gt, y_svm)
df.loc[3, 'Error rate'] = 1 - accuracy_score(y_gt, y_svm)

df.loc[4, 'Accuracy'] = accuracy_score(y_gt, y_lr)
df.loc[4, 'Precision'] = precision_score(y_gt, y_lr)
df.loc[4, 'Recall'] = recall_score(y_gt, y_lr)
df.loc[4, 'F1-score'] = f1_score(y_gt, y_lr)
df.loc[4, 'Error rate'] = 1 - accuracy_score(y_gt, y_lr)

df.loc[5, 'Accuracy'] = accuracy_score(y_gt, y_rf)
df.loc[5, 'Precision'] = precision_score(y_gt, y_rf)
df.loc[5, 'Recall'] = recall_score(y_gt, y_rf)
df.loc[5, 'F1-score'] = f1_score(y_gt, y_rf)
df.loc[5, 'Error rate'] = 1 - accuracy_score(y_gt, y_rf)

print(df)

from sklearn.metrics import confusion_matrix

print('Confusion matrix MOUT:\n', confusion_matrix(y_gt, y_mout))
print('Confusion matrix MUOD:\n', confusion_matrix(y_gt, y_muod))
print('Confusion matrix MSA:\n', confusion_matrix(y_gt, y_msa))
print('Confusion matrix SVM:\n', confusion_matrix(y_gt, y_svm))
print('Confusion matrix LR:\n', confusion_matrix(y_gt, y_lr))
print('Confusion matrix RF:\n', confusion_matrix(y_gt, y_rf))
