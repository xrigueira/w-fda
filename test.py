# Read npt files and put in a single df

import numpy as np
import pandas as pd
msa_pro = np.load('msa_pro.npy') #2D with 3 columns
msa_shu = np.load('msa_shu.npy') #2D with 3 column
timestamps_pro = np.load('timestamps_pro.npy') # 1D
timestamps_shu = np.load('timestamps_shu.npy') # 1D

# Convert 1D timestamp arrays to 2D by stacking them vertically
timestamps_pro = timestamps_pro.reshape(-1, 1)
timestamps_shu = timestamps_shu.reshape(-1, 1)

# Horizontally stack the arrays
stacked_arrays = np.hstack((msa_pro, timestamps_pro, msa_shu, timestamps_shu))

df = pd.DataFrame(stacked_arrays)

df.to_csv('combined.csv', sep=',', encoding='utf-8', index=False)