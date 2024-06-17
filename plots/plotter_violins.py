import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

"""This file is used to generate the violin-
style plots for the data of each variable."""

# # Read the data
# station = 901
# data = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8', parse_dates=['date'])

# # Set the 'date' column as the DataFrame's index
# data.set_index('date', inplace=True)

# # Drop not needed columns
# data = data.drop(data.columns[6:], axis=1)

# # Normalize the data
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# data.iloc[:, :] = scaler.fit_transform(data.iloc[:, :])


# fig, ax = plt.subplots(figsize=(12, 8))
# # fig.subplots_adjust(bottom)
# sns.set(style="whitegrid")
# sns.violinplot(data=data)

# names = ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature']
# plt.xticks(range(len(names)), names, fontsize=16)
# plt.ylabel("Nomalized values", fontsize=16)
# plt.title("Violin plot of each variable", fontsize=18)
# plt.show()

# fig.savefig(f'violins.png', format='png', dpi=300)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Read the data and select those columns with the desired variables
data_901 = pd.read_csv(f'data/labeled_901_pro.csv').iloc[:, 1:-2]
data_905 = pd.read_csv(f'data/labeled_905_pro.csv').iloc[:, 1:-2]
data_906 = pd.read_csv(f'data/labeled_906_pro.csv').iloc[:, 1:-2]
data_907 = pd.read_csv(f'data/labeled_907_pro.csv').iloc[:, 1:-2]

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_901.iloc[:, :] = scaler.fit_transform(data_901.iloc[:, :])
data_905.iloc[:, :] = scaler.fit_transform(data_905.iloc[:, :])
data_906.iloc[:, :] = scaler.fit_transform(data_906.iloc[:, :])
data_907.iloc[:, :] = scaler.fit_transform(data_907.iloc[:, :])

# Concatenating horizontally
data = pd.concat([data_901, data_905, data_906, data_907], axis=1)

# Sorting columns alphabetically
data = data.reindex(sorted(data.columns), axis=1)

# Assigning new names to columns
new_column_names = ['am-901', 'am-905', 'am-906', 'am-907',
                    'co-901', 'co-905', 'co-906', 'co-907', 
                    'do-901', 'do-905', 'do-906', 'do-907', 
                    'ph-901', 'ph-905', 'ph-906', 'ph-907',
                    'tu-901', 'tu-905', 'tu-906', 'tu-907',
                    'wt-901', 'wt-905', 'wt-906', 'wt-907']

data.columns = new_column_names

# Plotting the violins
plt.figure(figsize=(12, 8))
sns.violinplot(data=data)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel("Nomalized values", fontsize=16)
plt.title("Violin plot of each variable", fontsize=18)

plt.savefig('plots/violins.pdf', format='pdf', dpi=300, bbox_inches='tight')

plt.show()