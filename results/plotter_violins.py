import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

"""This file is used to generate the violin-
style plots for the data of each variable."""

# Read the data
station = 901
data = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8', parse_dates=['date'])

# Set the 'date' column as the DataFrame's index
data.set_index('date', inplace=True)

# Drop not needed columns
data = data.drop(data.columns[6:], axis=1)

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data.iloc[:, :] = scaler.fit_transform(data.iloc[:, :])


fig, ax = plt.subplots(figsize=(12, 8))
# fig.subplots_adjust(bottom)
sns.set(style="whitegrid")
sns.violinplot(data=data)

names = ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature']
plt.xticks(range(len(names)), names, fontsize=16)
plt.ylabel("Nomalized values", fontsize=16)
plt.title("Violin plot of each variable", fontsize=18)
plt.show()

fig.savefig(f'violins.png', format='png', dpi=300)
