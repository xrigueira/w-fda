import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

"""This file is used to plot the multivariate
data between two specific dates."""

# Read the data
station = 901
data = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8', parse_dates=['date'])

# Set the 'date' column as the DataFrame's index
data.set_index('date', inplace=True)

# Drop not needed columns
data = data.drop(data.columns[6:-1], axis=1)

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data.iloc[:, 0:-1] = scaler.fit_transform(data.iloc[:, 0:-1])

# Define start and end date
start_date = '2006-10-16 00:00:00'
end_date = '2006-10-18 00:00:00'
anomaly_number = 4

# Filter the data between those dates
filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]

# Plot the data
# Create a new plot for each interval
fig, ax = plt.subplots()

# Plot columns 1 to 6 for the current interval
filtered_data.iloc[:, 0:-1].plot(ax=ax)

# Customize the plot labels, title, etc.
ax.set_title(f'Anomaly {anomaly_number}')
ax.set_xlabel('Date')
ax.set_ylabel('Value')

# Reduce the size of the legend
ax.legend(["am", "co", "do", "ph", "tu", "wt"], fontsize='small', loc='upper left')

# Save the image
fig.subplots_adjust(bottom=0.19)
# fig.savefig(f'plots/plot_{station}_anomaly_{anomaly_number}.png', format='png', dpi=300)
fig.savefig(f'plots/plot_{station}_anomaly_{anomaly_number}.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Close the fig for better memory management
plt.close(fig=fig)
