import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

"""This file is used to plot the multivariate
data of each 4 hour period in a data base."""

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

# Read label data
y_16 = np.load('y_16.npy', allow_pickle=False, fix_imports=False)
y_msa = np.load('y_msa.npy', allow_pickle=False, fix_imports=False)
y_rf = np.load('y_rf.npy', allow_pickle=False, fix_imports=False)

counter = 0
# Group the data by 4-hour intervals and iterate over each interval
for interval, interval_data in data.groupby(pd.Grouper(freq='4H')):
    
    if interval_data.empty != True:
        
        # Create a new plot for each interval
        fig, ax = plt.subplots()

        # Plot columns 1 to 6 for the current interval
        interval_data.iloc[:, 0:-1].plot(ax=ax)

        # Customize the plot labels, title, etc.
        ax.set_title(f'Data for {interval}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        
        # Reduce the size of the legend
        ax.legend(fontsize='small')
        
        # Add a text box with the ground truth info and the results for each method
        text_box = f'Ground truth: {y_16[counter]}\nResult MSA: {int(y_msa[counter])}\nResult RF: {int(y_rf[counter])}'
        fig.text(0.8, 0.87, text_box, bbox=dict(boxstyle="square", facecolor="white", edgecolor="black"))

        counter += 1
        
        # Save the image
        fig.subplots_adjust(bottom=0.19)
        fig.savefig(f'images/plot_{station}_{str(interval)[0:10]}_{str(interval)[11:13]}.png', format='png', dpi=300)

        # Close the fig for better memory management
        plt.close(fig=fig)
        
        if (counter % 100) == 0:
            print(counter)
