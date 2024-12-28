import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

"""This file is used to plot the univariate data
generated for the simulation."""

data = pd.read_csv('data/generated_data.csv')

# Number of columns in the dataset
num_columns = data.shape[1]

# Create a figure with 6 subplots
fig, axes = plt.subplots(num_columns, 1, figsize=(16, 6))

# Iterate over each column
for i, column in enumerate(data.columns):
    # Get the data for the current column
    column_data = data[column]
    
    # Split the data into chunks of 96 values
    chunks = [column_data[j:j+96] for j in range(0, len(column_data), 96)]
    
    # Plot each chunk as a separate line
    for chunk in chunks:
        axes[i].plot(chunk)
    
    # Set the title for each subplot
    # axes[i].set_title(f'Plot for {column}')

# Adjust layout
plt.tight_layout()
plt.show()

