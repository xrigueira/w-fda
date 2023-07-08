# Try to run get_msa in this Python file and print its results
# or put them in a Pandas database

import csv
import numpy as np
import rpy2.robjects as robjects
import plotly.graph_objects as go

from outDec import outDec
from datetime import datetime

# # Load the R function from msaCalc.R
# with open('msaCalc.R', 'r') as file:
#     r_code = file.read()

# # Execute the R function get_msa()
# robjects.r(r_code)
# msa = robjects.r['get_msa']()

# # Convert and save the result to a numpy.ndarray
# msa = np.array(msa)
# np.save('msa.npy', msa, allow_pickle=False, fix_imports=False)

# Check if there are outliers in the data
msa = np.load('msa.npy')
magnitude = msa[:, 0]
shape = msa[:, 1]
amplitude = msa[:, 2]

outliers_in_data = outDec(magnitude=magnitude, shape=shape, amplitude=amplitude)

# Get time stamps
#########Turn this into a function#########

# Define file path
data_file = 'data/labeled_901_pro.csv'

# Open the CSV file and read the data
with open(data_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    
    # Initialize a list to store unique datetime objects
    unique_days = []
    
    # Iterate over each row in the dataset
    for row in csv_reader:
        # Extract the date and convert it to a datetime object
        date_str = row['date']
        datetime_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        
        # Truncate the time information (optional)
        datetime_obj = datetime_obj.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Add the datetime object to the unique_days list if it's not already present
        if datetime_obj not in unique_days:
            unique_days.append(datetime_obj)
        
# Convert unique_days to a numpy array and change datetime format
unique_days = [datetime_obj.strftime('%d-%m-%Y') for datetime_obj in unique_days]
timestamps = np.array(unique_days)
##################

# Now I have to build the outlier detector (unsupervised kNN)
contamination = 0.1
from sklearn.neighbors import NearestNeighbors

modelkNN = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
modelkNN.fit(msa)
distances, indexes = modelkNN.kneighbors(msa)

indexkNN = np.where(distances.mean(axis=1) >= np.quantile(distances.mean(axis=1), (1-contamination)))
valueskNN = msa[indexkNN]
timestampskNN = timestamps[indexkNN]

# # Plot distance
# fig = go.Figure(data=[go.Scatter3d(x=magnitude, y=shape, z=amplitude, mode='markers', 
#                                 marker=dict(color=distances.mean(axis=1), colorscale='Viridis', colorbar=dict(title='Distance')),
#                                 hovertemplate='<b>Timestamp:</b> %{text}<br>'
#                                 '<b>Magnitude:</b> %{x}<br>'
#                                 '<b>Shape:</b> %{y}<br>'
#                                 '<b>Amplitude:</b> %{z}<br>'
#                                 '<b>Distance:</b> %{marker.color}<extra></extra>',
#                                 text=unique_days)])

# # Set layout options
# fig.update_layout(
#     scene=dict(
#         xaxis=dict(title='Magnitude'),
#         yaxis=dict(title='Shape'),
#         zaxis=dict(title='Amplitude')
#     ),
#     title='Distance MSA Plot'
# )

# # Show the plot
# fig.show()

# Plot outliers
fig = go.Figure(data=[go.Scatter3d(x=magnitude, y=shape, z=amplitude, mode='markers', 
                                marker=dict(color=np.where(np.isin(np.arange(len(magnitude)), indexkNN), 'red', 'blue'), colorscale=[[0, 'blue'], [1, 'red']], colorbar=dict(title='Outlier')),
                                hovertemplate='<b>Timestamp:</b> %{text}<br>'
                                '<b>Magnitude:</b> %{x}<br>'
                                '<b>Shape:</b> %{y}<br>'
                                '<b>Amplitude:</b> %{z}<br>'
                                '<b>Distance:</b> %{marker.color}<extra></extra>',
                                text=unique_days)])

# Set layout options
fig.update_layout(
    scene=dict(
        xaxis=dict(title='Magnitude'),
        yaxis=dict(title='Shape'),
        zaxis=dict(title='Amplitude')
    ),
    title='Outliers MSA Plot'
)

# Show the plot
fig.show()



