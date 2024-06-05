# Novelty: amplitude, non parametric detector, GANs for the generation of artificial WQ data.

import csv
import time
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import plotly.graph_objects as go

from outDec import outDec
from datetime import datetime

class MSA():
    
    def __init__(self, station, projections, basis, detection_threshold, contamination, neighbors, real_outliers_threshold) -> None:
        self.station = station
        self.projections = projections
        self.basis = basis
        self.detection_threshold = detection_threshold
        self.contamination = contamination
        self.neighbors = neighbors
        self.real_outliers_threshold = real_outliers_threshold
        self.timestamps = None
        self.msa = None
        self.magnitude = None
        self.shape = None
        self.amplitude = None
        self.distances = None
        self.outliers_in_data = None
        self.index_outliers = None
    
    def get_timestamps(self):
        
        # Open the CSV file and read the data
        with open(f'data/labeled_{self.station}_pro.csv', 'r') as file:
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
        self.timestamps = timestamps
        
        return timestamps
        
    def call_msa(self):
        """Write documentation.
        ----------
        Arguments:
        self.
        projections (int): number of projections used to calculate magnitude and shape.
        basis (int): number of basis functions used to calculate amplitude.
        
        Return:
        msa (np.array): object with the magnitude, shape, and amplitude value
        of each function."""
        
        # Load the R function from msaCalc.R
        with open('msaCalc.R', 'r') as file:
            r_code = file.read()

            # Execute the R function get_msa()
            robjects.r(r_code)
            msa = robjects.r['get_msa'](self.projections, self.basis)

            # Convert and save the result to a numpy.ndarray
            msa = np.array(msa)
            self.msa = msa # Store the result in the instance variable
            # np.save('msa.npy', msa, allow_pickle=False, fix_imports=False) # Remove then the program is finished

    def outlier_detector(self):
        
        # Check if there are outliers in the data
        # self.msa = np.load('msa.npy', allow_pickle=False, fix_imports=False) # Remove then the program is finished
        magnitude = self.msa[:, 0]
        shape = self.msa[:, 1]
        amplitude = self.msa[:, 2]
        self.magnitude, self.shape, self.amplitude = magnitude, shape, amplitude

        outliers_in_data = outDec(magnitude=magnitude, shape=shape, amplitude=amplitude, detection_threshold=self.detection_threshold)
        self.outliers_in_data = outliers_in_data
        
        if outliers_in_data == True:

            # Now I have to build the outlier detector (unsupervised kNN)
            from sklearn.neighbors import NearestNeighbors

            modelkNN = NearestNeighbors(n_neighbors=self.neighbors, algorithm='ball_tree')
            modelkNN.fit(self.msa)
            distances, indexes = modelkNN.kneighbors(self.msa)
            self.distances = distances

            index_outliers = np.where(distances.mean(axis=1) >= np.quantile(distances.mean(axis=1), (1-self.contamination)))
            values_outliers = self.msa[index_outliers]
            timestamps_outliers = self.timestamps[index_outliers]
            self.index_outliers = index_outliers
        
        else:
            print('No outliers found in the data.')
            self.index_outliers = tuple()

    def plots(self):
        
        if self.outliers_in_data == True:

            # Plot distance
            fig = go.Figure(data=[go.Scatter3d(x=self.magnitude, y=self.shape, z=self.amplitude, mode='markers', 
                                            marker=dict(color=self.distances.mean(axis=1), colorscale='Viridis', colorbar=dict(title='Distance')),
                                            hovertemplate='<b>Timestamp:</b> %{text}<br>'
                                            '<b>Magnitude:</b> %{x}<br>'
                                            '<b>Shape:</b> %{y}<br>'
                                            '<b>Amplitude:</b> %{z}<br>'
                                            '<b>Distance:</b> %{marker.color}<extra></extra>',
                                            text=list(self.timestamps))])

            # Set layout options
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='Magnitude'),
                    yaxis=dict(title='Shape'),
                    zaxis=dict(title='Amplitude')
                ),
                title='Distance MSA Plot'
            )

            # Show the plot
            fig.show()

            # Plot outliers
            fig = go.Figure(data=[go.Scatter3d(x=self.magnitude, y=self.shape, z=self.amplitude, mode='markers', 
                                            marker=dict(color=np.where(np.isin(np.arange(len(self.magnitude)), self.index_outliers), 'red', 'black'), colorscale=[[0, 'black'], [1, 'red']], colorbar=dict(title='Outlier')),
                                            hovertemplate='<b>Timestamp:</b> %{text}<br>'
                                            '<b>Magnitude:</b> %{x}<br>'
                                            '<b>Shape:</b> %{y}<br>'
                                            '<b>Amplitude:</b> %{z}<br>'
                                            '<b>Distance:</b> %{marker.color}<extra></extra>',
                                            text=list(self.timestamps))])

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

            # Plot comparison to labeled outliers
            from main import real_outdec

            real_outliers_indices = real_outdec(station=901, real_outlier_threshold=self.real_outliers_threshold)

            # Create an array of outlier indices (indexkNN)
            outliers_indices = np.array(self.index_outliers)

            # Plot outliers
            fig = go.Figure(data=[go.Scatter3d(x=self.magnitude, y=self.shape, z=self.amplitude, mode='markers',
                    marker=dict(
                        color=['green' if i in real_outliers_indices and i in outliers_indices
                            else 'blue' if i in real_outliers_indices
                            else 'red' if i in outliers_indices
                            else 'black'  # Non-outliers in black
                            for i in range(len(self.magnitude))
                            ],
                        colorscale=[[0, 'blue'], [0.5, 'red'], [1, 'green']],
                        colorbar=dict(title='Leyend')
                    ),
                    hovertemplate='<b>Timestamp:</b> %{text}<br>'
                                '<b>Magnitude:</b> %{x}<br>'
                                '<b>Shape:</b> %{y}<br>'
                                '<b>Amplitude:</b> %{z}<br>'
                                '<b>Distance:</b> %{marker.color}<extra></extra>',
                    text=list(self.timestamps)
                )
            ])

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

    def metric(self):
        
        from main import real_outdec

        real_outliers_indices = real_outdec(station=901, real_outlier_threshold=self.real_outliers_threshold)

        # Create an array of outlier indices (indexkNN)
        outliers_indices = np.array(self.index_outliers)
        
        real_outliers_indices_set = set(real_outliers_indices.tolist()) # Convert to list and then to set
        outliers_indices_set = set(np.ravel(outliers_indices).tolist()) # Make 1D, convert to list and then to set
        intersection = len(real_outliers_indices_set.intersection(outliers_indices_set))
        union = len(real_outliers_indices_set.union(outliers_indices_set))
        
        return intersection / union if union > 0 else 1.0


if __name__ == '__main__':
    
    station = 901
    
    range_projections = [*range(200, 300, 100)]
    range_basis = [*range(48, 64, 16)]
    range_detection_threshold = [*range(15, 30, 15)]
    range_contamination = [0, 0.05, 0.1, 0.25, 0.2]
    range_neighbors = [*range(10, 20, 10)]
    range_real_outliers_threshold = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Get the data for each combination, save it and process it
    results = pd.DataFrame(columns=['projections', 'basis', 'detection_threshold', 'contamination',
                                    'neighbors', 'real_outliers_threshold', 'similarity'])
    
    for projections in range_projections:
        
        for basis in range_basis:
            
            for detection_threshold in range_detection_threshold:
                
                for contamination in range_contamination:
                    
                    for neighbors in range_neighbors:
                        
                        for real_outliers_threshold in range_real_outliers_threshold:
                            
                            t1 = time.time()
                            
                            # Create a class instance
                            msa_instance = MSA(station=station, projections=projections, basis=basis, detection_threshold=detection_threshold, 
                                            contamination=contamination, neighbors=neighbors, real_outliers_threshold=real_outliers_threshold)

                            # Get the timestamps
                            msa_instance.get_timestamps()
                            
                            # Calculate magnitude, shape, and amplitude
                            msa_instance.call_msa()
                            
                            # Detect outliers if any
                            msa_instance.outlier_detector()
                            
                            # Plot the results
                            # msa_instance.plots()
                            
                            # Calculate accuracy
                            similarity = msa_instance.metric()
                            
                            t2 = time.time() - t1
                            
                            print(f'Run time: {round(t2, ndigits=2)} seconds')
                            
                            # Add the results of each iteration to the dataframe
                            results.loc[len(results.index)] = [projections, basis, detection_threshold, contamination, neighbors, real_outliers_threshold, similarity]

    # Save the results
    results.to_csv(f'results.csv', sep=',', encoding='utf-8', index=True)


