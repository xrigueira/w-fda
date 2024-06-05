import csv
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import plotly.graph_objects as go

from outDec import outDec
from datetime import datetime

class MSA():
    
    def __init__(self, station, hours, nhours, simulation, search, projections, basis, detection_threshold, contamination, neighbors, real_outliers_threshold) -> None:
        self.station = station
        self.hours = hours
        self.nhours = nhours
        self.simulation = simulation
        self.search = search
        self.projections = projections
        self.basis = basis
        self.detection_threshold = detection_threshold
        self.contamination = contamination
        self.neighbors = neighbors
        self.real_outliers_threshold = real_outliers_threshold
        self.rf_weights = None
        self.msa = None
        self.timestamps = None
        self.magnitude = None
        self.shape = None
        self.amplitude = None
        self.distances = None
        self.outliers_in_data = None
        self.index_outliers = None

    def rf(self):
        
        """Use RF to enhance the MSA method (not used)."""

        # Read the data
        data = pd.read_csv(f'data/labeled_{self.station}_pro.csv', sep=',', encoding='utf-8')

        # Convert variable columns to np.ndarray
        X = data.iloc[:, 1:7].values
        y = data.iloc[:, -1].values
        
        # Get the number or rows in the database
        num_rows = X.shape[0]

        # Save the original order
        original_indices = np.arange(num_rows)

        # Generate random indices to shuffle the data
        np.random.seed(0)
        random_indices = np.random.permutation(num_rows)

        # Use the random indices to shuffle the data
        shuffled_X = X[random_indices]
        shuffled_y = y[random_indices]

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = shuffled_X[:378240], shuffled_X[378240:], shuffled_y[:378240], shuffled_y[378240:]
        
        if self.search == True:
            
            # Define the parameters to iterate over
            param_dist = {'n_estimators': [50, 75, 100, 125, 150, 175], 'max_depth': [1, 2, 3, 4, 5, 10, 15, 20, 50, None],
                        'min_samples_split': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 3, 4, 5]}
            
            from sklearn.model_selection import RandomizedSearchCV
            from sklearn.ensemble import RandomForestClassifier
            rand_search = RandomizedSearchCV(RandomForestClassifier(random_state=0), param_distributions = param_dist, n_iter=5, cv=5)
            
            rand_search.fit(X_train, y_train)
            
            # Get best params
            best_params = rand_search.best_params_
            best_model = rand_search.best_estimator_
            print('Best params', best_params, '| Best model', best_model)
            
            # Make predictions on the testing data
            y_hat = best_model.predict(shuffled_X)
            
        elif self.search == False:
            
            # Call the model
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=0)

            # Fit the model to the training data
            model.fit(X_train, y_train)

            # Make predictions on the testing data
            y_hat = model.predict(shuffled_X)
        
        # Get the accuracy of the model
        from sklearn.metrics import accuracy_score, confusion_matrix
        accuracy = accuracy_score(shuffled_y, y_hat)
        print('Accuracy', accuracy)

        # Get the number of rows labeled as anomalies in y_test
        print('Number of anomalies', len([i for i in shuffled_y if i==1]))

        # Display the confusion matrix
        if self.search == True:
            confusion_matrix = confusion_matrix(shuffled_y, best_model.predict(shuffled_X))
        elif self.search == False:
            confusion_matrix = confusion_matrix(shuffled_y, model.predict(shuffled_X))

        print(confusion_matrix)
        
        # Use the original indices to restore the predictions to the original order of the labels
        restored_y_hat = y[original_indices]

        # Extract the average predicted label per day
        grouped_y_hat = restored_y_hat[378240:].reshape(-1, 96)

        # Get the average if each group
        rf_weights = np.mean(grouped_y_hat, axis=1)
        # np.save('rf_weights.npy', rf_weights, allow_pickle=False, fix_imports=False) # Remove when the program is finixed
        self.rf_weights = rf_weights
        
    def call_msa(self):
        """Turn the data into functions and calculate the magnitude, shape, amplitude of each function.
        ----------
        Arguments:
        self.
        projections (int): number of projections used to calculate magnitude and shape.
        basis (int): number of basis functions used to calculate amplitude.
        
        Return:
        msa (np.array): object with the magnitude, shape, and amplitude value
        of each function."""
        
        # Load the R function 'get_msa()' from fda.R
        with open('fda.R', 'r') as file:
            r_code = file.read()

            # Execute the R function get_msa()
            robjects.r(r_code)
            msa = robjects.r['get_msa'](self.station, self.hours, self.nhours, self.simulation, self.projections, self.basis)

            # Convert and save the result to a numpy.ndarray
            msa = np.array(msa)
            self.msa = msa # Store the result in the instance variable
            # np.save('msa.npy', msa, allow_pickle=False, fix_imports=False) # Remove then the program is finished
            
            # Apply the weights obtained with Random Forest
            # self.rf_weights = np.load('rf_weights.npy')
            # self.msa = msa * (1 + self.rf_weights[:, np.newaxis])

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
        
        if self.hours == True:
        
            # Create the time stamps for the hourly setting
            number_blocks = 96 / (self.nhours * 4) # data points in a day / data points in the chosen hourly unit

            # Repeat each element `number_blocks` times
            repeated_timestamps = np.repeat(timestamps, number_blocks)
            
            # Create a sequence of numbers from 1 to `number_blocks` for each element
            sequence = np.tile(np.arange(1, number_blocks + 1), len(timestamps))
            
            # Combine the repeated timestamps and sequence as strings
            timestamps_hours = np.array([f'{timestamp}-{seq}' for timestamp, seq in zip(repeated_timestamps, sequence)])
            self.timestamps = timestamps_hours
        
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

            print('Outliers detected in the data')
            
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
            # np.save('y_indices_fda.npy', index_outliers, allow_pickle=False, fix_imports=False)
        
        else:
            print('No outliers found in the data.')
            self.index_outliers = tuple()
    
    def real_outliers(self):
    
        """Get the indices and dates of the real outliers in the data."""

        # Read the csv file
        data = pd.read_csv(f"data/labeled_{self.station}_pro.csv", sep=',', encoding='utf-8')

        # Convert the 'date' column to datetime type
        data['date'] = pd.to_datetime(data['date'])
        
        if self.hours == False:

            # Extract the date part from the datetime and create a new column 'day'
            data['day'] = data['date'].dt.date

            # Group the data by 'day' and calculate the average of the 'label' column within each group
            average_labels = data.groupby('day', sort=False)['label'].mean()

            # Apply thresholding operation
            average_labels = average_labels.apply(lambda x: 1 if x >= self.real_outliers_threshold else 0)
            
            outliers_dates = average_labels[average_labels == 1].index
            outliers_indexes = np.where(average_labels == 1)[0]
        
        elif self.hours == True:
            
            # Create a new column 'time_block' to group dates into 4-hour intervals
            data['time_block'] = data['date'].dt.floor(f'{self.nhours}H')

            # Group the data by 'time_block' and calculate the average of the 'label' column within each group
            average_labels = data.groupby('time_block', sort=False)['label'].mean()

            # Apply thresholding operation
            average_labels = average_labels.apply(lambda x: 1 if x >= self.real_outliers_threshold else 0)

            outliers_dates = average_labels[average_labels == 1].index
            outliers_indexes = np.where(average_labels == 1)[0]
        
        # Return the resulting objects
        return outliers_indexes, outliers_dates
    
    def detected_outliers(self):

        """Get the indices and dates of the outliers detected by the MSA method."""

        if self.outliers_in_data == True:

            # Read the csv file
            data = pd.read_csv(f"data/labeled_{self.station}_pro.csv", sep=',', encoding='utf-8')

            # Convert the 'date' column to datetime type
            data['date'] = pd.to_datetime(data['date'])
        
            if self.hours == False:

                # Extract the date part from the datetime and create a new column 'day'
                data['day'] = data['date'].dt.date

                # Group by day
                data_grouped = data.groupby('day', sort=False)['week'].first()

                # Get the outliers detected by the MSA method
                outliers_indixes = self.index_outliers
                
                # Get the dates of the outliers detected by the MSA method
                outliers_dates = data_grouped.iloc[outliers_indixes].index
        
            elif self.hours == True:
            
                # Create a new column 'time_block' to group dates into 4-hour intervals
                data['time_block'] = data['date'].dt.floor(f'{self.nhours}H')

                # Group the data by 'time_block' and apply some function (e.g., 'first') to turn it back into a DataFrame
                data_grouped = data.groupby('time_block', sort=False)['week'].first()
                
                # Get the outliers detected by the MSA method
                outliers_indixes =  self.index_outliers
                
                # Get the dates (index) of the outliers detected by the MSA method
                outliers_dates = data_grouped.iloc[outliers_indixes].index
            
            return outliers_indixes, outliers_dates
        
        else:
            return None, None

    def plots(self):
        
        """Plot the results of the MSA method."""

        if self.outliers_in_data == True:

            # Plot distance
            fig = go.Figure(data=[go.Scatter3d(x=self.magnitude, y=self.shape, z=self.amplitude, mode='markers', 
                                            marker=dict(size=4, color=self.distances.mean(axis=1), colorscale='Viridis', colorbar=dict(title='Distance')),
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
            
            # Save the interactive plot as an HTML file
            fig.write_html(f"results/distance_3Dplot_{self.station}.html", include_plotlyjs='cdn')
            
            # Plot outliers
            fig = go.Figure(data=[go.Scatter3d(x=self.magnitude, y=self.shape, z=self.amplitude, mode='markers', 
                                            marker=dict(size=4,color=np.where(np.isin(np.arange(len(self.magnitude)), self.index_outliers), 'red', 'black'), colorscale=[[0, 'black'], [1, 'red']]),
                                            hovertemplate='<b>Timestamp:</b> %{text}<br>'
                                            '<b>Magnitude:</b> %{x}<br>'
                                            '<b>Shape:</b> %{y}<br>'
                                            '<b>Amplitude:</b> %{z}<br>'
                                            '<b>Distance:</b> %{marker.color}<extra></extra>',
                                            text=list(self.timestamps),
                                            showlegend=False)])
            
            # Create a dummy scatter trace for "No outliers" (black dot)
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],  # Empty data, only for the legend
                mode='markers',
                marker=dict(color='black', size=8),  # Customize the size as needed
                name='No outlier'
            ))

            # Create a dummy scatter trace for "Outliers" (red dot)
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],  # Empty data, only for the legend
                mode='markers',
                marker=dict(color='red', size=8),  # Customize the size as needed
                name='Outlier'
            ))

            # Set layout options
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='Magnitude'),
                    yaxis=dict(title='Shape'),
                    zaxis=dict(title='Amplitude')
                ),
                title='Outliers MSA Plot',
                showlegend=True
            )

            # Show the plot
            fig.show()
            
            # Plot comparison to labeled outliers
            real_outliers_indices, real_outliers_dates = self.real_outliers()

            # Create an array of outlier indices (indexkNN)
            outliers_indices = np.array(self.index_outliers)

            # Plot outliers
            fig = go.Figure(data=[go.Scatter3d(x=self.magnitude, y=self.shape, z=self.amplitude, mode='markers',
                    marker=dict(
                        size=4,
                        color=['green' if i in real_outliers_indices and i in outliers_indices
                            else 'blue' if i in real_outliers_indices
                            else 'red' if i in outliers_indices
                            else 'black'  # Non-outliers in black
                            for i in range(len(self.magnitude))
                            ],
                        colorscale=[[0, 'blue'], [0.5, 'red'], [1, 'green']]
                    ),
                    hovertemplate='<b>Timestamp:</b> %{text}<br>'
                                '<b>Magnitude:</b> %{x}<br>'
                                '<b>Shape:</b> %{y}<br>'
                                '<b>Amplitude:</b> %{z}<br>'
                                '<b>Distance:</b> %{marker.color}<extra></extra>',
                    text=list(self.timestamps),
                    showlegend=False
                )
            ])
            
            # Create a dummy scatter trace for "No outliers" (black dot)
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],  # Empty data, only for the legend
                mode='markers',
                marker=dict(color='black', size=8),  # Customize the size as needed
                name='No outlier'
            ))

            # Create a dummy scatter trace for "MsA Outliers" (red dot)
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],  # Empty data, only for the legend
                mode='markers',
                marker=dict(color='red', size=8),  # Customize the size as needed
                name='MSA outlier'
            ))
            
            # Create a dummy scatter trace for "Real outliers" (blue dot)
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],  # Empty data, only for the legend
                mode='markers',
                marker=dict(color='blue', size=8),  # Customize the size as needed
                name='GT outlier'
            ))
            
            # Create a dummy scatter trace for "Correct detection"
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],  # Empty data, only for the legend
                mode='markers',
                marker=dict(color='green', size=8),  # Customize the size as needed
                name='GT and MSA outlier'
            ))
            
            # Set layout options
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='Magnitude'),
                    yaxis=dict(title='Shape'),
                    zaxis=dict(title='Amplitude')
                ),
                title='Outliers MSA Plot',
                showlegend=True
            )

            # Show the plot
            fig.show()
            
            # Save the interactive plot as an HTML file
            fig.write_html(f"results/labeled_3Dplot_{self.station}.html")

    def metric(self):

        """Calculate the accuracy of the MSA method."""

        real_outliers_indices, real_outliers_dates = self.real_outliers()
        print('Dates of the real outliers', real_outliers_dates)
        # Create an array of outlier indices (indexkNN)
        outliers_indices = np.array(self.index_outliers)
        
        print('Indices of the real outleirs', real_outliers_indices)
        print('Indices of the outliers detected', outliers_indices)
        # np.save(f'indices_y_real_outliers_{self.station}.npy', real_outliers_indices, allow_pickle=False, fix_imports=False)
        # np.save(f'indices_y_msa_{self.station}.npy', outliers_indices, allow_pickle=False, fix_imports=False)
        
        # Get ground truth binary list
        print(len(self.msa))
        y_gt = np.zeros(len(self.msa))
        y_gt[real_outliers_indices] = 1
        
        # Get msa binary list
        y_msa = np.zeros(len(self.msa))
        y_msa[outliers_indices] = 1

        # Get accuracy
        accuracy = np.sum(y_gt == y_msa) / len(y_gt)
        
        # Get confusion matrix
        from sklearn.metrics import confusion_matrix
        confusion = confusion_matrix(y_gt, y_msa)
        
        return accuracy, confusion
    
    def cleanser(self):

        """Remove the dates that have been correctly detected by the MSA method
        from the labeled data file."""

        # Get the indices and dates of the outliers detected by the MSA method
        detected_outliers_indices, detected_outliers_dates = self.detected_outliers()

        # Get the indices and dates of the real outliers in the data
        real_outliers_indices, real_outliers_dates = self.real_outliers()

        # Extract those instances present in real outliers but not in detected outliers
        dates_to_remove = real_outliers_dates.difference(detected_outliers_dates)
        print('Dates to remove', dates_to_remove)
        print('Number of dates to remove', len(dates_to_remove))

        # Read the labeled data file
        data = pd.read_csv(f"data/labeled_{self.station}_pro.csv", sep=',', encoding='utf-8')

        # Convert the 'date' column to datetime type
        data['date'] = pd.to_datetime(data['date'])

        # Create a new list of dates that includes the original dates and the dates 4 hours ahead
        from pandas.tseries.offsets import DateOffset
        dates_to_remove = dates_to_remove.union(dates_to_remove + DateOffset(hours=4))

        # Assume 'data' is your DataFrame and 'date' is the column with the dates
        mask = data['date'].isin(dates_to_remove)

        # Remove the dates
        data = data[~mask]



if __name__ == '__main__':
    
    station = 901
    
    # Create a class instance
    # nbasis days = 48; nbasis 4 hours = 8
    msa_instance = MSA(station=station, hours=True, nhours=4, simulation=False, search=False, projections=200, basis=8,
                    detection_threshold=15, contamination=0.15, neighbors=160, real_outliers_threshold=0.5)
    # Original values: detection_threshold=15, contamination=0.01, neighbors=10, real_outliers_threshold=0.5)
    # Best combo: detection_threshold=15, contamination=0.15, neighbors=160, real_outliers_threshold=0.5 || accuracy 60%
    # Good combo with less FN: detection_threshold=15, contamination=0.10, neighbors=300, real_outliers_threshold=0.5 || accuracy 56%
    
    # Calculate Random Forest scores
    # msa_instance.rf()
    
    # Calculate magnitude, shape, and amplitudec
    msa_instance.call_msa()
    
    # Get the timestamps
    msa_instance.get_timestamps()
    
    # Detect outliers if any
    msa_instance.outlier_detector()
    
    # Plot the results
    # msa_instance.plots()
    
    # Calculate accuracy
    accuracy, confusion_matrix = msa_instance.metric()
    print('Accuracy:', accuracy)
    print('Confusion matrix:\n', confusion_matrix)

    # Clean the data
    msa_instance.cleanser()

