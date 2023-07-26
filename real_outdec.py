
import pandas as pd
import numpy as np

def real_outdec(station, real_outlier_threshold):
    
    # Read the csv file
    data = pd.read_csv(f"data/labeled_{station}_pro.csv", sep=',', encoding='utf-8')

    # Convert the 'date' column to datetime type
    data['date'] = pd.to_datetime(data['date'])

    # Extract the date part from the datetime and create a new column 'day'
    data['day'] = data['date'].dt.date

    # Group the data by 'day' and calculate the average of the 'label' column within each group
    average_labels = data.groupby('day', sort=False)['label'].mean()

    # Apply thresholding operation
    average_labels = average_labels.apply(lambda x: 1 if x >= real_outlier_threshold else 0)
    
    outliers_dates = average_labels[average_labels == 1].index
    outliers_indexes = np.where(average_labels == 1)[0]

    # Return the resulting objects
    return outliers_indexes, outliers_dates