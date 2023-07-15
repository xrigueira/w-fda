import pandas as pd
import numpy as np

def real_outdec(station):
    # Read the csv file
    df = pd.read_csv(f"data/labeled_{station}_pro.csv", sep=',', encoding='utf-8')
    
    # Convert the 'date' column to a proper date format
    df['date'] = pd.to_datetime(df['date'])

    # Set the 'date' column as the index
    df.set_index('date', inplace=True)

    # Resample the data by day and calculate the mean label
    average_label = df['label'].resample('D').mean()

    # Store the average values in a numeric vector
    average_labels = average_label.values

    # Thresholding operation: values above 0.5 set to 1, rest set to 0
    average_labels = np.where(average_labels >= 0.5, 1, 0)

    # Get the index of the days labeled as 1
    outliers_indexes = np.where(average_labels == 1)[0]

    # Return the resulting numeric object with named values
    return outliers_indexes

