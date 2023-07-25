
import pandas as pd
import numpy as np

def real_outdec(station, real_outlier_threshold):
    
    # Read the csv file
    df = pd.read_csv(f"data/labeled_{station}_pro.csv", sep=',', encoding='utf-8')
    
    # Convert the 'date' column to a proper date format
    df['date'] = pd.to_datetime(df['date'])

    # Calculate the mean label for each unique date and reset the index
    average_labels = df.groupby(df['date'].dt.date)['label'].mean().reset_index()

    # Thresholding operation: values above the threshold set to 1, rest set to 0
    average_labels['label'] = (average_labels['label'] >= real_outlier_threshold).astype(int)

    # Get the numeric index of the days labeled as 1
    outliers_indexes = average_labels[average_labels['label'] == 1].index.values

    # Return the resulting numeric object with named values
    return outliers_indexes

outliers_indexes = real_outdec(station=901, real_outlier_threshold=0.1)
print(outliers_indexes)
