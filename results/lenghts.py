"""Program to calculate the length of each anomaly in the dataset."""

import pandas as pd

# Load the data
df = pd.read_csv('anomalies.csv', sep=';', parse_dates=['Start_date', 'End_date'], dayfirst=True)

# Convert the 'Start' and 'End' columns to datetime
df['Start_date'] = pd.to_datetime(df['Start_date'], format='%d-%m-%Y %H:%M:%S')
df['End_date'] = pd.to_datetime(df['End_date'], format='%d-%m-%Y %H:%M:%S')

# Calculate the length of each anomaly
df['Length_minutes'] = (df['End_date'] - df['Start_date']).dt.total_seconds() / 60
df['Length_hours'] = df['Length_minutes'] / 60
df['Length_days'] = df['Length_hours'] / 24

# Save the dataframe
df.to_csv('results/anomalies_lenghts.csv', sep=';', index=False)