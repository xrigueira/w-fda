import csv
import numpy as np
import pandas as pd
import rpy2.robjects as robject

from outDec import outDec
from datetime import datetime

class simulator():
    
    def __init__(self) -> None:
        pass
    
    def get_timestamps(self):
        
        # I have to adapt this function so it returns indices of the different days.
        # It could be just by using the length of the mts
        # Open the CSV file and read the data
        with open(f'data/labeled_{self.station}_pro_msa.csv', 'r') as file:
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
        

# R code that I need to run here
# Define its parameters
# N <- 200      # Number of distintc functional observations (number of days in my case: 1092)
# L <- 6      # Number of components of the data (number of variables)
# P <- 96    # Length of the series (legth of one day in my case: 96)

# data <- data_generator(N, L, P)

# data_contaminated <- data_contaminator(N, data, contamination = 0.05)

# saved_df <- data_saver(N = nrow(data_contaminated[[1]]), L, P, data_contaminated)

# outliers <- my_outliergram(data_contaminated)
# print(outliers$ID_outliers)

# my_muod(saved_df, P)