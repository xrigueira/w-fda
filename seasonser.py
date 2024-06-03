"""This program read the full processed data and saves
a new file with the months of the season desired."""

import pandas as pd

def seasonser(station: int = 901, season: str = 'spring'):

    """This function reads the full processed data and saves a new file with 
    the months of the season desired.
    """

    data = pd.read_csv(f'full_data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

    # Convert the 'date' column to datetime type
    data['date'] = pd.to_datetime(data['date'])

    # Define months of each season
    seasons = {
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'fall': [9, 10, 11],
        'winter': [12, 1, 2]
    }

    # Filter the data by the season
    data = data[data['date'].dt.month.isin(seasons[season])]
    
    # Save the data
    data.to_csv(f'data/labeled_{station}_pro.csv', index=False)

seasonser(station=901, season='winter')