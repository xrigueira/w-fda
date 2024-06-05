"""This program contains two functions:
- yearer: reads the full processed data and saves a 
new file with the years desired.
- seasoner: read the full processed data and saves
a new file with the months of the season desired.
-reverter: reverts the data to the original."""

import pandas as pd

def yearer(station: int = 901, years: list = [2018, 2019, 2020]):

    """This function reads the full processed data and saves a new file with 
    the years desired.
    """

    data = pd.read_csv(f'full_data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

    # Convert the 'date' column to datetime type
    data['date'] = pd.to_datetime(data['date'])

    # Filter the data by the years
    data = data[data['date'].dt.year.isin(years)]
    
    # Save the data
    data.to_csv(f'data/labeled_{station}_pro.csv', index=False)

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

def reverter(station: int = 901):

    """This function reverts the data to the original.
    """

    # Read the data
    data = pd.read_csv(f'full_data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

    # Save the data
    data.to_csv(f'data/labeled_{station}_pro.csv', index=False)

if __name__ == '__main__':

    # years = [2005, 2006, 2007, 2008, 2009]
    # yearer(station=901, years=years)

    # seasonser(station=901, season='winter')

    reverter(station=901)