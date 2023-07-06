import os

from checkGaps import checkGaps
from normalizer import normalizer
from joiner import joiner
from filterer import mfilterer

"""Preprocessing starts from the original univariate txt files."""

# Define the data we want to study
files = [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f))]

varNames = [i[0:-4] for i in files] # extract the names of the variables
stations = [901, 902, 904, 905, 906, 907, 910, 916] # Define with stations to process

# Define the time dram we want to use (a: months (not recommended), b: weeks, c: days). 
timeFrame = 'c'
timeStep = '15 min'

if __name__ == '__main__':

    for varName in varNames:

        # Fill in the gaps in the time series
        checkGaps(File=f'{varName}.txt', timestep=timeStep, varname=varName)
        print('[INFO] checkGaps() DONE')

        # Normalize the data. See normalizer.py for details
        normalizer(File=f'{varName}_full.csv', timeframe=timeFrame, timestep=timeStep, varname=varName)
        print('[INFO] normalizer() DONE')
    
    for station in stations:
    
        # Join the normalized databases
        joiner(station=station)
        print('[INFO] joiner() DONE')
    
    # RUN labeler.py MANUALLY. <- Fix this whenever I have some time
    
    for station in stations:

        # Filter out those months or weeks or days (depending on the desired
        # time unit) with too many NaN in several variables and iterate on the rest
        mfilterer(File=f'labeled_{station}.csv', timeframe=timeFrame, timestep=timeStep)
        print('[INFO] filterer() DONE')