import csv
import numpy as np
import pandas as pd
import rpy2.robjects as robjects

from outDec import outDec
from datetime import datetime

from main import MSA

# I am going to have to change builder, get_timestamps and real_outdec

class simulator(MSA):
    
    def __init__(self, simulation, N, L, P, projections, basis, detection_threshold, contamination, neighbors) -> None:
        self.simulation = simulation
        self.N = N                      # Number of distintc functional observations (number of days in my case: 1092)
        self.L = L                      # Number of components of the data (number of variables)
        self.P = P                      # Length of the series (legth of one day in my case: 96)
        self.projections = projections
        self.basis = basis
        self.detection_threshold = detection_threshold
        self.contamination = contamination
        self.neighbors = neighbors
        self.mdata = None
        self.cont_mdata = None
        self.saved_data = None
        self.outliers_outliergram = None
        self.outliers_muod = None
        self.outliers_ms = None
        self.timestamps = None
        self.magnitude = None
        self.shape = None
        self.amplitude = None
        self.distances = None
        self.outliers_in_data = None
        self.index_outliers = None
        
    def call_generator(self):
        
        with open('fda.R', 'r') as file:
            r_code = file.read()
    
            # Excecute the R function data_generator()
            robjects.r(r_code)
            mdata = robjects.r['data_generator'](self.N, self.L, self.P)
            self.mdata = mdata
    
    def call_contaminator(self):
        
        with open('fda.R', 'r') as file:
            r_code = file.read()
    
            # Excecute the R function data_contaminator()
            robjects.r(r_code)
            cont_mdata = robjects.r['data_contaminator'](self.N, self.L, self.P, self.mdata, self.contamination)
            self.cont_mdata = cont_mdata
    
    def call_saver(self):
        
        with open('fda.R', 'r') as file:
            r_code = file.read()
        
            # Excecute the R function data_saver()
            robjects.r(r_code)
            saved_data = robjects.r['data_saver'](int(len(self.cont_mdata[0])/self.P), self.L, self.P, self.cont_mdata)
            self.saved_data = saved_data
    
    def call_outliergram(self):

        with open('fda.R', 'r') as file:
            r_code = file.read()

            # Execute the R function get_msa()
            robjects.r(r_code)
            outliers_outliergram = robjects.r['my_outliergram'](self.P, self.cont_mdata)
            self.outliers_outliergram = outliers_outliergram

    def call_muod(self):
        
        with open('fda.R', 'r') as file:
            r_code = file.read()

            # Execute the R function get_msa()
            robjects.r(r_code)
            outliers_muod= robjects.r['my_muod'](self.P, self.saved_data)
            self.outliers_muod = outliers_muod
    
    def call_ms(self):
        
        with open('fda.R', 'r') as file:
            r_code = file.read()

            # Execute the R function get_msa()
            robjects.r(r_code)
            outliers_muod= robjects.r['my_ms'](self.cont_mdata, self.projections)
        
    def call_msa(self):
        
        with open('fda.R', 'r') as file:
            r_code = file.read()

            # Execute the R function get_msa()
            robjects.r(r_code)
            msa = robjects.r['get_msa'](self.simulation, self.projections, self.basis)
            
            # Convert and save the result to a numpy.ndarray
            msa = np.array(msa)
            self.msa = msa # Store the result in the instance variable
            np.save('msa.npy', msa, allow_pickle=False, fix_imports=False) # Remove then the program is finished
    
    def get_timestamps(self):
        
        timestamps = np.arange(1, len(self.msa) + 1, 1)
        self.timestamps = timestamps
    
    def real_outdec(self):
        print(self.index_outliers)
        print('Development pending')


if __name__ == '__main__':
    
    # Create an instance of the simulation class
    simulator_instance = simulator(simulation=True,  N=200, L=6, P=96, projections=200, basis=48, detection_threshold=15, contamination=0.05, neighbors=10)

    # Generate synthetic data
    simulator_instance.call_generator()
    
    # Contaminate the synthetic data
    simulator_instance.call_contaminator()
    
    # Saved the generated data
    simulator_instance.call_saver()
    
    # Call the outliergram
    # simulator_instance.call_outliergram()
    
    # Call MUOD
    # simulator_instance.call_muod()
    
    # Call MS Dai Genton
    simulator_instance.call_ms()
    
    # Calculate magnitude, shape, and amplitude
    # simulator_instance.call_msa()
    
    # Get the timestamps
    # simulator_instance.get_timestamps()
    
    # Detect outliers if any
    # simulator_instance.outlier_detector()
    
    # Extract real outliers
    # simulator_instance.real_outdec()
    
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

# my_muod(P, saved_df)