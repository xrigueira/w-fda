import csv
import time
import numpy as np
import pandas as pd
import statistics as stats
import rpy2.robjects as robjects

from datetime import datetime

from main import MSA

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
        self.real_outliers = None
        
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
            outliers_ms= robjects.r['my_ms'](self.saved_data, self.projections)
            self.outliers_ms = outliers_ms
        
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
        
        real_outliers = list(range(self.N + 1, int(self.N + self.N * self.contamination + 1), 1))
        self.real_outliers = real_outliers
    
    def metric(self):
        
        # Retrive results and turn into sets
        outliers_outliergram = set(list(self.outliers_outliergram))
        outliers_muod = set(list(self.outliers_muod))
        outliers_ms = set(list(self.outliers_ms))
        if len(self.index_outliers) == 0:
            outliers_msa = []
        else:
            outliers_msa = set([num + 1 for num in list(self.index_outliers[0])])
        real_outliers = set(self.real_outliers)
        
        # Calculate the length of each intersection and union
        intersection_outliergram = len(real_outliers.intersection(outliers_outliergram))
        union_outliergram = len(real_outliers.union(outliers_outliergram))
        intersection_muod = len(real_outliers.intersection(outliers_muod))
        union_muod = len(real_outliers.union(outliers_muod))
        intersection_ms = len(real_outliers.intersection(outliers_ms))
        union_ms = len(real_outliers.union(outliers_ms))
        intersection_msa = len(real_outliers.intersection(outliers_msa))
        union_msa = len(real_outliers.union(outliers_msa))
        
        # Get the Jaccard similarity indix for each method
        jaccard_index_outliergram = intersection_outliergram / union_outliergram if union_outliergram > 0 else 1.0
        jaccard_index_muod = intersection_muod / union_muod if union_muod > 0 else 1.0
        jaccard_index_ms = intersection_ms / union_ms if union_ms > 0 else 1.0
        jaccard_index_msa = intersection_msa / union_msa if union_msa > 0 else 1.0
        
        # Calculate the raw accuracy score for each method
        # accuracy_outliergram = intersection_outliergram / len(real_outliers)
        # accuracy_muod = intersection_muod / len(real_outliers)
        # accuracy_ms = intersection_ms / len(real_outliers)
        # accuracy_msa =intersection_msa / len(real_outliers)
        
        results = {'jaccard_outliergram': jaccard_index_outliergram,
                    'jaccard_muod': jaccard_index_muod,
                    'jaccard_ms': jaccard_index_ms,
                    'jaccard_msa': jaccard_index_msa}
                    # 'accuracy_outliergram': accuracy_outliergram,
                    # 'accuracy_muod': accuracy_muod,
                    # 'accuracy_ms': accuracy_ms,
                    # 'accuracy_msa': accuracy_msa}
        
        return results


if __name__ == '__main__':
    
    # Define dataframe to store the results
    df_results = pd.DataFrame(columns=['contamination', 'outliergram', 'muod', 'ms', 'msa'])
    
    # Define contamination levels:
    contaminations = [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    
    # Define lists to store the results and get their mean
    results_outliergram, results_muod, results_ms, results_msa = [], [], [], []
    
    for contamination in contaminations:
        
        for i in range(50):
            
            # Set up timer
            start = time.time()

            # Create an instance of the simulation class
            simulator_instance = simulator(simulation=True, N=200, L=6, P=96, projections=200, basis=48, detection_threshold=39.50, contamination=contamination, neighbors=10)

            # Generate synthetic data
            simulator_instance.call_generator()
            
            # Contaminate the synthetic data
            simulator_instance.call_contaminator()
            
            # Saved the generated data
            simulator_instance.call_saver()
            
            # Call the outliergram
            simulator_instance.call_outliergram()
            
            # Call MUOD
            simulator_instance.call_muod()
            
            # Call MS Dai Genton
            simulator_instance.call_ms()
            
            # Calculate magnitude, shape, and amplitude
            simulator_instance.call_msa()
            
            # Get the timestamps
            simulator_instance.get_timestamps()
            
            # Detect outliers if any
            simulator_instance.outlier_detector()
            
            # Extract real outliers
            simulator_instance.real_outdec()
            
            # Calculate accuracy
            results = simulator_instance.metric()
            
            results_outliergram.append(results['jaccard_outliergram'])
            results_muod.append(results['jaccard_muod'])
            results_ms.append(results['jaccard_ms'])
            results_msa.append(results['jaccard_msa'])
            
            print((time.time() - start) / (60), 'minutes elapsed')
        
        df_results.loc[len(df_results.index)] = [contamination, stats.mean(results_outliergram), stats.mean(results_muod), stats.mean(results_ms), stats.mean(results_msa)]
        
        # Clean the results lists
        results_outliergram, results_muod, results_ms, results_msa = [], [], [], []
    
    # Save the results
    df_results.to_csv('results.csv', sep=',', encoding='utf-8', index=True)
