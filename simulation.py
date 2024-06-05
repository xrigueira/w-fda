import csv
import time
import numpy as np
import pandas as pd
import statistics as stats
import rpy2.robjects as robjects

from datetime import datetime

from main import MSA

class simulator(MSA):
    
    def __init__(self, station, hours, nhours, simulation, N, L, P, projections, basis, detection_threshold, contamination, neighbors) -> None:
        
        self.station = station
        self.hours = hours
        self.nhours = nhours
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
            msa = robjects.r['get_msa'](self.station, self.hours, self.nhours, self.simulation, self.projections, self.basis)
            
            # Convert and save the result to a numpy.ndarray
            msa = np.array(msa)
            self.msa = msa # Store the result in the instance variable
            np.save('msa.npy', msa, allow_pickle=False, fix_imports=False) # Remove when the program is finished
    
    def get_timestamps(self):
        
        timestamps = np.arange(1, len(self.msa) + 1, 1)
        self.timestamps = timestamps
    
    def real_outdec(self):
        
        real_outliers = list(range(self.N + 1, int(self.N + self.N * self.contamination + 1), 1))
        self.real_outliers = real_outliers
    
    def metric(self):
        
        # Retrive results
        outliers_outliergram = list(self.outliers_outliergram)
        outliers_muod = list(self.outliers_muod)
        outliers_ms = list(self.outliers_ms)
        
        if len(self.index_outliers) == 0:
            outliers_msa = []
        else:
            outliers_msa = list([num + 1 for num in list(self.index_outliers[0])])
        
        real_outliers = self.real_outliers

        # Convert the results to binary lists of len 210 (N+10)
        outliers_outliergram = [1 if i in outliers_outliergram else 0 for i in range(1, self.N + 11)]
        outliers_muod = [1 if i in outliers_muod else 0 for i in range(1, self.N + 11)]
        outliers_ms = [1 if i in outliers_ms else 0 for i in range(1, self.N + 11)]
        outliers_msa = [1 if i in outliers_msa else 0 for i in range(1, self.N + 11)]
        real_outliers = [1 if i in real_outliers else 0 for i in range(1, self.N + 11)]

        # Calculate the accuracy score for each method
        accuracy_outliergram = sum([1 if outliers_outliergram[i] == real_outliers[i] else 0 for i in range(len(outliers_outliergram))]) / len(real_outliers)
        accuracy_muod = sum([1 if outliers_muod[i] == real_outliers[i] else 0 for i in range(len(outliers_muod))]) / len(real_outliers)
        accuracy_ms = sum([1 if outliers_ms[i] == real_outliers[i] else 0 for i in range(len(outliers_ms))]) / len(real_outliers)
        accuracy_msa = sum([1 if outliers_msa[i] == real_outliers[i] else 0 for i in range(len(outliers_msa))]) / len(real_outliers)

        # Calculate the precision score for each method
        precision_outliergram = sum([1 if outliers_outliergram[i] == real_outliers[i] else 0 for i in range(len(outliers_outliergram))]) / len(real_outliers)
        precision_muod = sum([1 if outliers_muod[i] == real_outliers[i] else 0 for i in range(len(outliers_muod))]) / len(real_outliers)
        precision_ms = sum([1 if outliers_ms[i] == real_outliers[i] else 0 for i in range(len(outliers_ms))]) / len(real_outliers)
        precision_msa = sum([1 if outliers_msa[i] == real_outliers[i] else 0 for i in range(len(outliers_msa))]) / len(real_outliers)

        # Calculate the recall score for each method
        recall_outliergram = sum([1 if outliers_outliergram[i] == real_outliers[i] else 0 for i in range(len(outliers_outliergram))]) / len(outliers_outliergram)
        recall_muod = sum([1 if outliers_muod[i] == real_outliers[i] else 0 for i in range(len(outliers_muod))]) / len(outliers_muod)
        recall_ms = sum([1 if outliers_ms[i] == real_outliers[i] else 0 for i in range(len(outliers_ms))]) / len(outliers_ms)
        recall_msa = sum([1 if outliers_msa[i] == real_outliers[i] else 0 for i in range(len(outliers_msa))]) / len(outliers_msa)

        # Calculate the F1 score for each method
        f1_outliergram = 2 * (precision_outliergram * recall_outliergram) / (precision_outliergram + recall_outliergram)
        f1_muod = 2 * (precision_muod * recall_muod) / (precision_muod + recall_muod)
        f1_ms = 2 * (precision_ms * recall_ms) / (precision_ms + recall_ms)
        f1_msa = 2 * (precision_msa * recall_msa) / (precision_msa + recall_msa)

        # Calculate the error rate
        error_rate_outliergram = 1 - sum([1 if outliers_outliergram[i] == real_outliers[i] else 0 for i in range(len(outliers_outliergram))]) / len(real_outliers)
        error_rate_muod = 1 - sum([1 if outliers_muod[i] == real_outliers[i] else 0 for i in range(len(outliers_muod))]) / len(real_outliers)
        error_rate_accuracy_ms = 1 - sum([1 if outliers_ms[i] == real_outliers[i] else 0 for i in range(len(outliers_ms))]) / len(real_outliers)
        error_rate_accuracy_msa = 1 - sum([1 if outliers_msa[i] == real_outliers[i] else 0 for i in range(len(outliers_msa))]) / len(real_outliers)

        # Calculate the Jaccard similarity index
        intersection_outliergram = len(set(outliers_outliergram).intersection(set(real_outliers)))
        union_outliergram = len(set(outliers_outliergram).union(set(real_outliers)))
        jaccard_index_outliergram = intersection_outliergram / union_outliergram if union_outliergram > 0 else 1.0
        
        intersection_muod = len(set(outliers_muod).intersection(set(real_outliers)))
        union_muod = len(set(outliers_muod).union(set(real_outliers)))
        jaccard_index_muod = intersection_muod / union_muod if union_muod > 0 else 1.0

        intersection_ms = len(set(outliers_ms).intersection(set(real_outliers)))
        union_ms = len(set(outliers_ms).union(set(real_outliers)))
        jaccard_index_ms = intersection_ms / union_ms if union_ms > 0 else 1.0

        intersection_msa = len(set(outliers_msa).intersection(set(real_outliers)))
        union_msa = len(set(outliers_msa).union(set(real_outliers)))
        jaccard_index_msa = intersection_msa / union_msa if union_msa > 0 else 1.0
        
        # # Retrive results and turn into sets
        # outliers_outliergram = set(list(self.outliers_outliergram))
        # outliers_muod = set(list(self.outliers_muod))
        # print(list(self.outliers_ms))
        # outliers_ms = set(list(self.outliers_ms))
        # if len(self.index_outliers) == 0:
        #     outliers_msa = []
        # else:
        #     outliers_msa = set([num + 1 for num in list(self.index_outliers[0])])
        # real_outliers = set(self.real_outliers)
        
        # # Calculate the length of each intersection and union
        # intersection_outliergram = len(real_outliers.intersection(outliers_outliergram))
        # union_outliergram = len(real_outliers.union(outliers_outliergram))
        # intersection_muod = len(real_outliers.intersection(outliers_muod))
        # union_muod = len(real_outliers.union(outliers_muod))
        # intersection_ms = len(real_outliers.intersection(outliers_ms))
        # union_ms = len(real_outliers.union(outliers_ms))
        # intersection_msa = len(real_outliers.intersection(outliers_msa))
        # union_msa = len(real_outliers.union(outliers_msa))

        # # Calculate the length of each intersection and union
        # intersection_outliergram = len(real_outliers.intersection(outliers_outliergram))
        # union_outliergram = len(real_outliers.union(outliers_outliergram))
        # intersection_muod = len(real_outliers.intersection(outliers_muod))
        # union_muod = len(real_outliers.union(outliers_muod))
        # intersection_ms = len(real_outliers.intersection(outliers_ms))
        # union_ms = len(real_outliers.union(outliers_ms))
        # intersection_msa = len(real_outliers.intersection(outliers_msa))
        # union_msa = len(real_outliers.union(outliers_msa))
        
        # # Get the Jaccard similarity indix for each method
        # jaccard_index_outliergram = intersection_outliergram / union_outliergram if union_outliergram > 0 else 1.0
        # jaccard_index_muod = intersection_muod / union_muod if union_muod > 0 else 1.0
        # jaccard_index_ms = intersection_ms / union_ms if union_ms > 0 else 1.0
        # jaccard_index_msa = intersection_msa / union_msa if union_msa > 0 else 1.0
        
        results = {'accuracy_outliergram': accuracy_outliergram,
                    'accuracy_muod': accuracy_muod,
                    'accuracy_ms': accuracy_ms,
                    'accuracy_msa': accuracy_msa,
                    'precision_outliergram': precision_outliergram,
                    'precision_muod': precision_muod,
                    'precision_ms': precision_ms,
                    'precision_msa': precision_msa,
                    'recall_outliergram': recall_outliergram,
                    'recall_muod': recall_muod,
                    'recall_ms': recall_ms,
                    'recall_msa': recall_msa,
                    'f1_outliergram': f1_outliergram,
                    'f1_muod': f1_muod,
                    'f1_ms': f1_ms,
                    'f1_msa': f1_msa,
                    'error_rate_outliergram': error_rate_outliergram,
                    'error_rate_muod': error_rate_muod,
                    'error_rate_accuracy_ms': error_rate_accuracy_ms,
                    'error_rate_accuracy_msa': error_rate_accuracy_msa,
                    'jaccard_outliergram': jaccard_index_outliergram,
                    'jaccard_muod': jaccard_index_muod,
                    'jaccard_ms': jaccard_index_ms,
                    'jaccard_msa': jaccard_index_msa,
                    
                    }
        
        return results


if __name__ == '__main__':
    
    # Define dataframe to store the results
    df_accuracy = pd.DataFrame(columns=['contamination', 'outliergram', 'muod', 'ms', 'msa'])
    df_precision = pd.DataFrame(columns=['contamination', 'outliergram', 'muod', 'ms', 'msa'])
    df_recall = pd.DataFrame(columns=['contamination', 'outliergram', 'muod', 'ms', 'msa'])
    df_f1 = pd.DataFrame(columns=['contamination', 'outliergram', 'muod', 'ms', 'msa'])
    df_error_rate = pd.DataFrame(columns=['contamination', 'outliergram', 'muod', 'ms', 'msa'])
    df_jaccard = pd.DataFrame(columns=['contamination', 'outliergram', 'muod', 'ms', 'msa'])
    
    # Define contamination levels:
    contaminations = [0, 0.01, 0.02, 0.05, 0.1, 0.2]
    
    # Define lists to store the results and get their mean
    accuracy_outliergram, accuracy_muod, accuracy_ms, accuracy_msa = [], [], [], []
    precision_outliergram, precision_muod, precision_ms, precision_msa = [], [], [], []
    recall_outliergram, recall_muod, recall_ms, recall_msa = [], [], [], []
    f1_outliergram, f1_muod, f1_ms, f1_msa = [], [], [], []
    error_rate_outliergram, error_rate_muod, error_rate_ms, error_rate_msa = [], [], [], []
    jaccard_outliergram, jaccard_muod, jaccard_ms, jaccard_msa = [], [], [], []
    
    for contamination in contaminations:
        
        for i in range(100):
            
            # Set up timer
            start = time.time()

            # Create an instance of the simulation class
            simulator_instance = simulator(station=901, hours=False, nhours=24, simulation=True, 
                                        N=200, L=6, P=96, projections=200, basis=48, 
                                        detection_threshold=39.50, contamination=contamination, neighbors=10)

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
            
            # Append the different resutls
            accuracy_outliergram.append(results['accuracy_outliergram'])
            accuracy_muod.append(results['accuracy_muod'])
            accuracy_ms.append(results['accuracy_ms'])
            accuracy_msa.append(results['accuracy_msa'])

            precision_outliergram.append(results['precision_outliergram'])
            precision_muod.append(results['precision_muod'])
            precision_ms.append(results['precision_ms'])
            precision_msa.append(results['precision_msa'])

            recall_outliergram.append(results['recall_outliergram'])
            recall_muod.append(results['recall_muod'])
            recall_ms.append(results['recall_ms'])
            recall_msa.append(results['recall_msa'])

            f1_outliergram.append(results['f1_outliergram'])
            f1_muod.append(results['f1_muod'])
            f1_ms.append(results['f1_ms'])
            f1_msa.append(results['f1_msa'])

            error_rate_outliergram.append(results['error_rate_outliergram'])
            error_rate_muod.append(results['error_rate_muod'])
            error_rate_ms.append(results['error_rate_accuracy_ms'])
            error_rate_msa.append(results['error_rate_accuracy_msa'])

            jaccard_outliergram.append(results['jaccard_outliergram'])
            jaccard_muod.append(results['jaccard_muod'])
            jaccard_ms.append(results['jaccard_ms'])
            jaccard_msa.append(results['jaccard_msa'])
            
            print((time.time() - start) / (60), 'minutes elapsed')
        
        df_accuracy.loc[len(df_accuracy.index)] = [contamination, stats.mean(accuracy_outliergram), stats.mean(accuracy_muod), stats.mean(accuracy_ms), stats.mean(accuracy_msa)]
        df_precision.loc[len(df_precision.index)] = [contamination, stats.mean(precision_outliergram), stats.mean(precision_muod), stats.mean(precision_ms), stats.mean(precision_msa)]
        df_recall.loc[len(df_recall.index)] = [contamination, stats.mean(recall_outliergram), stats.mean(recall_muod), stats.mean(recall_ms), stats.mean(recall_msa)]
        df_f1.loc[len(df_f1.index)] = [contamination, stats.mean(f1_outliergram), stats.mean(f1_muod), stats.mean(f1_ms), stats.mean(f1_msa)]
        df_error_rate.loc[len(df_error_rate.index)] = [contamination, stats.mean(error_rate_outliergram), stats.mean(error_rate_muod), stats.mean(error_rate_ms), stats.mean(error_rate_msa)]
        df_jaccard.loc[len(df_jaccard.index)] = [contamination, stats.mean(jaccard_outliergram), stats.mean(jaccard_muod), stats.mean(jaccard_ms), stats.mean(jaccard_msa)]
        
        # Clean the results lists
        accuracy_outliergram, accuracy_muod, accuracy_ms, accuracy_msa = [], [], [], []
        precision_outliergram, precision_muod, precision_ms, precision_msa = [], [], [], []
        recall_outliergram, recall_muod, recall_ms, recall_msa = [], [], [], []
        f1_outliergram, f1_muod, f1_ms, f1_msa = [], [], [], []
        error_rate_outliergram, error_rate_muod, error_rate_ms, error_rate_msa = [], [], [], []
        jaccard_outliergram, jaccard_muod, jaccard_ms, jaccard_msa = [], [], [], []
    
    # Save the results
    df_accuracy.to_csv('results/accuracy.csv', sep=',', encoding='utf-8', index=True)
    df_precision.to_csv('results/precision.csv', sep=',', encoding='utf-8', index=True)
    df_recall.to_csv('results/recall.csv', sep=',', encoding='utf-8', index=True)
    df_f1.to_csv('results/f1.csv', sep=',', encoding='utf-8', index=True)
    df_error_rate.to_csv('results/error_rate.csv', sep=',', encoding='utf-8', index=True)
    df_jaccard.to_csv('results/jaccard.csv', sep=',', encoding='utf-8', index=True)
