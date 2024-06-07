import numpy as np
import rpy2.robjects as robjects

"""Load the RData file and convert the R object to a binarized NumPy array.
"""
method = 'MUOD' # 'MOUT' or 'MUOD'

if method == 'MOUT':

    # Load the R object from the RData file
    robjects.r['load'](f'indices_outliers_MOUT.RData') 

    # Extract the data
    my_data = robjects.r['outliers_outliergram']

elif method == 'MUOD':

    # Load the R object from the RData file
    robjects.r['load'](f'indices_outliers_MUOD.RData')

    # Extract the data
    my_data = robjects.r['outliers_muod']
    
# Convert the R object to a NumPy array
numpy_array = np.array(my_data)

# Adjust to 0 indexing (R is 1 indexed)
numpy_array -= 1

# Binarize the NumPy array
total_length_data = 15096
y = np.zeros(total_length_data)
y[numpy_array] = 1

# Save the NumPy array to a file (e.g., in .npy format)
np.save(f'results/y_{method.lower()}.npy', numpy_array)
