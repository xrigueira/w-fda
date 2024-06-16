import numpy as np
import rpy2.robjects as robjects

"""Load the RData file and convert the R object to a binarized NumPy array.
"""
station = 907
method = 'MS' # 'MOUT', 'MUOD' or 'MS'

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

elif method == 'MS':

    # Load the R object from the RData file
    robjects.r['load'](f'indices_outliers_MS.RData')

    # Extract the data
    my_data = robjects.r['outliers_ms']
    
# Convert the R object to a NumPy array
numpy_array = np.array(my_data)

# Adjust to 0 indexing (R is 1 indexed)
numpy_array -= 1

# Binarize the NumPy array
if station == 901:
    total_length_data = 15096 # Len msa obtained and printed in main
elif station == 905:
    total_length_data = 12015
elif station == 906:
    total_length_data = 16086
elif station == 907:
    total_length_data = 14175
y = np.zeros(total_length_data)
y[numpy_array] = 1

# Save the NumPy array to a file (e.g., in .npy format)
np.save(f'results/y_{method.lower()}_{station}.npy', y, allow_pickle=False, fix_imports=False)
