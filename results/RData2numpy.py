import numpy as np
import rpy2.robjects as robjects

# Load the R object from the RData file
robjects.r['load']('outliers_MUOD_901.RData')
# my_data = robjects.r['outliers_outliergram']
my_data = robjects.r['outliers_muod']

# Convert the R object to a NumPy array
numpy_array = np.array(my_data)

# Save the NumPy array to a file (e.g., in .npy format)
np.save('results/outliers_MUOD_901.npy', numpy_array)
