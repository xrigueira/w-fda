# Try to run get_msa in this Python file and print its results
# or put them in a Pandas database

import numpy as np
import rpy2.robjects as robjects

from outDec import outDec

# # Load the R function from msaCalc.R
# with open('msaCalc.R', 'r') as file:
#     r_code = file.read()

# # Execute the R function get_msa()
# robjects.r(r_code)
# msa = robjects.r['get_msa']()

# # Convert and save the result to a numpy.ndarray
# msa = np.array(msa)
# np.save('msa.npy', msa, allow_pickle=False, fix_imports=False)

# Check if there are outliers in the data
msa = np.load('msa.npy')
magnitude = msa[:, 0]
shape = msa[:, 1]
amplitude = msa[:, 2]

outliers_in_data = outDec(magnitude=magnitude, shape=shape, amplitude=amplitude)


