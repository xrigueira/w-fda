# This file contains the code to run the MOUT and MUOD algorithms on the data of a given station

source("fda.R")

# Read the data
station <- "901"

# Read the csv file
df <- read.csv(paste("data/labeled_", station, "_pro.csv", sep = ""), header = TRUE, sep = ",", stringsAsFactors = FALSE)

# Select the columns with variables
df <- df[, 2:7]


# Reshape data before feeding it into MOUT
num_days <- nrow(df) / 32 # Calculate the number of days

# Create a list to hold the matrices
matrix_list <- list()

# Iterate over each variable
for (col_index in 1:6) {

    variable_data <- df[, col_index] # Extract the data for each variable
    variable_matrix <- matrix(variable_data, nrow = num_days, ncol = 32, byrow = TRUE) # Create the matrix
    matrix_list[[col_index]] <- variable_matrix  # Add the matrix to the list

}

# Implement MOUT
outliers_outliergram <- my_outliergram(P = 32, matrix_list)
print(outliers_outliergram)

# Save outliergram results
save(outliers_outliergram, file="indices_outliers_MOUT.RData")

# Implement MUOD
saved_df <- data_saver(N = num_days, L = 6, P = 32, data = matrix_list)

outliers_muod <- my_muod(P = 32, saved_data = saved_df)
print(outliers_muod)

# Save MUOD results
save(outliers_muod, file="indices_outliers_MUOD.RData")