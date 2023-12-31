# Get the accuracy of outliergram and muod on 901

# jaccard_msa_days -> 0.051470588
# accuracy_msa_days -> 41.17%
# jaccard_msa_4hours -> 0.008293461
# accuracy_msa_4hours -> 18.44%
# jaccard_outliergram -> 0
# accuracy_outliergram -> 0%
# jaccard_muod -> 0.07561437
# accuracy_muod -> 58.82%

source("fda.R")

# Read the data
station <- "910"

# Read the csv file
df <- read.csv(paste("data/labeled_", station, "_pro.csv", sep = ""), header = TRUE, sep = ",", stringsAsFactors = FALSE)

# Select the columns with variables
df <- df[, 2:7]

## Reshape data before feeding it into the outliergram
# Calculate the number of days
num_days <- nrow(df) / 16

# Create a list to hold the matrices
matrix_list <- list()

# Iterate over each variable
for (col_index in 1:6) {

    variable_data <- df[, col_index] # Extract the data for each variable
    variable_matrix <- matrix(variable_data, nrow = num_days, ncol = 16, byrow = TRUE) # Create the matrix
    matrix_list[[col_index]] <- variable_matrix  # Add the matrix to the list

}

outliers_outliergram <- my_outliergram(P = 16, matrix_list)
print(outliers_outliergram)

# Save outliergram results
save(outliers_outliergram, file="outliers_MOUT.RData")

# Implement MUOD
saved_df <- data_saver(N = num_days, L = 6, P = 16, data = matrix_list)

outliers_muod <- my_muod(P = 16, saved_data = saved_df)
print(outliers_muod)

# Save MUOD results
save(outliers_muod, file="outliers_MUOD.RData")

# real_outliers <- real_outdec(station)
# print(real_outliers)

# Get Jaccard Index and Accuracy score
# intersection_outliergram <- length(intersect(real_outliers, outliers_outliergram))
# union_outliergram <- length(union(real_outliers, outliers_outliergram))

# intersection_muod <- length(intersect(real_outliers, outliers_muod))
# union_muod <- length(union(real_outliers, outliers_muod))

# jaccard_outliergram <- ifelse(union_outliergram > 0, intersection_outliergram / union_outliergram, 1.0)
# jaccard_muod <- ifelse(union_muod > 0, intersection_muod / union_muod, 1.0)

# accuracy_outliergram <- intersection_outliergram / length(real_outliers)
# accuracy_muod <- intersection_muod / length(real_outliers)

# print(jaccard_outliergram)
# print(jaccard_muod)

# print(accuracy_outliergram)
# print(accuracy_muod)