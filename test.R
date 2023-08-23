# Get the accuracy of outliergram and muod on 901

# msa -> 41%
# outliergram -> %
# muod -> %

source("fda.R")

real_outdec <- function(station) {

    # Extracts the indexes of those time frames labeles as anomalous (1) in the database. 

    # Read the csv file
    data <- read.csv(paste("data/labeled_", station, "_pro.csv", sep = ""), header = TRUE, sep = ",", stringsAsFactors = FALSE)

    # Convert the 'date' column to a proper date format
    data$date <- as.Date(data$date)

    # Group the data by 'date' and calculate the average of the 'label' column within each group
    average_label_per_day <- data %>%
        group_by(date) %>%
        summarize(average_label = mean(label)) %>%
        ungroup() %>%
        arrange(match(date, data$date))

    # Store the average values in a numeric vector
    average_labels <- average_label_per_day$average_label

    # Assign names to the numeric vector in the desired format
    names(average_labels) <- as.character(average_label_per_day$date)

    # Thresholding operation: values above 0.1 set to 1, rest set to 0
    average_labels <- ifelse(average_labels >= 0.1, 1, 0)

    # Get the index of the days labeled as 1
    outliers_indexes <- which(average_labels == 1)

    return(outliers_indexes)

}

# Read the data
station <- "901"

# Read the csv file
df <- read.csv(paste("data/labeled_", station, "_pro.csv", sep = ""), header = TRUE, sep = ",", stringsAsFactors = FALSE)

# Select the columns with variables
df <- df[, 2:7]

## Reshape data before feeding it into the outliergram
# Calculate the number of days
num_days <- nrow(df) / 96

# Create a list to hold the matrices
matrix_list <- list()

# Iterate over each variable
for (col_index in 1:6) {

    variable_data <- df[, col_index] # Extract the data for each variable
    variable_matrix <- matrix(variable_data, nrow = num_days, ncol = 96, byrow = TRUE) # Create the matrix
    matrix_list[[col_index]] <- variable_matrix  # Add the matrix to the list

}

outliers_outliergram <- my_outliergram(P = 96, matrix_list)
print(outliers_outliergram)

# Implement MUOD
# Continue here:
# 1. Use data saver to adapt the data for MUOD
# 2. Get the results from MUOD
# 3. Implement the Jaccard Index and Accuract metrics.

# real_outliers <- real_outdec(station)
# print(real_outliers)

# data <- data_generator(N = 20, L = 6, P = 10)
