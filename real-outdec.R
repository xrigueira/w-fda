# This file extracts the real outliers

real_outdec <- function(mts, station) {

    # Read the csv file
    df <- read.csv(paste("data/labeled_", station, "_pro.csv", sep = ""), header = TRUE, sep = ",", stringsAsFactors = FALSE)

    # Convert the 'date' column to a proper date format
    df$date <- as.Date(df$date)

    # Calculate the average label per day
    average_label <- aggregate(label ~ date, df, mean)

    # Store the average values in a numeric vector
    average_labels <- average_label$label

    # Assign names to the numeric vector in the desired format
    names(average_labels) <- as.character(average_label$date)

    # Thresholding operation: values above 0.5 set to 1, rest set to 0
    average_labels <- ifelse(average_labels >= 0.1, 1, 0)

    # Get the index of the days labeled as 1
    outliers_indexes <- which(average_labels == 1)

    # Return the resulting numeric object with named values
    return(outliers_indexes)

}