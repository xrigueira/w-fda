# This file extracts the real outliers

real_outdec <- function(station) {

    # Read the csv file
    data <- read.csv(paste("data/labeled_", station, "_pro_msa.csv", sep = ""), header = TRUE, sep = ",", stringsAsFactors = FALSE)

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