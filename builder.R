library(lubridate)

# Function to get time data from the user
time_getter <- function() {

    total_timeunit <- as.integer(readline(prompt = "Enter the desired number of time units: "))

    x <- 0
    number_timeunit <- vector()

    while (x < total_timeunit) {

        data <- readline(prompt = "Enter each of the time units: ")

        number_timeunit <- c(number_timeunit, data)

        x <- x + 1
    }

    return(as.integer(number_timeunit))
}

builder <- function(time_frame, time_step, station, variables) {

    # Read the csv file
    df <- read.csv(paste("data/labeled_", station, "_pro_msa.csv", sep = ""), header = TRUE, sep = ",", stringsAsFactors = FALSE)

    # Convert the date column to datetime objects
    df$date <- as.POSIXct(df$date, format = "%Y-%m-%d %H:%M:%S")

    if (time_step == "15 min") {
        # Set number of row for 15 min data
        nrow_months <- 2976
        nrow_weeks <- 672
        nrow_days <- 96

    } else if (time_step == "1 day") {
        # Set the number of rows for daily data
        nrow_months <- 32
        nrow_weeks <- 8
    }

    # Subsetting the data.frame to create the list of matrices
    mts <- list()
    counter <- 1

    if (time_frame == "a") {

        print("Refactored code not implemented yet for time_frame a")

    } else if (time_frame == "b") {

        print("Refactored code not implemented yet for time_frame b")

    } else if (time_frame == "c") {

        # Loop through the data frame to create a matrix for each day
        for (i in 1:(nrow(df) / nrow_days)) {
            start_index <- (i - 1) * nrow_days + 1
            end_index <- i * nrow_days

            mat <- data.matrix(df[start_index:end_index, variables])

            mts$data[[counter]] <- mat
            mts$time[[counter]] <- c(day(df$date[start_index]), month(df$date[start_index]), year(df$date[start_index]))

            counter <- counter + 1

        }

    return(mts)

    }

}