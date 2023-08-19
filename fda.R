# This file contains the calculation of the magnitude,
# shape, amplitude of each function in the dataset.

# Supress warnings
options(warn = -1)

# Load the libraries
library(glue)
library(dplyr)
library(tidyr)
library(vctrs)
library(tidyverse)
library(reshape2)
library(lubridate)
library(fda)
library(fda.usc)
library(fdaoutlier)
library(roahd)

time_getter <- function() {

    # Gets time data from the user. Currently unused.

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

    # Reads the database and makes groups of months (a), weeks (b) or days (c).

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

builder_sim <- function(time_frame, time_step) {

    # Reads the database and makes groups of months (a), weeks (b) or days (c).

    # Read the csv file
    df <- read.csv(paste("data/generated_data.csv", sep = ""), header = TRUE, sep = ",", stringsAsFactors = FALSE)

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

            mat <- data.matrix(df[start_index:end_index, ])

            mts$data[[counter]] <- mat
            mts$time[[counter]] <- counter

            counter <- counter + 1

        }

    return(mts)

    }

}

get_amplitude <- function(mts, basis) {

    # Calculates the amplitude metric of a functional object.

    # Matrix to create the univariate fdata object
    mat <- matrix(ncol = nrow(mts$data[[1]]), nrow = length(mts$time))

    # Matrix that will contain the amplitudes of all curves of each variable
    mat_amplitudes <- matrix(ncol = length(mts$time), nrow = ncol(mts$data[[1]]))
    colnames(mat_amplitudes) <- mts$time

    number_variables <- seq_len(ncol(mts$data[[1]]))

    for (i in number_variables) {

        counter <- 1
        for (j in mts$data) {

            mat[counter, ] <- j[, i]
            counter <- counter + 1

        }

        # Insert the time stamps as column names
        rownames(mat) <- mts$time

        # Convert the matrix to fdata class (discrete data)
        mat_fdata <- fdata(mat)

        # Convert to functional data with a corr >= 0.95
        corr <- 0
        n_basis <- basis

        while (corr < 0.95) {

            mat_fd <- fdata2fd(mat_fdata, type.basis = "fourier", nbasis = n_basis)
            # Reverse fd object to fdata to be able to do the comparison in the test
            mat_fd2fdata <- fdata(mat_fd, argvals = mat_fdata$argvals)

            # Pearson correlation test
            result_pearson <- cor.test(mat_fdata$data, mat_fd2fdata$data, method = c("pearson"))

            corr <- result_pearson$estimate

            n_basis <- n_basis + 1

        }

        # Get the mean function
        mean_function <- mean.fd(mat_fd)

        # Convert to fdata object
        mean_function <- fdata(mean_function)

        # Convert mat_fd to fdata object
        functions <- fdata(mat_fd)

        # Create object to store the distances
        distances <- numeric(nrow(functions$data))

        # Assign column names to the distances vector
        names(distances) <- rownames(functions$data)

        # Calculate the distance for each function with respect to the mean function
        for (k in 1:nrow(functions$data)) {
            function_data <- functions$data[k, ]
            distance <- mean(abs(mean_function$data - function_data))
            distances[k] <- distance
        }

        mat_amplitudes[i, ] <- unname(distances)

        # Here I could save the functional plot for each variable

    }

    # Get multivariate amplitudes
    # Create an empty vector to store the dot product sums
    amplitudes <- numeric(ncol(mat_amplitudes))

    # Iterate over each column and calculate the dot product sum
    for (i in 1:ncol(mat_amplitudes)) {
        amplitudes[i] <- mean(mat_amplitudes[, i])
    }

    # Assign names and sort
    names(amplitudes) <- mts$time

    return(amplitudes)

}

get_magnitude_shape <- function(mts, projections) {

    # Calculates the magnitude and shape metrics of a functional object.

    # Determine the dimensions of the matrix
    n <- length(mts$data)
    p <- nrow(mts$data[[1]])
    d <- ncol(mts$data[[1]])

    # Combine the extracted columns into a single 3D array
    matrix_data <- array(unlist(mts), dim = c(p, d, n))

    matrix_data <- aperm(matrix_data, c(3, 1, 2))

    dirout <- dir_out(dts = matrix_data, n_projections = projections, seed = 0, return_distance = TRUE)

    magnitude <- apply(dirout$ms_matrix[, 1:d], 1, mean)
    magnitude_shape <- cbind(magnitude, dirout$ms_matrix[, d + 1])

    rownames(magnitude_shape) <- mts$time
    colnames(magnitude_shape) <- c("magnitude", "shape")

    return(magnitude_shape)

}

get_msa <- function(simulation, projections, basis) {

    # This function combines 'builder', 'get_amplitude',  'get_magnitude_shape'
    # and 'real_outdec' to extract the msa score of a functional object and
    # obtain the index of the real outliers marked in the database.

    # Get starting time
    start_time <- Sys.time()

    # Define the variables for the desired time units
    time_frame <- "c" # "a" for months, "b" for weeks, "c" for days
    time_step <- "15 min"
    station <- "901"
    variables <- c(paste("ammonium_", station, sep = ""),
                    paste("conductivity_", station, sep = ""),
                    paste("dissolved_oxygen_", station, sep = ""),
                    paste("pH_", station, sep = ""),
                    paste("turbidity_", station, sep = ""),
                    paste("water_temperature_", station, sep = "")
                )

    if (simulation == TRUE) {

        mts <- builder_sim(time_frame = time_frame, time_step = time_step)
        print("[INFO] mts obtained")

    } else if (simulation == FALSE) {

        # Get multivariate time series object (mts)
        mts <- builder(time_frame = time_frame, time_step = time_step, station = station, variables = variables)
        print("[INFO] mts obtained")

    }

    # Directional outlyingness (magnitude and shape)
    magnitude_shape <- get_magnitude_shape(mts, projections)
    print("[INFO] directional outlyingness obtained")

    # Amplitudes
    amplitude <- get_amplitude(mts, basis)
    print("[INFO] amplitude obtained")

    # Get magnitude, shape, and amplitude (msa) object by combining magnitude_shape and amplitude
    msa <- cbind(magnitude_shape, amplitude)
    print("[INFO] msa obtained")

    if (simulation == FALSE) {

        # Extract the real outliers to check the results
        real_outliers <- real_outdec(station)
        print("[INFO] ground truth outliers extracted")
        print(real_outliers)

    }


    # Get ending time
    end_time <- Sys.time()

    # Output time elapsed
    print(end_time - start_time)

    return(msa)

}

real_outdec <- function(station) {

    # Extracts the indexes of those time frames labeles as anomalous (1) in the database. 

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

data_generator <- function(N, L, P) {

    # This function generates synthetic data for simulation purposes.
    # Arguments:
    # N (int) <- number of distinct functional observations.
    # L (iny) <- number of variables of the data.
    # P (int) <- lenght of the series.

    # Define the seed
    set.seed(sample(1:1000, 1))

    # Define the configuration of the horizontal axis
    time_grid <- seq(0, 1, length.out = P)

    # Define the centerline for each variable
    centerline <- matrix(c(sin(2 * pi * time_grid),
                            sqrt(time_grid),
                            10 * (time_grid - 0.5) * time_grid,
                            cos(2 * pi * time_grid),
                            time_grid**2,
                            2 * time_grid + 1),
                            nrow = 6, byrow = TRUE)

    # Define the covariance for each variable
    Cov1 <- exp_cov_function(time_grid, alpha = 0.1, beta = 0.1)
    Cov2 <- exp_cov_function(time_grid, alpha = 0.1, beta = 0.1)
    Cov3 <- exp_cov_function(time_grid, alpha = 0.1, beta = 0.1)
    Cov4 <- exp_cov_function(time_grid, alpha = 0.1, beta = 0.1)
    Cov5 <- exp_cov_function(time_grid, alpha = 0.1, beta = 0.1)
    Cov6 <- exp_cov_function(time_grid, alpha = 0.1, beta = 0.1)

    # Define the correlation for each variable (3 for 3 var, 6 for 4 var, and 15 for 6 var)
    corr1 <- 0.5
    corr2 <- 0.5
    corr3 <- 0.5
    corr4 <- 0.5
    corr5 <- 0.5
    corr6 <- 0.5
    corr7 <- 0.5
    corr8 <- 0.5
    corr9 <- 0.5
    corr10 <- 0.5
    corr11 <- 0.5
    corr12 <- 0.5
    corr13 <- 0.5
    corr14 <- 0.5
    corr15 <- 0.5

    mdata <- generate_gauss_mfdata(N = N, L = L, centerline = centerline,
                                    correlations = c(corr1, corr2, corr3, corr4, corr5, corr6, corr7, corr8, corr9, corr10, corr11, corr12, corr13, corr14, corr15),
                                    listCov = list(Cov1, Cov2, Cov3, Cov4, Cov5, Cov6))

    return(mdata)
}

data_contaminator <- function(N, L, P, data, contamination) {

    # Adds a certain number of contaminates series to the generated data.
    # The number of contaminates series is defined by the contamination (float)
    # parameter.

    # Get the number of outliers based on the contamination
    if (contamination == 0) {

        N_outliers <- 0

        return(data)

    } else if (contamination != 0) {

        N_outliers <- round(N * contamination)

        # Define the seed
        set.seed(1)

        # Define the configuration of the horizontal axis
        time_grid <- seq(0, 1, length.out = P)

        # Define the centerline for each variable
        centerline <- matrix(c(sin(2 * pi * time_grid),
                                sqrt(time_grid),
                                10 * (time_grid - 0.5) * time_grid,
                                cos(2 * pi * time_grid),
                                time_grid**2,
                                2 * time_grid + 1),
                                nrow = 6, byrow = TRUE)

        # Define the covariance for each variable
        Cov1 <- exp_cov_function(time_grid, alpha = runif(1, min = 0.5, max = 0.9), beta = runif(1, min = 0.5, max = 0.9))
        Cov2 <- exp_cov_function(time_grid, alpha = runif(1, min = 0.5, max = 0.9), beta = runif(1, min = 0.5, max = 0.9))
        Cov3 <- exp_cov_function(time_grid, alpha = runif(1, min = 0.5, max = 0.9), beta = runif(1, min = 0.5, max = 0.9))
        Cov4 <- exp_cov_function(time_grid, alpha = runif(1, min = 0.5, max = 0.9), beta = runif(1, min = 0.5, max = 0.9))
        Cov5 <- exp_cov_function(time_grid, alpha = runif(1, min = 0.5, max = 0.9), beta = runif(1, min = 0.5, max = 0.9))
        Cov6 <- exp_cov_function(time_grid, alpha = runif(1, min = 0.5, max = 0.9), beta = runif(1, min = 0.5, max = 0.9))

        # Define the correlation for each variable (3 for 3 var, 6 for 4 var, and 15 for 6 var)
        corr1 <- 0.5
        corr2 <- 0.5
        corr3 <- 0.5
        corr4 <- 0.5
        corr5 <- 0.5
        corr6 <- 0.5
        corr7 <- 0.5
        corr8 <- 0.5
        corr9 <- 0.5
        corr10 <- 0.5
        corr11 <- 0.5
        corr12 <- 0.5
        corr13 <- 0.5
        corr14 <- 0.5
        corr15 <- 0.5

        mdata_contaminated <- generate_gauss_mfdata(N = N_outliers, L = L, centerline = centerline,
                                        correlations = c(corr1, corr2, corr3, corr4, corr5, corr6, corr7, corr8, corr9, corr10, corr11, corr12, corr13, corr14, corr15),
                                        listCov = list(Cov1, Cov2, Cov3, Cov4, Cov5, Cov6))


        # Combine the data object with the contaminated data object
        for (i in 1:length(data)) {

            data[[i]] <- rbind(data[[i]], mdata_contaminated[[i]])

        }

        return(data)

    }

}

data_saver <- function(N, L, P, data) {

    # Saved the generated data as a CSV file

    # Create an empty data frame
    df <- data.frame(matrix(nrow = N * P, ncol = L))

    # Iterate over each variable (matrix) in the list
    for (i in seq_along(data)) {

        values <- c()  # Initialize combined_values for each matrix

        # Iterate over each row in the matrix
        for (j in 1:nrow(data[[i]])) {

            values <- c(values, data[[i]][j, ])

        }

        # Add the concatenated values as a new column to the data frame
        df[[i]] <- values
    }

    # Save the generated data
    write.csv(df, file = "data/generated_data.csv", row.names = FALSE)

    return(df)

}

my_outliergram <- function(P, data) {

    # Implementation of the multivariate outliergram proposed by Ieva and Paganoni in 2020
    # https://cran.r-project.org/web/packages/roahd/index.html
    # https://journal.r-project.org/archive/2019/RJ-2019-032/RJ-2019-032.pdf

    # Define the configuration of the horizontal axis
    time_grid <- seq(0, 1, length.out = P)

    mfD <- mfData(time_grid, data) # convert to S3 functional data object

    # plot(mfD)

    results_outliegram <- multivariate_outliergram(mfD, Fvalue = 2, shift = TRUE, display = FALSE)

    return(results_outliegram$ID_outliers)

}

my_muod <- function(P, saved_data) {

    # Implementation of the MUOD algorithm for outlier detection in functional data
    # proposed by in Ojo in 2019.
    # https://link.springer.com/article/10.1007/s11634-021-00460-9

    # dt1 <- simulation_model1()

    # Define an empty data frame
    new_data <- data.frame()

    # Iterate through each group and transform the data
    for (start_row in seq(1, nrow(saved_data), by = P)) {

        end_row <- min(start_row + P - 1, nrow(saved_data))

        group <- saved_data[start_row:end_row, ]

        transformed_group <- as.vector(unlist(group))

        # Convert the vector to a data frame with a single row
        transformed_row <- as.data.frame(t(transformed_group))

        # Append the transformed row to the new_data dataframe
        new_data <- rbind(new_data, transformed_row)

    }

    results_muod <- muod(dts = new_data, cut_method = "boxplot")

    outliers <- sort(unique(Reduce(combine, results_muod$outliers)))

    return(outliers)

}

my_ms <- function(saved_data, projections) {

    # Determine the dimensions of the matrix
    p <- 96
    n <- nrow(saved_data) / p
    d <- ncol(saved_data)

    # Combine the extracted columns into a single 3D array
    matrix_data <- array(0, dim = c(n, p, d))

    # Fill the array with data from the data.frame
    for (i in 1:6) {

        matrix_data[, , i] <- matrix(saved_data[, i], ncol = 96, byrow = TRUE)

    }

    dirout <- dir_out(dts = matrix_data, n_projections = projections, seed = 0, return_distance = TRUE)

    # Extract the distance
    distance <- dirout$distance

    # Perform Min-Max scaling
    min_value <- min(distance)
    max_value <- max(distance)
    scaled_distance <- (distance - min_value) / (max_value - min_value)

    # Get the F distribution
    y_qf <- qf(scaled_distance, df1 = 2, df2 = 3)

    # Extract the index of those values above the quantile
    desired_quantile <- 0.993
    index_of_quantile <- which.min(abs(y_qf - desired_quantile))

    outliers <- which(distance >= distance[index_of_quantile])

    # Extract the outliers
    # outliers <- which(dirout$distance > (quantile(dirout$distance, probs = 0.95)))

    return(outliers)

}