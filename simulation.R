# This file uses the methods implemented in the "roahd" library
# to generate multivariate functional data

# https://cran.r-project.org/web/packages/roahd/index.html
# Contains uselful description of the methods for the paper https://journal.r-project.org/archive/2019/RJ-2019-032/RJ-2019-032.pdf
# Page 73 may have the key to add outliers
# https://cran.r-project.org/web/packages/adamethods/index.html

library(fda)
library(roahd)
library(fdaoutlier)
library(dplyr)
library(tidyr)
library(vctrs)

data_generator <- function(N, L, P) {

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

data_contaminator <- function(N, data, contamination) {

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
    write.csv(df, file = "generated_data.csv", row.names = FALSE)

    return(df)

}

my_outliergram <- function(data) {

    # Define the configuration of the horizontal axis
    time_grid <- seq(0, 1, length.out = P)

    mfD <- mfData(time_grid, data) # convert to S3 functional data object

    plot(mfD)

    out <- multivariate_outliergram(mfD, Fvalue = 2, shift = TRUE, display = TRUE)

    return(out)

}

my_muod <- function(saved_df, P) {

    # dt1 <- simulation_model1()

    # Define an empty data frame
    new_data <- data.frame()

    # Iterate through each group and transform the data
    for (start_row in seq(1, nrow(saved_df), by = P)) {

        end_row <- min(start_row + P - 1, nrow(saved_df))

        group <- saved_df[start_row:end_row, ]

        transformed_group <- as.vector(unlist(group))

        # Convert the vector to a data frame with a single row
        transformed_row <- as.data.frame(t(transformed_group))

        # Append the transformed row to the new_data dataframe
        new_data <- rbind(new_data, transformed_row)

    }

    md <- muod(dts = new_data, cut_method = "boxplot")
    print(md$outliers)
    # print(md$indices)

}

# Define its parameters
N <- 200      # Number of distintc functional observations (number of days in my case: 1092)
L <- 6      # Number of components of the data (number of variables)
P <- 96    # Length of the series (legth of one day in my case: 96)

data <- data_generator(N, L, P)

data_contaminated <- data_contaminator(N, data, contamination = 0.05)

saved_df <- data_saver(N = nrow(data_contaminated[[1]]), L, P, data_contaminated)

outliers <- my_outliergram(data_contaminated)
print(outliers$ID_outliers)

my_muod(saved_df, P)
