# This file uses the methods implemented in the "roahd" library
# to generate multivariate functional data

# https://cran.r-project.org/web/packages/roahd/index.html
# Contains uselful description of the methods for the paper https://journal.r-project.org/archive/2019/RJ-2019-032/RJ-2019-032.pdf
# Page 73 may have the key to add outliers

library(fda)
library(roahd)
library(adamethods)

source("builder.R")

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
    Cov1 <- exp_cov_function(time_grid, alpha = 0.2, beta = 0.2)
    Cov2 <- exp_cov_function(time_grid, alpha = 0.2, beta = 0.2)
    Cov3 <- exp_cov_function(time_grid, alpha = 0.2, beta = 0.2)
    Cov4 <- exp_cov_function(time_grid, alpha = 0.2, beta = 0.2)
    Cov5 <- exp_cov_function(time_grid, alpha = 0.2, beta = 0.2)
    Cov6 <- exp_cov_function(time_grid, alpha = 0.2, beta = 0.2)

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
    write.csv(df, file = "data.csv", row.names = FALSE)

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

my_fadalara <- function(saved_df) {

    # Start from saved_df and create basis (line 27 fadalara)

    n <- nrow(saved_df)

    # Number of archetypoids:
    k <- 3
    numRep <- 20
    huge <- 200

    # Size of the random sample of observations:
    m <- 15

    # Number of samples:
    N_samples <- floor(1 + (n - m) / (m - k))
    prob <- 0.75

    # Read the generated data
    library(doParallel)
    no_cores <- detectCores() - 1
    cluster <- makeCluster(no_cores)
    registerDoParallel(cl)
    clusterSetRNGStream(cl, iseed = 1)

    results <- fadalara(data = saved_df, N = N_samples, m = m, numArchoid = k, numRep = numRep,
                        huge = huge, prob = prob, type_alg = "fada_rob", compare = FALSE,
                        PM = PM, method = "adjbox", multiv = TRUE, frame = FALSE)

    stopCluster(cluster)
}


# Define its parameters
N <- 5      # Number of distintc functional observations (number of days in my case: 1092)
L <- 6      # Number of components of the data (number of variables)
P <- 16     # Length of the series

data <- data_generator(N, L, P)

saved_df <- data_saver(N, L, P, data)

# outliers <- my_outliergram(data)
# print(outliers$ID_outliers)

my_fadalara(saved_df)
