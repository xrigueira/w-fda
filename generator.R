# This file uses the methods implemented in the "roahd" library
# to generate multivariate functional data

# https://cran.r-project.org/web/packages/roahd/index.html
# https://journal.r-project.org/archive/2019/RJ-2019-032/RJ-2019-032.pdf

library(roahd)

data_generator <- function() {

    # Define its parameters
    N <- 100 # Number of distintc functional observations
    L <- 6 # Number of components of the data
    P <- 1e2
    time_grid <- seq(0, 1, length.out = P)
    centerline <-  matrix(c(sin(2 * pi * time_grid), sqrt(time_grid), 10 * (time_grid - 0.5) * time_grid), nrow = 3, byrow = TRUE)

    # Continue here. Understand well how this works. At least its parameters

    generate_gauss_mfdata(N = N, L = L, centerline = centerline)
}