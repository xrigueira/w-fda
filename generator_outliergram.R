# This file uses the methods implemented in the "roahd" library
# to generate multivariate functional data

# https://cran.r-project.org/web/packages/roahd/index.html
# Contains uselful description of the methods for the paper https://journal.r-project.org/archive/2019/RJ-2019-032/RJ-2019-032.pdf
# Page 73 may have the key to add outliers

library(roahd)

set.seed(1)

data_generator <- function() {

    # Define its parameters
    N <- 100 # Number of distintc functional observations (number of days in my case: 1092)
    L <- 3 # Number of components of the data (number of variables)
    P <- 1e2 # Length of the series
    time_grid <- seq(0, 1, length.out = P)
    # Define the centerline for each variable
    centerline = matrix(c(sin(2 * pi * time_grid), sqrt(time_grid), 10 * (time_grid - 0.5) * time_grid), nrow = 3, byrow = TRUE)
    # Define the covariance for each variable
    Cov1 <- exp_cov_function(time_grid, alpha = 0.2, beta = 0.3)
    Cov2 <- exp_cov_function(time_grid, alpha = 0.1, beta = 0.5)
    Cov3 <- exp_cov_function(time_grid, alpha = 0.4, beta = 0.3)
    # Define the correlation for each variable
    corr1 <- 0.5
    corr2 <- 0.5
    corr3 <- 0.5

    mdata <- generate_gauss_mfdata(N = N, L = L, centerline = centerline, correlations = c(corr1, corr2, corr3), listCov = list(Cov1, Cov2, Cov3))

    mfD <- mfData(time_grid, mdata) # convert to S3 functional data object

    plot(mfD)

    # Univariate case
    # N <- 192
    # centerline <- sin(2 * pi * seq(0, 96, length.out = 96))
    # alpha <- 0.1
    # beta <- 0.1
    # Cov <- exp_cov_function(seq(0, 96, length.out = 96), alpha = alpha, beta = beta)
    
    # udata <- generate_gauss_fdata(N = N, centerline = centerline, Cov = Cov)

    # grid <- seq(0, 96, length.out = 96)
    # ufD <- fData(grid, data) # convert to S3 functional data object

    # plot(ufD)

    return(mfD)
}


data <- data_generator()

# Outliergram

dev.new()

out <- multivariate_outliergram(data, Fvalue = 2, shift = TRUE, display = TRUE)

N <- 100
col_non_outlying <- scales::hue_pal(h = c( 180, 270 ), l = 60 )(N - length(out$ID_outliers))
col_non_outlying <- set_alpha(col_non_outlying, 0.5)
col_outlying <- scales::hue_pal(h <- c( - 90, 180 ), c = 150 )(length(out$ID_outliers))
colors <- rep('black', N)
colors[out$ID_outliers] <- col_outlying
colors[colors == 'black'] <- col_non_outlying
lwd <- rep(1, N)
lwd[out$ID_outliers] <- 2

dev.new()

plot(data, col=colors, lwd=lwd)
