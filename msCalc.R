get_magnitude_shape <- function(mts){

    # Determine the dimensions of the matrix
    n <- length(mts$data)
    p <- length(mts$data[[1]])
    d <- length(mts$data[[1]][[1]])

    # Create an empty matrix
    matrix_data <- array(NA, dim = c(n, p, d))

    # Populate the matrix with the extracted data
    for (i in 1:n) {
        for (j in 1:p) {
            for (k in 1:d) {
                matrix_data[i, j, k] <- mts$data[[i]][[j]][[k]]
            }
        }
    }

    dirout <- dir_out(matrix_data, return_distance = TRUE)

    magnitude_shape <- dirout$ms_matrix

    rownames(magnitude_shape) <- mts$time
    colnames(magnitude_shape) <- c("magnitude", "shape")

    return(magnitude_shape)

}
