# This file does the amplitude outlier detection

# library("reticulate")
# use_python("C:/Users/BAYESIA 2/OneDrive - Universidade de Vigo/1_Ph.D/1_Code/m-fdaPy/venv/Scripts/python.exe")

get_amplitude <- function(mts, station) {

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
        n_basis <- 32

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