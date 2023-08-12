# I am going to try to generate artificial data and contaminate it
#  based on what I have built already


data_contaminator <- function(N, data, contamination) {

    # Get number of outliers
    if (contamination == 0) {

        N_outliers <- 0
        print(N_outliers)

        return(data)

    } else if (contamination != 0) {

        N_outliers <- floor(N * contamination)
        print(N_outliers)

        # Do the rest

        return(data)

    }

}

N <- 200
data <- c(matrix())
data_contaminator(N, data, contamination = 0.20)
