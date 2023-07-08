# This file contains the calculation of the magnitude,
# shape, amplitude of each function in the dataset.

# Supress warnings
options(warn=-1)

# Load the libraries
library(tidyverse)
library(glue)
library(reshape2)
library(fdaoutlier)
library(fda)
library(fda.usc)

# Load the files
source("builder.R")
source("msCalc.R")
source("aCalc.R")
source("real-outdec.R")

get_msa <- function() {

    # Get starting time
    start_time <- Sys.time()

    # Define the variables for the desired time units
    time_frame <- "c" # "a" for months, "b" for weeks, "c" for days
    span <- "a" # This variable is to select different combinations later
    time_step <- "15 min"
    station <- "901"
    variables <- c(paste("ammonium_", station, sep = ""),
                    paste("conductivity_", station, sep = ""),
                    paste("dissolved_oxygen_", station, sep = ""),
                    paste("pH_", station, sep = ""),
                    paste("turbidity_", station, sep = ""),
                    paste("water_temperature_", station, sep = "")
                )

    # Get multivariate time series object (mts)
    mts <- builder(time_frame = time_frame, span = span, time_step = time_step, station = station, variables = variables)
    print("[INFO] mts obtained")

    # Directional outlyingness (magnitude and shape)
    magnitude_shape <- get_magnitude_shape(mts)
    print("[INFO] dirrectional outlyingness obtained")

    # Amplitudes
    amplitude <- get_amplitude(mts, station)
    print("[INFO] amplitude obtained")

    # Get magnitude, shape, and amplitude (msa) object by combining magnitude_shape and amplitude
    msa <- cbind(magnitude_shape, amplitude)
    print("[INFO] msa obtained")

    # # Extract the real outliers to check the results
    # real_outliers <- real_outdec(mts, station)
    # print("[INFO] ground truth outliers extracted")

    # Get ending time
    end_time <- Sys.time()

    # Output time elapsed
    print(end_time - start_time)

    return(msa)

}