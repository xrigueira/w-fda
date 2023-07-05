# This file contains the implementation of the ms-outdec
# algorithm, which is a magnitude and shape outlier detector.
# The shape component is based on the CCR-periodogram proposed
# by Lopez-Oriona 2021, and the magnitude component is obtained
# with a weak multivariate version of the Fraiman-Muñiz depth.

# Get starting time
start_time <- Sys.time()

# Load the libraries
library(tidyverse)
library(glue)
library(reshape2)
library(mlmts)
library(fda)
library(fda.usc)

# Load the files
source("builder.R")
source("sha-outdec.R")
source("mag-outdec.R")
source("amp-outdec.R")
source("glob-outdec.R")
source("real-outdec.R")
source("accuracy.R")
source("u-plotter.R")
source("m-plotter.R")
source("inter_u-plotter.R")
source("fda_u-plotter.R")

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

# Call the functions to get the results
mts <- builder(time_frame = time_frame, span = span, time_step = time_step, station = station, variables = variables)
print("[INFO] mts obtained")

# Shape depth
shape_depth <- shape_outdec(mts)
print("[INFO] shape depth obtained")

# Magnitude depth
magnitude_depth <- magnitude_outdec(mts, station)
print("[INFO] magnitude depth obtained")

# Amplitudes
amplitudes <- amplitude_outdec(mts, station)
print("[INFO] amplitude obtained")

# Global depth (combination of magnitude and shape)
global_depth <- global_outdec(mts, shape_depth, magnitude_depth, amplitudes)
print("[INFO] global depth obtained")

# Define the outliers. Needs reviewing. Highest amplitudes are the outliers and
# not the smallest
outliers <- global_depth[global_depth < quantile(global_depth, probs = c(0.10))]
print("[INFO] outliers detected")

# Extract the real outliers to check the results
real_outliers <- real_outdec(mts, station)
print("[INFO] ground truth outliers extracted")

Check accuracy
matching_outliers <- accuracy(outliers, real_outliers)
print(paste("Number matching outleirs", matching_outliers))

# Plot the results
uni_graphic <- u_plotter(mts, outliers, variable_index = 1, variables) # univariate results
multi_graphic <- m_plotter(mts, time_unit = 1) # multivariate results
inter_uni_graphic <- inter_u_plotter(mts, outliers, variable_index = 1, variables) # interactive univariate
fda_graphic <- fda_u_plotter(mts, outliers, variable_index = 1, variables) # univariate plot

# Get ending time
end_time <- Sys.time()

# Output time elapsed
print(end_time - start_time)