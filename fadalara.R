# This file uses the methods from the "adamethods" library to implement
# the fadalara method for multivariate data

# https://cran.r-project.org/web/packages/adamethods/index.html

library(adamethods)

# I am going to try to implement the example from the docs

## Not run:
library(fda)
?growth
# str(growth)

hgtm <- growth$hgtm # Height males (39 males in total)
hgtf <- growth$hgtf[, 1:39] # Height females (54 females in total)

# Create array
nvars <- 2
data.array <- array(0, dim = c(dim(hgtm), nvars))
data.array[, , 1] <- as.matrix(hgtm)
data.array[, , 2] <- as.matrix(hgtf)
rownames(data.array) <- 1:nrow(hgtm)
colnames(data.array) <- colnames(hgtm)
# str(data.array)

# Create basis
nbasis <- 10
basis_fd <- create.bspline.basis(c(1, nrow(hgtm)), nbasis)
PM <- eval.penalty(basis_fd)

# Make fd object
temp_points <- 1:nrow(hgtm)
temp_fd <- Data2fd(argvals = temp_points, y = data.array, basisobj = basis_fd)

X <- array(0, dim = c(dim(t(temp_fd$coefs[, , 1])), nvars))
X[, , 1] <- t(temp_fd$coef[, , 1])
X[, , 2] <- t(temp_fd$coef[, , 2])

# Standarized the variables
Xs <- X
Xs[, , 1] <- scale(X[, , 1])
Xs[, , 2] <- scale(X[, , 2])

# Give names to the dimensions to know the
# observations identified as archetypoids
dimnames(Xs) <- list(paste("Obs", 1:dim(hgtm)[2], sep = ""),
                    1:nbasis,
                    c("boys", "girls"))

n <- dim(Xs)[1]

# Number of archetypoids:
k <- 3
numRep <- 20
huge <- 200

# Size of the random sample of observations:
m <- 15

# Number of samples:
N <- floor(1 + (n - m) / (m - k))
prob <- 0.75
data_alg <- Xs

# Paralelization
library(doParallel)
no_cores <- detectCores() - 1
cl <- makeCluster(no_cores)
registerDoParallel(cl)
clusterSetRNGStream(cl, iseed = 1)

results <- fadalara(data = data_alg, N = N, m = m, numArchoid = k, numRep = numRep,
                    huge = huge, prob = prob, type_alg = "fada_rob", compare = FALSE,
                    PM = PM, method = "adjbox", multiv = TRUE, frame = FALSE)

stopCluster(cl)

# results_copy <- results
# results <- results[which.min(unlist(sapply(results, function(x) x[2])))][[1]]
# str(results)
# print(results$cases)
# print(results$rss)
# print(results$outliers)
# as.vector(results$outliers)

# Continue studying the data types used in this example
# and implement it with my data
