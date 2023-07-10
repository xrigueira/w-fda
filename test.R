library("RandomFields")

model1 <- function(n, p, d=6, c) {
    
    data <- array(0, dim=c(n, p, d), dimnames=list(c(1:n), c(1:p), c("var1", "var2", "var3", "var4", "var5", "var6")))
    
    # Define p points in the range [0, 1]
    x <- (1:p/p)
    
    model <- RMbiwm(nudiag=c(1.2, 0.6), nured=1, rhored=0.6, cdiag=c(1, 1), s=c(0.02, 0.016, 0.01))
    
    for (i in 1:n) {
        a <- matrix(RFsimulate(model , x), p, 2)
        data[i, , ] <- cbind(a[, 1], a[, 2])
    }
}

model1(n=25, p=96, c=0.1)