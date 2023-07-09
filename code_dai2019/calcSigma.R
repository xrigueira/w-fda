calcSigma <- function(X1,X2,k,c,mu) {
  Sigma <- matrix(rep(0, length(X1)*length(X2)), nrow=length(X1))
  for (i in 1:nrow(Sigma)) {
    for (j in 1:ncol(Sigma)) {
      Sigma[i,j] <- k*exp(-c*(abs(X1[i]-X2[j])^mu))
    }
  }
  return(Sigma)
}