Data_add=function(n,p,c)
{
  x=(1:p/p)
  data=matrix(0,n,p)
  #Sigma=calcSigma(x,x,1,10,1/3)
  for (i in 1:n)
  {
    #data[i,]=mvrnorm(1,rep(0,p),Sigma)+4*x
    data[i,]=m0(L=p)
  }
  return(data)
}


m0 <- function(epsilon=0,M=5,L=p){
  truth=4*(1:L)/L
  del <- 1/L
  theta <- exp(-1000*del)
  sigep <- sqrt(1-theta^2)
  rnorme <- function(n){
    return(rnorm(n,mean=0,sd=sigep))
  }
  x <- as.numeric(truth + arima.sim(list(ar=theta),L,rand.gen=rnorme))
  return(x)
}