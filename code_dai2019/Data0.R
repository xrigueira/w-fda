Data0=function(n,p)
{
x=(1:p/p)
data=matrix(0,n,p)
Sigma=calcSigma(x,x,0.1,1,1)
for (i in 1:n)
{
data[i,]=mvrnorm(1,rep(0,p),Sigma)+4*x+(i-50)/10
}

data[1,]=mvrnorm(1,rep(0,p),Sigma)+4*x+20*runif(1,0.9,1)
data[2,]=mvrnorm(1,rep(0,p),Sigma)+4*x+3*sin(6*pi*x)+2-20*runif(1,0.9,1)
data[3,]=mvrnorm(1,rep(0,p),Sigma)+4*x+10*runif(1,0.9,1)
data[4,]=mvrnorm(1,rep(0,p),Sigma)+4*x+3*sin(6*pi*x)
data[5,]=mvrnorm(1,rep(0,p),Sigma)+4*x+6*abs(sin(6*pi*x))-3

return(data)
}