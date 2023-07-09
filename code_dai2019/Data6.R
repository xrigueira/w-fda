Data6=function(n,p,c)
{
x=(1:p/p)
data=matrix(0,n,p)
Sigma=calcSigma(x,x,1,1,1)

for (i in 1:n)
{
data[i,]=mvrnorm(1,rep(0,p),Sigma)+4*x
}

if (c>0)
{
for (i in 1:floor(c*n))
{
data[i,]=mvrnorm(1,rep(0,p),Sigma)+4*x+(-1)^(rbinom(1,1,0.1))*1.8+(2*pi*0.01)^(-1/2)*exp(-(x-runif(1,0.25,0.75))^2/0.02)
}
}
return(data)
}