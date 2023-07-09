Data4=function(n,p,c)
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
Sigma=calcSigma(x,x,4,1,0.2)
for (i in 1:floor(c*n))
{
data[i,]=mvrnorm(1,rep(0,p),Sigma)+4*x
}
}
return(data)
}