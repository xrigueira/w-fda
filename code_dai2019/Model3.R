Model3=function(n,p,c)
{
x=(1:p/p)
data=matrix(0,n,p)
Sigma=calcSigma(x,x,1,1,2)
for (i in 1:n)
{
data[i,]=mvrnorm(1,rep(0,p),Sigma)+4*x
}
Sigma=calcSigma(x,x,1,1,0.2)
if (c>0)
{
for (i in 1:floor(c*n))
{
data[i,]=mvrnorm(1,rep(0,p),Sigma)+4*x
}
}
return(data)
}