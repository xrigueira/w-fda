Data1=function(n,p,c)
{
x=(1:p/p)
data=matrix(0,n,p)
Sigma=calcSigma(x,x,0.3,1/0.3,1)
#Sigma=calcSigma(x,x,1,1,1)
for (i in 1:n)
{
data[i,]=mvrnorm(1,rep(0,p),Sigma)+30*x*(1-x)^(3/2)
}
if (c>0)
{
for (i in 1:floor(c*n))
{
data[i,]=mvrnorm(1,rep(0,p),Sigma)+30*(1-x)*(x)^(3/2)
}
}
return(data)
}