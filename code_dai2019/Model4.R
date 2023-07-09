Model4=function(n,p,c)
{
x=(1:p/p)
data=matrix(0,n,p)
Sigma=calcSigma(x,x,1,1,2)
for (i in 1:n)
{
data[i,]=runif(1,-4,4)*rep(1,p)
}
Sigma=calcSigma(x,x,1,1,2)
if (c>0)
{
for (i in 1:5)
{
data[i,]=4*abs(sin(4*pi*(x-runif(1))))
data[i+5,]=4*sin(4*pi*(x-runif(1)))
}
}
return(data)
}