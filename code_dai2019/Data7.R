Data7=function(n,p,c)
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
temp=runif(1)
temp2=runif(1,0,0.9)
data[i,]=mvrnorm(1,rep(0,p),Sigma)+4*x+sign(temp-0.5)*8*(x>temp2)*(x<(temp2+0.1))
}
}
return(data)
}