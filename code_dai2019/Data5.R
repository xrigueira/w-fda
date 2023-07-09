Data5=function(n,p,c)
{
x=(1:p/p)*2*pi
data=matrix(0,n,p)
for (i in 1:n)
{
temp=rbinom(1,1,c)
data[i,]=(runif(1,0,0.05)*sin(x)+runif(1,0,0.05)*cos(x))
}
if (c>0)
{
for (i in 1:floor(n*c))
{
data[i,]=(runif(1,0.1,0.15)*sin(x)+runif(1,0.1,0.15)*cos(x))
}
}
return(data)
}