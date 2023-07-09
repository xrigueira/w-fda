Data8=function(n,p,c)
{
x=(1:p/p)
data=matrix(0,n,p)
for (i in 1:n)
{
data[i,]=sin(2*pi*(x+runif(1,0.1,0.11)))+runif(1)
}
if (c>0)
{
for (i in 1:floor(c*n))
{
data[i,]=sin(2*pi*x)+runif(1)+10*(x>0.9)
}
}
return(data)
}