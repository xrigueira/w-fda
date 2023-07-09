Data10=function(n,p,c)
{
x=(1:p/p)
data=matrix(0,n,p)

for (i in 1:n)
{
data[i,]=2*sin(4*pi*x)+(i-27.5)/5
}

data[1,]=0
data[2,]=0.5
data[3,]=-0.5
data[4,]=2*cos(4*pi*x)
data[5,]=2*cos(2*pi*x)

return(data)
}