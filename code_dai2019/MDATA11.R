MDATA11=function(n,p,d=2,c)
{
data=array(0,dim=c(n,p,d),dimnames=list(c(1:n),c(1:p),c("var1","var2")))
x=(1:p/p)
model <- RMbiwm(nudiag=c(1.2, 0.6), nured=1, rhored=0.6, cdiag=c(1, 1),s=c(0.02, 0.016, 0.01))
for (i in 1:n)
{
a=matrix(RFsimulate(model, x),p,2)
#data[i,,]=a+cbind(runif(1,1,1.5)*sin(2*pi*x),runif(1,1,1.5)*cos(2*pi*x))
data[i,,]=cbind(a[,1],a[,2])
}

if (c>0)
{
for (i in 1:floor(c*n))
{
a=matrix(RFsimulate(model, x),p,2)
temp=runif(1)
temp2=runif(1,0,0.9)
temp3=runif(1,0,0.9)
data[i,,]=cbind(a[,1]*((x<=temp3)+(x>=(temp3+0.1))+12*(x>temp3)*(x<(temp3+0.1))),a[,2]*((x<=temp2)+(x>=(temp2+0.1))+12*(x>temp2)*(x<(temp2+0.1))))
}
}
return(data)
}