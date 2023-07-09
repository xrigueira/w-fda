MDATA2=function(n,p,d=2,c)
{
data=array(0,dim=c(n,p,d),dimnames=list(c(1:n),c(1:p),c("var1","var2")))
x=(1:p/p)
model <- RMbiwm(nudiag=c(1.2, 0.6), nured=1, rhored=0.6, cdiag=c(1, 1),s=c(0.02, 0.016, 0.01))
for (i in 1:n)
{
a=matrix(RFsimulate(model, x),p,2)
data[i,,]=cbind(a[,1],a[,2])
}
##contamined curves##

if (c>0)
{
for (i in 1:floor(c*n))
{
a=matrix(RFsimulate(model, x),p,2)
data[i,,]=cbind(a[,1]+1.2,a[,2]+1.2)
}
}
return(data)
}
