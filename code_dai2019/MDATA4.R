MDATA4=function(n,p,d=2,c)
{
data=array(0,dim=c(n,p,d),dimnames=list(c(1:n),c(1:p),c("var1","var2")))
x=(1:p/p)
model <- RMbiwm(nudiag=c(1.2, 0.6), nured=1, rhored=0.6, cdiag=c(1, 1),s=c(0.02, 0.016, 0.01))
for (i in 1:n)
{
a=matrix(RFsimulate(model, x),p,2)
data[i,,]=cbind(a[,1]+runif(1,2,3)*cos(4*pi*x),a[,2]+runif(1,2,3)*sin(4*pi*x))
#data[i,,]=cbind(runif(1,2,3)*cos(4*pi*x),runif(1,2,3)*sin(4*pi*x))
#data[i,,]=cbind(a[,1],c(a[1:2,2],12*a[3:10,2],a[11:100,2]))
}
##contamined curves##
if (c>0)
{
for (i in 1:floor(c*n))
{
a=matrix(RFsimulate(model, x),p,2)
data[i,,]=cbind(a[,1]+runif(1,4,5)*cos(4*pi*x),a[,2]+runif(1,4,5)*sin(4*pi*x))
#data[i,,]=cbind(a[,1],c(a[1:2,2],12*a[3:10,2],a[11:100,2]))
#data[i,,]=cbind(runif(1,3.2,3.5)*cos(4*pi*x),runif(1,3.2,3.5)*sin(4*pi*x))

}
}
return(data)
}
