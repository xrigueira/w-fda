MDATA5=function(n,p,d=2)
{
data=array(0,dim=c(n,p,d),dimnames=list(c(1:n),c(1:p),c("var1","var2")))
x=(1:p/p)
model <- RMbiwm(nudiag=c(0.5, 0.5), nured=1, rhored=0.6, cdiag=c(1, 1)/10,s=10*c(0.02, 0.016, 0.01))
for (i in 1:floor(n/4))
{
a=matrix(RFsimulate(model, x),p,2)/10
data[i,,]=cbind(a[,1]+runif(1,0,1/4)*sin(i/100*16*pi)*rep(1,p),a[,2]+runif(1,0,1/4)*cos(i/100*16*pi)*rep(1,p))
data[floor(n/4)+i,,]=cbind(a[,1]+runif(1,3/4,1)*sin(i/100*16*pi)*rep(1,p),a[,2]+runif(1,3/4,1)*cos(i/100*16*pi)*rep(1,p))
data[floor(n/4)*2+i,,]=cbind(a[,1]+runif(1,1/2,3/4)*sin(i/100*16*pi)*rep(1,p),a[,2]+runif(1,1/2,3/4)*cos(i/100*16*pi)*rep(1,p))
data[floor(n/4)*3+i,,]=cbind(a[,1]+runif(1,1/4,1/2)*sin(i/100*16*pi)*rep(1,p),a[,2]+runif(1,1/4,1/2)*cos(i/100*16*pi)*rep(1,p))
#data[i,,]=cbind(a[,1],c(a[1:2,2],12*a[3:10,2],a[11:100,2]))
}
##contamined curves##
data[4,,]=cbind(a[,1]+1/2*sin(x*4*pi),a[,2]+1/2*cos(x*4*pi))
data[5,,]=cbind(a[,1]+1/2*sin(x*8*pi),a[,2]+1/2*cos(x*8*pi))
data[1,,]=cbind(a[,1]+rep(5,p),a[,2]+rep(0,p))
data[3,,]=cbind(a[,1]+rep(0,p),a[,2]-rep(5,p))
data[2,,]=cbind(a[,1]+5+1/2*sin(x*4*pi),a[,2]-5+1/2*cos(x*4*pi))


return(data)
}
