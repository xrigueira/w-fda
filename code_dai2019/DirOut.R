
DirOut=function(data,DirOutmatrix=FALSE,h=0.55,method="Mah")
{
temp=dim(data)
#####################
#Univariate cases   #
#####################

if (length(temp)==2)
{

data=t(data)
p=dim(data)[1]
n=dim(data)[2]
Dirout=matrix(0,n,p)
dmat=matrix(0,p,n)
medvec=apply(data,1,median)
madvec=apply(data,1,mad)
outmat=abs((data-medvec)/(madvec))
signmat=sign((data-medvec))
Dirout=t(outmat*signmat)
out_avr=apply(Dirout,1,mean)
out_var=apply(Dirout,1,var)
M=cbind(out_avr,out_var)
ans=cov.rob(M,method="mcd",nsamp="best",quantile.used=floor(n*h))
cov=ans$cov
me=ans$center
D=mahalanobis(M,me,cov)
}

#####################
#Multivariate cases #
#####################


if (length(temp)==3)
{
n=temp[1]
p=temp[2]
d=temp[3]
Dirout=array(0,dim=c(n,p,d))
for (j in 1:p)
{
temp=covMcd(data[,j,], alpha = 0.51,control = rrcov.control(alpha=0.95))
me=temp$center
if (method=="Mah")
{
out=mahalanobis(data[,j,],temp$center,temp$cov)
}

if (method=="SDO")
{
out=adjOutlyingness(data[,j,],clower=0,cupper=0)$adjout
}
for (i in 1:n)
{
if ((sum((data[i,j,]-me)^2))^(1/2)==0)
{
Dirout[i,j,]=rep(0,d)
}
else{
dir=(data[i,j,]-me)/(sum((data[i,j,]-me)^2))^(1/2)
Dirout[i,j,]=dir*out[i]
}
}
}
out_avr=apply(Dirout,c(1,3),mean)
out_var=apply(Dirout^2,1,mean)*d-apply(out_avr^2,1,sum)
M=cbind(out_avr,out_var)
ans=cov.rob(M,method="mcd",nsamp="best",quantile.used=floor(n*h))
cov=ans$cov
me=ans$center
D=mahalanobis(M,me,cov)
}

if (DirOutmatrix)
list(D=D,Dirout=Dirout,out_avr=out_avr,out_var=out_var)
else 
list(D=D,out_avr=out_avr,out_var=out_var)
}
