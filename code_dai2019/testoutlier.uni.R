testoutlier.uni=function(model,c,repeats,fac=0.154,cutoff=6.91)
{
n=dim(data)[1]
p=dim(data)[2]
N.T=N.F=N=N1.T=N1.F=N1=N2.T=N2.F=N2=N3.T=N3.F=N3=NULL



for (times in 1:repeats)
{
if (model==1)
{data=Model2(n,p,c)}
if (model==2)
{data=Data7(n,p,c)}
if (model==3)
{data=Data1(n,p,c)}
if (model==4)
{data=Data4(n,p,c)}
if (model==5)
{data=Data2(n,p,c)}
if (model==6)
{data=Data3(n,p,c)}
if (model==7)
{data=Data6(n,p,c)}



##################
## DirOutlying  ##
##################
results=DirOut(data)
M=cbind(results$out_avr,results$out_var)
ans=cov.rob(M,method="mcd",nsamp="best",quantile.used=floor(n*0.55))
cov=ans$cov
me=ans$center
D=mahalanobis(M,me,cov)
num=sum(fac*D>cutoff)
if (num==0)
{
num.t=0
num.f=0
}
if (num>0)
{
num.t=length(intersect(order(-D)[1:num],c(1:floor(n*c))))
num.f=num-num.t
}
N=c(N,num)
N.T=c(N.T,num.t)
N.F=c(N.F,num.f)

##################
## Outliergrame ##
##################
OGAdj<-OutGramAdj(data,c(1:100)/100,plotting=FALSE)
S1=OGAdj$shape.outliers
S2=OGAdj$magnitude.outliers
num1=length(S1)+length(S2)
if (num1==0)
{
num1.t=0
num1.f=0
}
if (num1>0)
{
num1.t=length(intersect(union(S1,S2),c(1:floor(n*c))))
num1.f=num1-num1.t
}
N1=c(N1,num1)
N1.T=c(N1.T,num1.t)
N1.F=c(N1.F,num1.f)

##################
## HUoutliers   ##
##################

DD=fds(x=c(1:p)/p,y=t(data),yname=c(1:n),xname=c(1:p))
colnames(DD$y)<-c(1:n)
a=foutliers(data=DD, method = "HUoutliers")
num2=length(a$outlier)
if (num2==0)
{
num2.t=0
num2.f=0
}
if (num2>0)
{
num2.t=length(intersect(a$outlier,c(1:floor(n*c))))
num2.f=num2-num2.t
}
N2=c(N2,num2)
N2.T=c(N2.T,num2.t)
N2.F=c(N2.F,num2.f)

##################
## Robust Mah   ##
##################

a1=foutliers(data=DD, method = "robMah")
num3=length(a1$outlier)
if (num3==0)
{
num3.t=0
num3.f=0
}
if (num3>0)
{
num3.t=length(intersect(a1$outlier,c(1:floor(n*c))))
num3.f=num3-num3.t
}
N3=c(N3,num3)
N3.T=c(N3.T,num3.t)
N3.F=c(N3.F,num3.f)
}

if (c==0)
{
mean(N.F)/(n-n*c)
mean(N1.F)/(n-n*c)
mean(N2.F)/(n-n*c)
}
if (c>0)
{
mean(N.T)/(length(c(1:floor(n*c))))
mean(N.F)/(n-n*c)
mean(N1.T)/(length(c(1:floor(n*c))))
mean(N1.F)/(n-n*c)
mean(N2.T)/(length(c(1:floor(n*c))))
mean(N2.F)/(n-n*c)
}

Dirout=c(mean(N.T)/(n*c),sd(N.T)/(n*c),mean(N.F)/(n-n*c),sd(N.F)/(n-n*c))*100
Outliergram=c(mean(N1.T)/(n*c),sd(N1.T)/(n*c),mean(N1.F)/(n-n*c),sd(N1.F)/(n-n*c))*100
ISE=c(mean(N2.T)/(n*c),sd(N2.T)/(n*c),mean(N2.F)/(n-n*c),sd(N2.F)/(n-n*c))*100
RobMah=c(mean(N3.T)/(n*c),sd(N3.T)/(n*c),mean(N3.F)/(n-n*c),sd(N3.F)/(n-n*c))*100

list(Dirout=Dirout,Outliergram=Outliergram,ISE=ISE,RobMah=RobMah)

}