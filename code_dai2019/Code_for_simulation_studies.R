##############################################################
##Code for Univariate Functional data Simulation #############
##############################################################

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




ptm <- proc.time()
set.seed(10000)
testoutlier.uni(model=1,c=0.0,repeats=500)
set.seed(10000)
testoutlier.uni(model=1,c=0.1,repeats=500)
set.seed(10000)
testoutlier.uni(model=1,c=0.2,repeats=500)
proc.time() - ptm


ptm <- proc.time()
set.seed(10000)
testoutlier.uni(model=2,c=0.0,repeats=500)
set.seed(10000)
testoutlier.uni(model=2,c=0.1,repeats=500)
set.seed(10000)
testoutlier.uni(model=2,c=0.2,repeats=500)
proc.time() - ptm



ptm <- proc.time()
set.seed(10000)
testoutlier.uni(model=7,c=0.0,repeats=500)
set.seed(10000)
testoutlier.uni(model=7,c=0.1,repeats=500)
set.seed(10000)
testoutlier.uni(model=7,c=0.2,repeats=500)
proc.time() - ptm


ptm <- proc.time()
set.seed(10000)
testoutlier.uni(model=4,c=0.0,repeats=500)
set.seed(10000)
testoutlier.uni(model=4,c=0.1,repeats=500)
set.seed(10000)
testoutlier.uni(model=4,c=0.2,repeats=500)
proc.time() - ptm



################################################################
##Code for Multivariate Functional data Simulation #############
################################################################




ptm <- proc.time()
set.seed(10000)
c=0.1
repeats=500
n=100
p=50
N.T=N.F=N=N1.T=N1.F=N1=N2.T=N2.F=N2=N3.T=N3.F=N3=NULL
for (times in 1:repeats)
{
#DATA=MDATA1(n,p,2,0)   ###Clean model
#DATA=MDATA11(n,p,2,c) ###Isolated Outlier
DATA=MDATA12(n,p,2,c)  ###Shifted Outlier
#DATA=MDATA2(n,p,2,c)  ###Shifted Outlier
#DATA=MDATA3(n,p,2,c)  ###  Shape Outlier
#DATA=MDATA4(n,p,2,c)  ###Shape Outlier

#####################################
## Adjusted Simplicial band depth  ##
#####################################

y=t(cbind(DATA[,,1],DATA[,,2]))
n=ncol(y)
p=nrow(y)/2
x1=y[1:p,]
x2=y[(p+1):(2*p),]
dp=bivmbd(y)
out=outbiv(x1,x2,dp,factor=3)
num=length(out)

if (num==0)
{
num.t=0
num.f=0
}
if (num>0)
{
num.t=length(intersect(out,c(1:floor(n*c))))
num.f=num-num.t
}
N=c(N,num)
N.T=c(N.T,num.t)
N.F=c(N.F,num.f)


#####################################
## Weighted Modified band depth  ##
#####################################


y=t(cbind(DATA[,,1],DATA[,,2]))

WMBD=0.75*fMBD(t(DATA[,,1]))+0.25*fMBD(t(DATA[,,2]))

out1=fbplot(t(DATA[,,1]),depth=WMBD,plot=FALSE,factor=2.45)$outpoint
out2=fbplot(t(DATA[,,2]),depth=WMBD,plot=FALSE,factor=2.45)$outpoint

out=union(out1,out2)
num1=length(out)

if (num1==0)
{
num1.t=0
num1.f=0
}
if (num1>0)
{
num1.t=length(intersect(out,c(1:floor(n*c))))
num1.f=num1-num1.t
}
N1=c(N1,num1)
N1.T=c(N1.T,num1.t)
N1.F=c(N1.F,num1.f)
}

if (c==0)
{
mean(N.F)/(n-n*c)
mean(N1.F)/(n-n*c)
}
if (c>0)
{
mean(N.T)/(n*c)
mean(N.F)/(n-n*c)
mean(N1.T)/(n*c)
mean(N1.F)/(n-n*c)
}
proc.time() - ptm
list(WMBD=c(mean(N1.T)/(n*c),sd(N1.T)/(n*c),mean(N1.F)/(n-n*c),sd(N1.F)/(n-n*c)),
MSBD=c(mean(N.T)/(n*c),sd(N.T)/(n*c),mean(N.F)/(n-n*c),sd(N.F)/(n-n*c)))



ptm <- proc.time()
set.seed(10000)
c=0.1
repeats=500
n=100
p=50
N.T=N.F=N=N1.T=N1.F=N1=N2.T=N2.F=N2=N3.T=N3.F=N3=NULL
fac1=0.1550955
cutoff1=8.3969612


fac2=0.1299469
cutoff2=6.001425

#temp=facCal(n,2)
#fac=temp[1]
#cutoff=temp[2]

for (times in 1:repeats)
{
#DATA=MDATA1(n,p,2,0)   ###Clean model
DATA=MDATA11(n,p,2,c)   ###Isolated Outlier
#DATA=MDATA12(n,p,2,c)  ###Consistent Outlier
#DATA=MDATA1(n,p,2,c)   ###Isolated Outlier
#DATA=MDATA3(n,p,2,c)   ###Shape Outlier
#DATA=MDATA4(n,p,2,c)   ###Shape Outlier



###########################
## Marginal DirOutlying  ##
###########################

data=DATA[,,1]
results=DirOut(data,method="SDO")
M=cbind(results$out_avr,results$out_var)
ans=cov.rob(M,method="mcd",nsamp="best",quantile.used=floor(n*0.75))
cov=ans$cov
me=ans$center
D=mahalanobis(M,me,cov)
num1=sum(fac1*D>cutoff1)
out1=order(-D)[1:num1]

data=DATA[,,2]
results=DirOut(data,method="SDO")
M=cbind(results$out_avr,results$out_var)
ans=cov.rob(M,method="mcd",nsamp="best",quantile.used=floor(n*0.75))
cov=ans$cov
me=ans$center
D=mahalanobis(M,me,cov)
num2=sum(fac1*D>cutoff1)
out2=order(-D)[1:num2]

out=union(out1,out2)
num=length(out)



if (num==0)
{
num.t=0
num.f=0
}
if (num>0)
{
num.t=length(intersect(out,c(1:floor(n*c))))
num.f=num-num.t
}
N=c(N,num)
N.T=c(N.T,num.t)
N.F=c(N.F,num.f)

##################
## DirOutlying  ##
##################

results=DirOut(DATA,method="SDO")
M=cbind(results$out_avr,results$out_var)
cov=cov.rob(M,method="mcd",nsamp="best",quantile.used=floor(n*0.75))$cov
me=cov.mcd(M)$center
D=mahalanobis(M,me,cov)
num=sum(fac2*D>cutoff2)
if (num==0)
{
num1.t=0
num1.f=0
}
if (num>0)
{
num1.t=length(intersect(order(-D)[1:num],c(1:floor(n*c))))
num1.f=num-num1.t
}
N1=c(N1,num)
N1.T=c(N1.T,num1.t)
N1.F=c(N1.F,num1.f)

}

if (c==0)
{
mean(N.F)/(n-n*c)
mean(N1.F)/(n-n*c)
}
if (c>0)
{
mean(N.T)/(n*c)
mean(N.F)/(n-n*c)
mean(N1.T)/(n*c)
mean(N1.F)/(n-n*c)
}

proc.time() - ptm


list(Marginal=c(mean(N.T)/(n*c),sd(N.T)/(n*c),mean(N.F)/(n-n*c),sd(N.F)/(n-n*c)),
Total=c(mean(N1.T)/(n*c),sd(N1.T)/(n*c),mean(N1.F)/(n-n*c),sd(N1.F)/(n-n*c)))








#####################################################
####            Codes for Table 3           #########
#####################################################


#path <- system.file("mat-files", package="R.matlab")
#pathname <- file.path(path, "ECG106_Training.mat")
#DD1 <- readMat(pathname)
DD1=readRDS("ecg.rds")

data=DD1[c(1:810),6:80]

pc4=pc5=pc6=pf4=pf5=pf6=seq(50)*0
pc1=pc2=pc3=pf1=pf2=pf3=seq(50)*0


e=0
m=400*e


for (i in 1:50)
{
data1=data[c(sample(208,m),sample(602,400-m)+208),]
datafull=data


ptm <- proc.time()

OGAdj<-OutGramAdj(data1,c(1:75),plotting=FALSE)
S1=OGAdj$shape.outliers
S2=OGAdj$magnitude.outliers
out1=union(S1,S2)

proc.time() - ptm

ptm <- proc.time()

n=dim(data1)[1]
p=dim(data1)[2]
DD=fds(x=c(1:p)/p,y=t(data1),yname=c(1:n),xname=c(1:p))
colnames(DD$y)<-c(1:n)
ans=foutliers(data=DD, method = "HUoutliers")

out2=ans$out


proc.time() - ptm



ptm <- proc.time()

n=dim(data1)[1]
p=dim(data1)[2]
DD=fds(x=c(1:p)/p,y=t(data1),yname=c(1:n),xname=c(1:p))
colnames(DD$y)<-c(1:n)
ans=foutliers(data=DD, method = "robMah")

out3=ans$out


proc.time() - ptm

ptm <- proc.time()



ptm <- proc.time()


n=dim(data1)[1]
p=dim(data1)[2]

#temp=facCal(810,2)
#fac=0.153556           ##for n=500
#cutoff=5.652435
#fac=0.1542397           ##for n=600
#cutoff=5.5555214
#fac=0.1541646           ##for n=2026
#cutoff=5.0988159
#fac=0.1543659           ##for n=810
#cutoff=5.3676389
fac=0.1538101            ##for n=400 
cutoff=6.0185641


ans=RBD2(t(data1),prop=0.55)
D=ans$D
num=sum(D>cutoff/fac)
out4=order(-D)[1:num]


proc.time() - ptm


ptm <- proc.time()
data2=fdata.deriv(data1,nderiv=1,method="bspline",class.out='fdata')$data

n=dim(data1)[1]
p=dim(data1)[2]

Data=array(0,dim=c(n,p,2))
Data[,,1]=data1
Data[,,2]=data2


#temp=facCal(600,3)
#fac=0.153556           ##for n=500
#cutoff=5.652435
#fac=0.2048584           ##for n=600
#cutoff=5.3431774
#fac=0.1541646           ##for n=2026
#cutoff=5.0988159
#fac=0.1349261           ##for n=810
#cutoff=4.3215012

fac=0.1325181            ## for n=400
cutoff=4.5973351       

results=DirOut(Data,method="SDO")
M=cbind(results$out_avr,results$out_var)
cov=cov.rob(M,method="mcd",nsamp="best",quantile.used=floor(n*0.55))$cov
me=cov.mcd(M)$center
D=mahalanobis(M,me,cov)
num=sum(D>cutoff/fac)
out5=order(-D)[1:num]


proc.time() - ptm

ptm <- proc.time()
data2=fdata.deriv(data1,nderiv=1,method="bspline",class.out='fdata')$data
data3=fdata.deriv(data1,nderiv=2,method="bspline",class.out='fdata')$data

Data=array(0,dim=c(n,p,3))
Data[,,1]=data1
Data[,,2]=data2
Data[,,3]=data3

n=dim(data1)[1]
p=dim(data1)[2]

#temp=facCal(600,3)
#fac=0.153556           ##for n=500
#cutoff=5.652435
#fac=0.2048584           ##for n=600
#cutoff=5.3431774
#fac=0.1541646           ##for n=2026
#cutoff=5.0988159
#fac=0.1169638           ##for n=810
#cutoff=3.7326697
fac=0.115727             ##for n=400
cutoff=3.945562

results=DirOut(Data,method="SDO")
M=cbind(results$out_avr,results$out_var)
cov=cov.rob(M,method="mcd",nsamp="best",quantile.used=floor(n*0.55))$cov
me=cov.mcd(M)$center
D=mahalanobis(M,me,cov)
num=sum(D>cutoff/fac)
out6=order(-D)[1:num]

proc.time() - ptm




######

pc1[i]=sum(out1<(m+1))/m
pf1[i]=sum(out1>=(m+1))/(400-m)

pc2[i]=sum(out2<(m+1))/m
pf2[i]=sum(out2>=(m+1))/(400-m)

pc3[i]=sum(out3<(m+1))/m
pf3[i]=sum(out3>=(m+1))/(400-m)

pc4[i]=sum(out4<(m+1))/m
pf4[i]=sum(out4>=(m+1))/(400-m)

pc5[i]=sum(out5<(m+1))/m
pf5[i]=sum(out5>=(m+1))/(400-m)

pc6[i]=sum(out6<(m+1))/m
pf6[i]=sum(out6>=(m+1))/(400-m)
}

mean(pc4)

mean(pc5)

mean(pc6)

mean(pf4)

mean(pf5)

mean(pf6)

mean(pc1)

mean(pc2)

mean(pc3)

mean(pf1)

mean(pf2)

mean(pf3)










