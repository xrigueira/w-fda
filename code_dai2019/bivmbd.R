#rm(list=ls())
#library(depth)

#compute MSBD for bivariate functional data x1 and x2
#y: 2p by n matrix; x1: the first 1:p rows; x2: (p+1):2p rows 
bivmbd=function(y){
yd=dim(y)
n=yd[2]
p=yd[1]/2
x1=y[1:p,]
x2=y[(p+1):(2*p),]

dp=NULL
for (j in 1:n){
one=NULL
for (i in 1:p){
one=c(one,depth(c(x1[i,j],x2[i,j]),method='Liu',cbind(x1[i,],x2[i,])))
}
dp=c(dp,mean(one))
}
dp
}

#return outlier columns given depth values (dp) and factor 
outbiv=function(y1,y2,dp,factor=1.5){
  #y: only one variable, dp from marginal or joint
  n=dim(y1)[2]
  p=dim(y1)[1]
  index=order(dp,decreasing=T)
  m=ceiling(n/2)#at least 50%
  #Only for x1
  y=sqrt(2)/2*(y1-y2)
  center=y[,index[1:m]]
  inf=apply(center,1,min)
  sup=apply(center,1,max)
  dist=factor*(sup-inf)
  upper=sup+dist
  lower=inf-dist
  outly=(y<=lower)+(y>=upper)
  outcol1=colSums(outly)
  
  y=sqrt(2)/2*(y1+y2)
  center=y[,index[1:m]]
  inf=apply(center,1,min)
  sup=apply(center,1,max)
  upper=sup+dist
  lower=inf-dist
  outly=(y<=lower)+(y>=upper)
  outcol2=colSums(outly)
  #outlier column
  outpoint=which(outcol1>0|outcol2>0)
  outpoint
}


#bivariate funcitonal data y: 2p by n
#y=matrix(rnorm(20*100),20,100)
#first 5 observations are outliers
Data=MDATA12(100,20,2,0.1)
y=t(cbind(Data[,,1],Data[,,2]))
n=ncol(y)
p=nrow(y)/2
x1=y[1:p,]
x2=y[(p+1):(2*p),]
dp=bivmbd(y)
outbiv(x1,x2,dp,factor=3)
