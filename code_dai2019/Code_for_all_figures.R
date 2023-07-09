

#################################
##Code for Figure 1 #############
#################################

par(mfrow=c(1,2),mgp=c(1.5,0.5,0),bty="o",mar=c(2.5,2.5,1.2,1))
x=1:200/200
p=200
n=42
d=2
a=array(0,dim=c(n,p,d))


a[1,,1] <-y<-0.5*sin(4*pi*x)
a[1,,2] <-z<-0.5*cos(4*pi*x)
s3d <- scatterplot3d(y,z,x,type="l",pch=19,angle=60,mar=c(2.5,2.5,1.2,1),color="red",lwd=2,xlim=c(-1.5,1.5),
                ylim=c(-1.5,1.5),zlab="t",xlab=expression(X[1]), ylab=expression(X[2]),main=expression((a)),box=FALSE)

for (i in 1:20)
{
  a[i+1,,1] <-y <- sin(i/20*2*pi)*rep(1,p)
  a[i+1,,2] <-z <-cos(i/20*2*pi)*rep(1,p)
  s3d$points3d(y,z,x,pch = 20,col=8,type="l")
}
for (i in 1:20)
{
  a[i+21,,1] <-y <- 0.5*sin(i/20*2*pi)*rep(1,p)
  a[i+21,,2] <-z <-0.5*cos(i/20*2*pi)*rep(1,p)
  s3d$points3d(y,z,x,pch = 20,col=8,type="l")
}


a[42,,1] <-y <-0*sin(i/20*2*pi)*rep(1,p)
a[42,,2] <-z <-0*cos(i/20*2*pi)*rep(1,p)
s3d$points3d(y,z,x,pch =20,col=8,type="l")

#par(mar=c(2.5,2.5,1.2,0.5))

data.plot=data.frame(y=a[1,,1],
                     z=a[1,,2])
data.point=data.frame(x=a[2:42,1,1],y=a[2:42,1,2])

p<-ggplot(data.plot,aes(y,z))+geom_path(size=1,color="red")+labs(title=expression((b)))+theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))+
  geom_point(data=data.point,aes(x,y),color="grey",size=2)+xlab(expression(X[1]))+ylab(expression(X[2]))
plot.new()
vp <- viewport(height=unit(1,"npc"),width=unit(0.5,"npc"),just=c("right","bottom"),y=0, x=1)
print(p,vp=vp)







#################################
##Code for Figure 2 #############
#################################

#plot(1,1)#
par(mfcol=c(3,3),mar=c(2.5,2.5,1.2,0.1),mgp=c(1.5,0.5,0),oma=c(0.1,0.1,0.5,0.1))
par(font.lab=1,cex.lab=1)
set.seed(100)
a=mvrnorm(200,rep(0,2),matrix(2*c(1,0.7,0.7,1),2,2))
a1=mvrnorm(1,c(3,-3),matrix(0.2*c(1,0.7,0.7,1),2,2))
a4=mvrnorm(1,c(-3,+3),matrix(0.2*c(1,0.7,0.7,1),2,2))
a2=mvrnorm(1,c(-10,10),matrix(0.2*c(1,0.7,0.7,1),2,2))
a3=mvrnorm(1,c(10,10),matrix(0.2*c(1,0.7,0.7,1),2,2))

data=matrix(0,204,2)
data[1:200,]=a
data[201,]=a1
data[202,]=a2
data[203,]=a3
data[204,]=a4
median=med(data,"Liu")$median

data.plot=data.frame(x=c(data[1:204,1],median[1]),
                     y=c(data[1:204,2],median[2]),
                     color=c(rep("grey",200),rep("red",4),"green"))
p<-ggplot(data.plot,aes(x,y,colour=color))+
  geom_point(show.legend = FALSE,colour=c(rep("grey",200),rep("red",4),"green"),shape=c(rep(20,200),11:14,19))+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))+
  xlab(expression(x[1]))+ylab(expression(x[2]))+labs(title="Two Dimensional Points")
plot.new()
vp <- viewport(height=unit(1/3,"npc"),width=unit(1/3,"npc"),just=c("right","bottom"),y=2/3, x=1/3)
print(p,vp=vp)

#plot(1,2)#
depth<-depth::depth

dep=seq(205)
for (i in 1:204)
{
  dep[i]=depth(data[i,],data,method="Liu")
}
dep[205]=depth(median,data,"Liu")


data.plot=data.frame(x=dep,y=rep(1,205),color=c(rep("grey",200),rep("red",4),"green"))

p<-ggplot(data.plot,aes(x,y,colour=color))+
  geom_point(show.legend = FALSE,colour=c(rep("grey",200),rep("red",4),"green"),shape=c(rep(20,200),11:14,19))+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))+
  labs(title="Simplicial Depth")+xlab("Depth")+ylab("")
plot.new()
vp <- viewport(height=unit(1/3,"npc"),width=unit(1/3,"npc"),just=c("right","bottom"),y=1/3, x=1/3)
print(p,vp=vp)



#plot(1,3)#
dep=mdepth.RP(data)$dep

dirdep=matrix(0,205,2)
median=med(data,"Liu")$median
for (i in 1:204)
{
  if(sum((data[i,]-median)^2)==0)
  {dirdep[i,]=rep(0,2)}
  else
  {
    dirdep[i,]=(1/dep[i]-1)*(data[i,]-median)/(sum((data[i,]-median)^2))^(1/2)
  }
}
dirdep[205,]=c(0,0)

data.plot=data.frame(x=dirdep[,1],y=dirdep[,2],color=c(rep("grey",200),rep("red",4),"green"))

p<-ggplot(data.plot,aes(x,y,colour=color))+
  geom_point(show.legend = FALSE,colour=c(rep("grey",200),rep("red",4),"green"),shape=c(rep(20,200),11:14,19))+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))+
  labs(title="Directional SDO")+xlab(expression(SDO(X[1])))+ylab(expression(SDO(X[2])))
plot.new()
vp <- viewport(height=unit(1/3,"npc"),width=unit(1/3,"npc"),just=c("right","bottom"),y=0, x=1/3)
print(p,vp=vp)


#plot(2,1)#

par(font.lab=1,cex.lab=1)

set.seed(100)


data=Data0(100,100)


data1<-data.frame(
  x=1:100/100,
  y=t(data[c(6:100,1:5),])
)

data1_long <- melt(data1, id="x")  # convert to long format

p1<-ggplot(data=data1_long,aes(x=x, y=value,colour=variable,size=variable))+ geom_line(show.legend=FALSE)+ xlab("t")+ylab("Value")+
  labs(title="Univariate Curves") + scale_colour_manual(values=c(rep(8,95),2:6))+scale_size_manual(values=c(rep(0.5,95),rep(1,5)))+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))+
  xlab("t")+ylab("X")
plot.new()
vp <- viewport(height=unit(1/3,"npc"),width=unit(1/3,"npc"),just=c("right","bottom"),y=2/3, x=2/3)
print(p1,vp=vp)




BD=MBD((data),plot=FALSE)$MBD

#plot(BD,rep(1,n),pch=19,col=8,cex=1,main="MBD",xlab="Depth",ylab="")
#points(BD[1:5],rep(1,5),col=c(2:6),pch=19)

data.plot=data.frame(x=BD[c(6:100,1:5)],y=rep(1,100),color=c(rep(8,95),2:6))

p<-ggplot(data.plot,aes(x,y,colour=color))+
  geom_point(show.legend = FALSE,colour=c(rep(8,95),2:6),shape=c(rep(20,95),rep(19,5)))+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))+
  labs(title="MBD")+xlab("Depth")+ylab("")
plot.new()
vp <- viewport(height=unit(1/3,"npc"),width=unit(1/3,"npc"),just=c("right","bottom"),y=1/3, x=2/3)
print(p,vp=vp)

  


results1=DirOut(data)
var=results1$out_var
mr=results1$out_avr

data.plot=data.frame(x=mr[c(6:100,1:5)],y=var[c(6:100,1:5)],color=c(rep(8,95),2:6))

p<-ggplot(data.plot,aes(x,y,colour=color))+
  geom_point(show.legend = FALSE,colour=c(rep(8,95),2:6),shape=c(rep(20,95),rep(19,5)))+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))+
  labs(title="dFSDO")+xlab("MO")+ylab("VO")
plot.new()
vp <- viewport(height=unit(1/3,"npc"),width=unit(1/3,"npc"),just=c("right","bottom"),y=0, x=2/3)
print(p,vp=vp)


par(font.lab=1,cex.lab=0.7)
set.seed(100)

data=MDATA5(100,50,2)
a=data
temp=dim(a)
n=temp[1]
p=temp[2]
x=1:p/p
y <-a[1,,1]
z <-a[1,,2]
s3d <- scatterplot3d(y,z,x,type="l",pch=20,color=1,lab=c(2, 2, 2),angle=60,mar=c(2.5,2.5,1.2,2),
                     xlim=c(-5,5),ylim=c(-5,5),xlab=expression(X[1]),ylab=expression(X[2]),zlab="t",
                     main="Bivariate Curves",box=FALSE)

#plot3d(y,z,x)

for (i in 1:n)
{
  y <-a[i,,1]
  z <-a[i,,2]
  s3d$points3d(y,z,x,pch = 20,col=8,type="l")
}

for (i in 1:5)
{
  y <-a[i,,1]
  z <-a[i,,2]
  s3d$points3d(y,z,x,pch =19,col=i+1,type="l",lwd=2)
}

y1=data[,,1]
y2=data[,,2]
ans=MFHD(y1,y2,alpha=0.125,Beta=0.9)

par(mar=c(2.5,2.5,1.2,0.5))
par(font.lab=1,cex.lab=1)
data.plot=data.frame(x=ans$MFHDdepth[c(6:100,1:5)],y=rep(1,length(ans$MFHDdepth)),color=c(rep(8,95),2:6))
p<-ggplot(data.plot,aes(x,y,colour=color))+
  geom_point(show.legend = FALSE,colour=c(rep(8,95),2:6),shape=c(rep(20,95),rep(19,5)))+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))+
  labs(title="MFHD")+xlab("Depth")+ylab("")
plot.new()
vp <- viewport(height=unit(1/3,"npc"),width=unit(1/3,"npc"),just=c("right","bottom"),y=1/3, x=1)
print(p,vp=vp)


par(font.lab=1,cex.lab=0.7)

results1=DirOut(data,method="SDO")
par(font.lab=1,cex.lab=0.7)
scatterplot3d(cbind(results1$out_avr,results1$out_var),angle=55,mar=c(2.5,2.5,1.2,2),type="h",
              color=c(2:6,rep(8,95)),mgp=c(1,0.5,0),lab=c(2, 2, 5),pch=19,cex.symbols=c(rep(1,5),rep(1,95)),main="dFSDO",
              xlab=expression(MO~(X[1])),ylab=expression(MO~(X[2])),zlab="VO",box=FALSE)

#################################
##Code for Figure 3 #############
#################################

par(mfrow=c(2,2),mar=c(2.5,2.9,2.2,1),mgp=c(1.5,0.5,0),font.lab=1,cex.lab=1)


# n=500
# 
# FF=DD1=DD2=matrix(0,100,n)
# for (k in 1:100)
# {
#   data=Data_add(n,10000,0)
#   res=DirOut(data)
#   #temp=faccal_asy(n,2)
#   DD1[k,]=sort(res$out_avr)
#    DD2[k,]=sort(res$out_var)
#  }
# 
# save(DD1,file="MO_normal_approximation_k10000")
# save(DD2,file="VO_normal_approximation_k10000")




load("MO_normal_approximation_k1000")
load("VO_normal_approximation_k1000")
DD1=t(apply(DD1,1,function(x) x=(x-mean(x))/sd(x)))
DD2=t(apply(DD2,1,function(x) x=(x-mean(x))/sd(x)))
DD1=DD1[1:100,]
DD2=DD2[1:100,]


x=c(qnorm(1:500/501,mean=mean(DD1),sd=sd(DD1)))
qqPlot(c(-5,x,5),distribution = "norm",mean=mean(DD1),sd=sd(DD1),pch=0,col=0,main="Normal Q-Q Plot of MO (n=500,k=1000)",
       xlab="Theoretical Normal Quantiles",ylab="MO")
matplot(x%*%t(rep(1,dim(DD1)[1])),t(DD1[,]),type="l",lty=4,add=TRUE,col="grey")



x=c(qnorm(1:500/501,mean=mean(DD2),sd=sd(DD2)))
qqPlot(c(-5,x,5),distribution = "norm",mean=mean(DD2),sd=sd(DD2),pch=0,col=0,main="Normal Q-Q Plot of VO (n=500,k=1000)",
       xlab="Theoretical Normal Quantiles",ylab="VO")
matplot(x%*%t(rep(1,dim(DD2)[1])),t(DD2[,]),type="l",lty=4,add=TRUE,col="grey")



load("MO_normal_approximation_k10000")
load("VO_normal_approximation_k10000")
DD1=t(apply(DD1,1,function(x) x=(x-mean(x))/sd(x)))
DD2=t(apply(DD2,1,function(x) x=(x-mean(x))/sd(x)))

DD1=DD1[1:100,]
DD2=DD2[1:100,]

x=c(qnorm(1:500/501,mean=mean(DD1),sd=sd(DD1)))
qqPlot(c(-5,x,5),distribution = "norm",mean=mean(DD1),sd=sd(DD1),pch=0,col=0,main="Normal Q-Q Plot of MO (n=500,k=10000)",
       xlab="Theoretical Normal Quantiles",ylab="MO")
matplot(x%*%t(rep(1,dim(DD1)[1])),t(DD1[,]),type="l",lty=4,add=TRUE,col="grey")



x=c(qnorm(1:500/501,mean=mean(DD2),sd=sd(DD2)))
qqPlot(c(-5,x,5),distribution = "norm",mean=mean(DD2),sd=sd(DD2),pch=0,col=0,main="Normal Q-Q Plot of VO (n=500,k=10000)",
       xlab="Theoretical Normal Quantiles",ylab="VO")
matplot(x%*%t(rep(1,dim(DD2)[1])),t(DD2[,]),type="l",lty=4,add=TRUE,col="grey")


#################################
##Code for Figure 4 #############
#################################
# n=100
# 
# 
# 
# FF=DD=matrix(0,1000,n)
# for (k in 1:1000)
# {
#   data=Data_add(n,50,0)
#   res=DirOut(data)
#   temp=faccal_asy(n,2)
#   DD[k,]=sort(res$D)*temp$fac1
# }
# 
# save(DD,file="DD_k1mu1000c1n100p50N1000")
# 
# 
# 
# FF=DD=matrix(0,1000,n)
# for (k in 1:1000)
# {
#   data=Data4(n,50,0)
#   res=DirOut(data)
#   temp=faccal_asy(n,2)
#   DD[k,]=sort(res$D)*temp$fac1
# }
# 
# save(DD,file="DD_k1mu1c1n100p50N1000")
# 
# 
# FF=DD=matrix(0,1000,n)
# for (k in 1:1000)
# {
#   data=Data1(n,50,0)
#   res=DirOut(data)
#   temp=faccal_asy(n,2)
#   DD[k,]=sort(res$D)*temp$fac1
# }
# 
# save(DD,file="DD_k03mu33c1n100p50N1000")







par(mfcol=c(1,3),mar=c(2.5,2.9,1.2,2),mgp=c(1.5,0.5,0),font.lab=1,cex.lab=1)

temp=facCal_asy(100,2)
load(file="DD_k1mu1000c1n100p50N1000")
#DD=DD[1:100,]

data_RMD=data.frame(x=as.vector(log(DD)))
data_lines1=data.frame(x=rep(log(qchisq(0.993,df=2)*temp$fac1),100),y=(0:99)*150)
data_lines2=data.frame(x=rep(log(temp$fac2),100),y=(0:99)*150)
quantile=sort(log(DD))[0.993*length(DD)]
data_lines3=data.frame(x=rep(quantile,100),y=(0:99)*150)
sum(DD>qchisq(0.993,df=2)*temp$fac1)/length(DD)
sum(DD>temp$fac2)/length(DD)


p1<-ggplot(data_RMD,aes(x=x))+geom_histogram(bindwidth=0.5)+ylim(c(0,25000))+
geom_line(data=data_lines1,aes(x,y),lty=2,col=2,lwd=1)+
geom_line(data=data_lines2,aes(x,y),lty=3,col=3,lwd=1)+
geom_line(data=data_lines3,aes(x,y),lty=4,col=4,lwd=1)+
  xlab(expression(paste("Logarithm of the Scaled ", RMD^2)))+xlim(c(-12,6))+
labs(title="Model 0")+ylab("Count")+ theme(plot.title = element_text(face = "bold",size=12,hjust = 0.5),axis.title.x=element_text(size=10),axis.title.y=element_text(size=10))

plot.new() 
vp <- viewport(height=unit(1,"npc"),width=unit(1/3,"npc"),just=c("right","bottom"),y=0, x=1/3) 
print(p1,vp=vp) 



load(file="DD_k03mu33c1n100p50N1000")
#DD=DD[1:100,]

data_RMD=data.frame(x=as.vector(log(DD)))
data_lines1=data.frame(x=rep(log(qchisq(0.993,df=2)*temp$fac1),100),y=(0:99)*150)
data_lines2=data.frame(x=rep(log(temp$fac2),100),y=(0:99)*150)
quantile=sort(log(DD))[0.993*length(DD)]
data_lines3=data.frame(x=rep(quantile,100),y=(0:99)*150)
sum(DD>qchisq(0.993,df=2)*temp$fac1)/length(DD)
sum(DD>temp$fac2)/length(DD)



p1<-ggplot(data_RMD,aes(x=x))+geom_histogram(bindwidth=0.5)+ylim(c(0,25000))+
  geom_line(data=data_lines1,aes(x,y),lty=2,col=2,lwd=1)+
  geom_line(data=data_lines2,aes(x,y),lty=3,col=3,lwd=1)+
  geom_line(data=data_lines3,aes(x,y),lty=4,col=4,lwd=1)+
  xlab(expression(paste("Logarithm of the Scaled ", RMD^2)))+xlim(c(-12,6))+
  labs(title="Model 3")+ylab("Count")+ theme(plot.title = element_text(face = "bold",size=12,hjust = 0.5),axis.title.x=element_text(size=10),axis.title.y=element_text(size=10))

plot.new() 
vp <- viewport(height=unit(1,"npc"),width=unit(1/3,"npc"),just=c("right","bottom"),y=0, x=2/3) 
print(p1,vp=vp) 



load(file="DD_k1mu1c1n100p50N1000")
#DD=DD[1:100,]

data_RMD=data.frame(x=as.vector(log(DD)))
data_lines1=data.frame(x=rep(log(qchisq(0.993,df=2)*temp$fac1),100),y=(0:99)*150)
data_lines2=data.frame(x=rep(log(temp$fac2),100),y=(0:99)*150)
quantile=sort(log(DD))[0.993*length(DD)]
data_lines3=data.frame(x=rep(quantile,100),y=(0:99)*150)
sum(DD>qchisq(0.993,df=2)*temp$fac1)/length(DD)
sum(DD>temp$fac2)/length(DD)

p1<-ggplot(data_RMD,aes(x=x))+geom_histogram(bindwidth=0.5)+ylim(c(0,25000))+
  geom_line(data=data_lines1,aes(x,y),lty=2,col=2,lwd=1)+
  geom_line(data=data_lines2,aes(x,y),lty=3,col=3,lwd=1)+
  geom_line(data=data_lines3,aes(x,y),lty=4,col=4,lwd=1)+
  xlab(expression(paste("Logarithm of the Scaled ", RMD^2)))+xlim(c(-12,6))+
  labs(title="Models 1,2, and 4")+ylab("Count")+ theme(plot.title = element_text(face = "bold",size=12,hjust = 0.5),axis.title.x=element_text(size=10),axis.title.y=element_text(size=10))


plot.new() 
vp <- viewport(height=unit(1,"npc"),width=unit(1/3,"npc"),just=c("right","bottom"),y=0, x=3/3) 
print(p1,vp=vp) 

#################################
##Code for Figure 5 #############
#################################



par(mfrow=c(2,3),mar=c(2.7,2.5,2,0.5),mgp=c(1.5,0.5,0),font.lab=1,cex.lab=1.2)
#aemet=readRDS("aemet.rds")

data(aemet)

geo=abind(aemet$df$longitude,aemet$df$latitude,aemet$df$altitude,along=2)

bsp11<-create.bspline.basis(aemet$temp$rangeval,nbasis=5)
S.bsp11  <-  S.basis(aemet$temp$argvals, bsp11)
temp.bsp11<-temp.bsp111<-aemet$temp
temp.bsp11$data <- aemet$temp$data%*%S.bsp11
sdata1=temp.bsp11$data
data=sdata1
#sdata1=data=aemet$temp$data


se1=c(34,35,36,55,57,58,60)
se2=c(45)
se3=c(56)
se4=c(59)
se5=c(39,33,44,66)
se=c(1:73)[-c(se1,se2,se3,se4,se5)]

col=seq(73)*0+1
col[se1]=3
col[se]=8
col[se5]=6
col[c(se2,se3,se4)]=c(4,5,2)

#data=aemet$temp$data


data1<-data.frame(x=1:365,y=t(data))
data1_long <- melt(data1, id="x")  # convert to long format
p1<-ggplot(data=data1_long,aes(x=x, y=value,colour=variable))+ geom_line(show.legend=FALSE)+ xlab("Time (Day)")+ylab(expression(~degree~C))+
  labs(title="Temperature (NBSP=5)") + scale_colour_manual(values=col)+scale_size_manual(values=c(rep(1,94),rep(2,6)))+ylim(c(-5,30))+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))
plot.new()
vp <- viewport(height=unit(1/2,"npc"),width=unit(1/3,"npc"),just=c("right","bottom"),y=1/2, x=1/3)
print(p1,vp=vp)


#fpca.score(data)

n=dim(data)[1]
p=dim(data)[2]

dmat=matrix(0,n,p)
medvec=apply(t(data),1,median)
madvec=apply(t(data),1,mad)
dmat=(t(data)-medvec)/(madvec)
mr=apply(dmat,2,mean)
var=apply(dmat,2,var)
data.plot=data.frame(x=mr,y=var,col=col)
p<-ggplot(data.plot,aes(x,y,colour=color))+
  geom_point(show.legend = FALSE,colour=col)+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))+
  labs(title="Directional Outlyingness")+xlab("MO (Temperature)")+ylab("VO")+ylim(c(0,1.2))+xlim(c(-3,2))
plot.new()
vp <- viewport(height=unit(1/2,"npc"),width=unit(1/3,"npc"),just=c("right","bottom"),y=0, x=1/3)
print(p,vp=vp)



geo=abind(aemet$df$longitude,aemet$df$latitude,aemet$df$altitude,along=2)

bsp11<-create.bspline.basis(aemet$temp$rangeval,nbasis=11)
S.bsp11  <-  S.basis(aemet$temp$argvals, bsp11)
temp.bsp11<-temp.bsp111<-aemet$temp
temp.bsp11$data <- aemet$temp$data%*%S.bsp11
sdata1=temp.bsp11$data
data=sdata1
#sdata1=data=aemet$temp$data


se1=c(34,35,36,55,57,58,60)
se2=c(45)
se3=c(56)
se4=c(59)
se5=c(39,33,44,66)
se=c(1:73)[-c(se1,se2,se3,se4,se5)]

col=seq(73)*0+1
col[se1]=3
col[se]=8
col[se5]=6
col[c(se2,se3,se4)]=c(4,5,2)

#data=aemet$temp$data


data1<-data.frame(x=1:365,y=t(data))
data1_long <- melt(data1, id="x")  # convert to long format
p1<-ggplot(data=data1_long,aes(x=x, y=value,colour=variable))+ geom_line(show.legend=FALSE)+ xlab("Time (Day)")+ylab(expression(~degree~C))+
  labs(title="Temperature (NBSP=11)") + scale_colour_manual(values=col)+scale_size_manual(values=c(rep(1,94),rep(2,6)))+ylim(c(-5,30))+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))
plot.new()
vp <- viewport(height=unit(1/2,"npc"),width=unit(1/3,"npc"),just=c("right","bottom"),y=1/2, x=2/3)
print(p1,vp=vp)




n=dim(data)[1]
p=dim(data)[2]

dmat=matrix(0,n,p)
medvec=apply(t(data),1,median)
madvec=apply(t(data),1,mad)
dmat=(t(data)-medvec)/(madvec)
mr=apply(dmat,2,mean)
var=apply(dmat,2,var)
data.plot=data.frame(x=mr,y=var,col=col)
p<-ggplot(data.plot,aes(x,y,colour=color))+
  geom_point(show.legend = FALSE,colour=col)+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))+
  labs(title="Directional Outlyingness")+xlab("MO (Temperature)")+ylab("VO")+ylim(c(0,1.2))+xlim(c(-3,2))
plot.new()
vp <- viewport(height=unit(1/2,"npc"),width=unit(1/3,"npc"),just=c("right","bottom"),y=0, x=2/3)
print(p,vp=vp)




geo=abind(aemet$df$longitude,aemet$df$latitude,aemet$df$altitude,along=2)

bsp11<-create.bspline.basis(aemet$temp$rangeval,nbasis=100)
S.bsp11  <-  S.basis(aemet$temp$argvals, bsp11)
temp.bsp11<-temp.bsp111<-aemet$temp
temp.bsp11$data <- aemet$temp$data%*%S.bsp11
sdata1=temp.bsp11$data
data=sdata1
#sdata1=data=aemet$temp$data


se1=c(34,35,36,55,57,58,60)
se2=c(45)
se3=c(56)
se4=c(59)
se5=c(39,33,44,66)
se=c(1:73)[-c(se1,se2,se3,se4,se5)]

col=seq(73)*0+1
col[se1]=3
col[se]=8
col[se5]=6
col[c(se2,se3,se4)]=c(4,5,2)

#data=aemet$temp$data


data1<-data.frame(x=1:365,y=t(data))
data1_long <- melt(data1, id="x")  # convert to long format
p1<-ggplot(data=data1_long,aes(x=x, y=value,colour=variable))+ geom_line(show.legend=FALSE)+ xlab("Time (Day)")+ylab(expression(~degree~C))+
  labs(title="Temperature (NBSP=100)") + scale_colour_manual(values=col)+scale_size_manual(values=c(rep(1,94),rep(2,6)))+ylim(c(-5,30))+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))
plot.new()
vp <- viewport(height=unit(1/2,"npc"),width=unit(1/3,"npc"),just=c("right","bottom"),y=1/2, x=1)
print(p1,vp=vp)




n=dim(data)[1]
p=dim(data)[2]

dmat=matrix(0,n,p)
medvec=apply(t(data),1,median)
madvec=apply(t(data),1,mad)
dmat=(t(data)-medvec)/(madvec)
mr=apply(dmat,2,mean)
var=apply(dmat,2,var)
data.plot=data.frame(x=mr,y=var,col=col)
p<-ggplot(data.plot,aes(x,y,colour=color))+
  geom_point(show.legend = FALSE,colour=col)+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))+
  labs(title="Directional Outlyingness")+xlab("MO (Temperature)")+ylab("VO")+ylim(c(0,1.2))+xlim(c(-3,2))
plot.new()
vp <- viewport(height=unit(1/2,"npc"),width=unit(1/3,"npc"),just=c("right","bottom"),y=0/2, x=1)
print(p,vp=vp)






#################################
##Code for Figure 6 #############
#################################

par(mfrow=c(3,2),mar=c(2.7,2.5,2,0.5),mgp=c(1.5,0.5,0),font.lab=1,cex.lab=1.2)
#aemet=readRDS("aemet.rds")

data(aemet)

geo=abind(aemet$df$longitude,aemet$df$latitude,aemet$df$altitude,along=2)

bsp11<-create.bspline.basis(aemet$temp$rangeval,nbasis=11)
S.bsp11  <-  S.basis(aemet$temp$argvals, bsp11)
temp.bsp11<-temp.bsp111<-aemet$temp
temp.bsp11$data <- aemet$temp$data%*%S.bsp11
sdata1=temp.bsp11$data
data=sdata1
#sdata1=data=aemet$temp$data


se1=c(34,35,36,55,57,58,60)
se2=c(45)
se3=c(56)
se4=c(59)
se5=c(39,33,44,66)
se=c(1:73)[-c(se1,se2,se3,se4,se5)]

col=seq(73)*0+1
col[se1]=3
col[se]=8
col[se5]=6
col[c(se2,se3,se4)]=c(4,5,2)

#data=aemet$temp$data


data1<-data.frame(x=1:365,y=t(data))
data1_long <- melt(data1, id="x")  # convert to long format
p1<-ggplot(data=data1_long,aes(x=x, y=value,colour=variable))+ geom_line(show.legend=FALSE)+ xlab("Time (Day)")+ylab(expression(~degree~C))+
  labs(title="Temperature") + scale_colour_manual(values=col)+scale_size_manual(values=c(rep(1,94),rep(2,6)))+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))
plot.new()
vp <- viewport(height=unit(1/3,"npc"),width=unit(1/2,"npc"),just=c("right","bottom"),y=2/3, x=0.5)
print(p1,vp=vp)




n=dim(sdata1)[1]
p=dim(sdata1)[2]

dmat=matrix(0,n,p)
medvec=apply(t(data),1,median)
madvec=apply(t(data),1,mad)
dmat=(t(data)-medvec)/(madvec)
mr=apply(dmat,2,mean)
var=apply(dmat,2,var)
data.plot=data.frame(x=mr,y=var,col=col)
p<-ggplot(data.plot,aes(x,y,colour=color))+
  geom_point(show.legend = FALSE,colour=col)+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))+
  labs(title="Directional Outlyingness")+xlab("MO (Temperature)")+ylab("VO")
plot.new()
vp <- viewport(height=unit(1/3,"npc"),width=unit(0.5,"npc"),just=c("right","bottom"),y=2/3, x=1)
print(p,vp=vp)






bsp11<-create.bspline.basis(aemet$logprec$rangeval,nbasis=11)
S.bsp11  <-  S.basis(aemet$logprec$argvals, bsp11)
temp.bsp11<-temp.bsp111<-aemet$logprec
temp.bsp11$data <- aemet$logprec$data%*%S.bsp11
sdata2=temp.bsp11$data
data=sdata2
#sdata2=data=aemet$logprec$data

data1<-data.frame(x=1:365,y=t(data))
data1_long <- melt(data1, id="x")  # convert to long format
p1<-ggplot(data=data1_long,aes(x=x, y=value,colour=variable))+ geom_line(show.legend=FALSE)+ xlab("Time (Day)")+ylab("")+
  labs(title="Log Precipitation") + scale_colour_manual(values=col)+scale_size_manual(values=c(rep(1,94),rep(2,6)))+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))
plot.new()
vp <- viewport(height=unit(1/3,"npc"),width=unit(1/2,"npc"),just=c("right","bottom"),y=1/3, x=0.5)
print(p1,vp=vp)


n=dim(sdata2)[1]
p=dim(sdata2)[2]

dmat=matrix(0,n,p)
medvec=apply(t(data),1,median)
madvec=apply(t(data),1,mad)
dmat=(t(data)-medvec)/(madvec)
mr=apply(dmat,2,mean)
var=apply(dmat,2,var)

data.plot=data.frame(x=mr,y=var,col=col)
p<-ggplot(data.plot,aes(x,y,colour=color))+
  geom_point(show.legend = FALSE,colour=col)+
  theme(plot.title = element_text(face = "bold",size=10,hjust = 0.5),axis.title.x=element_text(size=8),axis.title.y=element_text(size=8))+
  labs(title="Directional Outlyingness")+xlab("MO (Log Precipitation)")+ylab("VO")
plot.new()
vp <- viewport(height=unit(1/3,"npc"),width=unit(0.5,"npc"),just=c("right","bottom"),y=1/3, x=1)
print(p,vp=vp)


par(font.lab=1,cex.lab=0.7)

n=dim(aemet$temp)[1]
p=dim(aemet$temp)[2]
data=array(0,dim=c(n,p,2),dimnames=list(c(1:n),c(1:p),c("var1","var2")))
data[,,1]=sdata1
data[,,2]=sdata2
results=DirOut(data,method="SDO")
tt=cbind(results$out_avr,results$out_var)

geo=cbind(aemet$df$longitude,aemet$df$latitude,aemet$df$altitude)
a<-scatterplot3d(geo,type="n",xlab="Longitude",ylab="Latitude",zlab="Altitude",lab=c(2, 2, 2),angle=60,mar=c(2.7,2.5,2,2),main="Locations of Stations",box=FALSE)
a$points3d(geo[se,],lab=c(2, 2, 2),col=8,pch=19,cex=1)
a$points3d(geo[se1,],lab=c(2, 2, 2),col=3,pch=19,cex=1)
a$points3d(geo[se5,],lab=c(2, 2, 2),col=6,pch=19,cex=1)
a$points3d(geo[c(se2,se3,se4),],lab=c(2, 2, 2),col=c(4,5,2),pch=19,cex=1)

dimnames(tt)=list(c(1:n),c("MO (Temperature)","MO (Log precipitation)","VO"))
b<-scatterplot3d(tt,lab=c(2, 2, 2),main="Directional Outlyingness",type="h",mar=c(2.7,2.5,2,2),angle=60,box=FALSE)
b$points3d(tt[se,],lab=c(2, 2, 2),col=8,pch=19,cex=1)
b$points3d(tt[se1,],lab=c(2, 2, 2),col=3,pch=19,cex=1)
b$points3d(tt[se5,],lab=c(2, 2, 2),col=6,pch=19,cex=1)
b$points3d(tt[c(se2,se3,se4),],lab=c(2, 2, 2),col=c(4,5,2),pch=19,cex=1)


#col=seq(n)
#col[se]=1
#col[se1]=3
#col[se5]=6
#col[c(se2,se3,se4)]=c(4,5,2)

#plot(rep(1,n),ans$MBD,col=col)



#################################
##Code for Figure 7 #############
#################################



par(mfrow=c(1,1),mar=c(2.5,2.5,2,0.5),mgp=c(1.5,0.5,0))

data=readRDS("ecg.rds")
col=seq(810)
col[1:208]=rgb(1,0,0,0.3)
col[209:810]=rgb(190/255,190/255,190/255,0.3)

data1<-data.frame(x=1:85,y=t(data))
data1_long <- melt(data1, id="x")  # convert to long format
p1<-ggplot(data=data1_long,aes(x=x, y=value,colour=variable))+ geom_line(lwd=1,show.legend=FALSE)+ xlab("Time (seconds)")+ylab("Valtage (mV)")+
  labs(title="ECG Data") + scale_colour_manual(values=col)+
  theme(plot.title = element_text(face = "bold",size=12,hjust = 0.5),axis.title.x=element_text(size=10),axis.title.y=element_text(size=10))
plot.new()
vp <- viewport(height=unit(1,"npc"),width=unit(1,"npc"),just=c("right","bottom"),y=0, x=1)
print(p1,vp=vp)













