# This function calculates the values of m and c from
# the values of n,p, and the number of simulations 
# desired.


cm<-function(n,p,sims,iter){

library(MASS)

	mcd.ave<-numeric(1)
	mcd.sq<-numeric(1)

	for(t in 1:sims){
		mat<-mvrnorm(n,mu=numeric(p),Sigma=diag(p))
	
		mcd.info<-mcd.est(mat,n,p,iter)  	# notice that this is a function we
							# wrote and is not internal to R
							# see additional R code
		mcd.set<-as.logical(mcd.info[[1]])

		mcd.cov<-cov(mat[mcd.set,])

		mcd.ave<-mcd.ave + sum(diag(mcd.cov))
		mcd.sq<-mcd.sq + sum(diag(mcd.cov)^2)
		}

	mcd.c<-mcd.ave/(sims*p)
	mcd.var<-(mcd.sq - (mcd.c*mcd.c*sims*p))/(sims*p-1)
	mcd.m<-2*mcd.c*mcd.c/mcd.var

	c(mcd.c,mcd.m)   # output is the simulated values of c and m

	}

	





