
# This function calculates the MCD for a set of data.  
# cov.mcd( ) in R reweights the matrix as to not give 
# the raw MCD estimates.


mcd.est<-function(mat,n,p,iter){
	h<-floor((n+p+1)/2)
	bestdet<-1000

for(t in 1:iter){	
	newmat<-mat[sample(1:n,h),]
	
	dist<-mahalanobis(mat,apply(newmat,2,mean),cov(newmat))
	dist2<-sort(dist)

	crit<-10
	jvec<-numeric(n)
	l<-0

	while(crit !=0 & l <=15){limit<-dist2[h]
		l<-l+1

		ivec<-ifelse(dist <= limit,1,0)	
		h2<-sum(ivec)
		diffvec<-abs(ivec-jvec)
		crit<-sum(diffvec)
		jvec<-ivec
	
		i<-0
		for(j in 1:n){	if(dist[j] <= limit){	i<-i+1
				newmat[i,]<-mat[j,]}}
		
	dist<-mahalanobis(mat,apply(newmat,2,mean),cov(newmat))
	dist2<-sort(dist)}

	tempdet<-det(cov(newmat))
	if(tempdet<bestdet){
		bestdet<-tempdet
		bestvec<-jvec
		bestdist<-dist2
		unsortdist<-dist}
}
	mcdmean<-apply(mat[bestvec==1,],2,mean)
	mcdcov<-cov(mat[bestvec==1,])	

	list(bestvec,bestdet,mcdmean,mcdcov)	# a list of [[1]] the points in the MCD
						# [[2]] the mcd determinant
						# [[3]] the mcd mean vector
						# [[4]] the mcd var-cov matrix
	
}

