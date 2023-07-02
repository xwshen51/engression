#' Prediction Function for Bagged Engression Models
#'
#' This function computes predictions from a trained bagged Engression model. It allows for the generation of point estimates, 
#' quantiles, or samples from the estimated distribution. 
#'
#' @param object A trained bagged engression model returned from the engressionBagged function.
#' @param Xtest A matrix or data frame representing the predictors in the test set. If NULL, out-of-bag samples from the training 
#' set are used for prediction (default: NULL).
#' @param type The type of prediction to make. "mean" for point estimates, "sample" for samples from the estimated distribution, 
#' or "quantile" for quantiles of the estimated distribution (default: "mean").
#' @param trim The proportion of extreme values to trim when calculating the mean (default: 0.05).
#' @param quantiles The quantiles to estimate if type is "quantile" (default: 0.1*(1:9)).
#' @param nsample The number of samples to draw if type is "sample" (default: 200).
#' @param drop A boolean indicating whether to drop dimensions of length 1 from the output (default: TRUE).
#' @param ... additional arguments (currently ignored)
#' 
#' @return A matrix or array of predictions.
#'#'
#' @examples
#' \dontrun{
#'   n = 1000
#'   p = 5
#'   X = matrix(rnorm(n*p),ncol=p)
#'   Y = (X[,1]+rnorm(n)*0.1)^2 + (X[,2]+rnorm(n)*0.1) + rnorm(n)*0.1
#'   Xtest = matrix(rnorm(n*p),ncol=p)
#'   Ytest = (Xtest[,1]+rnorm(n)*0.1)^2 + (Xtest[,2]+rnorm(n)*0.1) + rnorm(n)*0.1
#' 
#'   ## fit bagged engression object
#'   engb = engressionBagged(X,Y,K=3)
#'   print(engb)
#' 
#'   ## prediction on test data
#'   Yhat = predict(engb,Xtest,type="mean")
#'   cat("\n correlation between predicted and realized values:  ", signif(cor(Yhat, Ytest),3))
#'   plot(Yhat, Ytest,xlab="estimated conditional mean", ylab="observation")
#' 
#'   ## out-of-bag prediction
#'   Yhat_oob = predict(engb,type="mean")
#'   cat("\n correlation between predicted and realized values on oob data:  ")
#'   print(signif(cor(Yhat_oob, Y),3))
#'   plot(Yhat_oob, Y,xlab="estimated conditional mean", ylab="observation")
#' 
#'   ## quantile prediction
#'   Yhatquant = predict(engb,Xtest,type="quantiles")
#'   ord = order(Yhat)
#'   matplot(Yhat[ord], Yhatquant[ord,], type="l", col=2,lty=1,xlab="prediction", ylab="observation")
#'   points(Yhat[ord],Ytest[ord],pch=20,cex=0.5)
#' 
#'   ## sampling from estimated model
#'   Ysample = predict(engb,Xtest,type="sample",nsample=1)
#'   par(mfrow=c(1,2))
#'   ## plot of realized values against first variable
#'   plot(Xtest[,1], Ytest, xlab="Variable 1", ylab="Observation")     
#'   ## plot of sampled values against first variable
#'   plot(Xtest[,1], Ysample, xlab="Variable 1", ylab="Sample from engression model")   
#' }
#' 
#' @export
predict.engressionBagged <- function(object, Xtest=NULL, type=c("mean","sample","quantile")[1],trim=0.05, quantiles=0.1*(1:9), nsample=200, drop=TRUE, ...){
    useoob=FALSE
    if(is.null(Xtest)){
        useoob = TRUE
        if(!is.null(object$Xtrain)) Xtest = object$Xtrain else stop("if Xtest is not provided, need to set keepoutbag=TRUE when fitting bagged engression model")
    }
    if (is.data.frame(Xtest))   Xtest = dftomat(Xtest)
    if (is.vector(Xtest) && is.numeric(Xtest)) Xtest <- matrix(Xtest, ncol = 1)
    
    K = length(object$models)
    
    nsam = if(useoob) 5*ceiling(nsample/K) else ceiling(nsample/K)
    Yhat1 = predict.engression(object$models[[1]],Xtest, type="sample", nsample=nsam, drop=FALSE)

    Yhat = array(dim=c(dim(Yhat1),K))
    for (k in 1:K){
        if(!useoob){
              Yhat[,,,k] = predict.engression(object$models[[k]],Xtest, type="sample", nsample=nsam, drop=FALSE)
        }else{
            usesam = which(apply(object$inbag!=k,1,all))
            Yhat[usesam,,,k] = predict.engression(object$models[[k]],Xtest[usesam,], type="sample", nsample=nsam, drop=FALSE)
        }
    }
    Yhat = aperm( apply(Yhat,c(1,2),as.vector  ),c(2,3,1)) 
    if(useoob) Yhat = aperm(apply(Yhat,1:2,function(x) x[which(!is.na(x))]),c(2,3,1))
    if(type=="sample")  dimnames(Yhat)[[3]] = paste("sample_",1:nsample,sep="") 
    if(type=="mean") Yhat = apply(Yhat,1:(length(dim(Yhat))-1), mean, trim=trim)
    if(type %in% c("quantile","quantiles")){
        if(length(quantiles)==1){
            Yhat = apply(Yhat,1:(length(dim(Yhat))-1), quantile, quantiles)
        }else{
            Yhat = aperm( apply(Yhat,1:(length(dim(Yhat))-1), quantile, quantiles), if(length(dim(Yhat)==3)) c(2,3,1) else c(2,1) )
        }
    }
    return(if(drop) drop(Yhat) else Yhat)
}