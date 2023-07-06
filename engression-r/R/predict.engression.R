#' Prediction Function for Engression Models
#'
#' This function computes predictions from a trained engression model. It allows for the generation of point estimates, quantiles, 
#' or samples from the estimated distribution. 
#'
#' @param object A trained engression model returned from engression, engressionBagged or engressionfit functions.
#' @param Xtest A matrix or data frame representing the predictors in the test set.
#' @param type The type of prediction to make. "mean" for point estimates, "sample" for samples from the estimated distribution, 
#' or "quantile" for quantiles of the estimated distribution (default: "mean").
#' @param trim The proportion of extreme values to trim when calculating the mean (default: 0.05).
#' @param quantiles The quantiles to estimate if type is "quantile" (default: 0.1*(1:9)).
#' @param nsample The number of samples to draw if type is "sample" (default: 200).
#' @param drop A boolean indicating whether to drop dimensions of length 1 from the output (default: TRUE).
#' @param ... additional arguments (currently ignored)
#'
#' @return A matrix or array of predictions.
#'
#' @examples
#' \donttest{
#'   n = 1000
#'   p = 5
#' 
#'   X = matrix(rnorm(n*p),ncol=p)
#'   Y = (X[,1]+rnorm(n)*0.1)^2 + (X[,2]+rnorm(n)*0.1) + rnorm(n)*0.1
#'   Xtest = matrix(rnorm(n*p),ncol=p)
#'   Ytest = (Xtest[,1]+rnorm(n)*0.1)^2 + (Xtest[,2]+rnorm(n)*0.1) + rnorm(n)*0.1
#' 
#'   ## fit engression object
#'   engr = engression(X,Y)
#'   print(engr)
#' 
#'   ## prediction on test data
#'   Yhat = predict(engr,Xtest,type="mean")
#'   cat("\n correlation between predicted and realized values:  ", signif(cor(Yhat, Ytest),3))
#'   plot(Yhat, Ytest,xlab="prediction", ylab="observation")
#' 
#'   ## quantile prediction
#'   Yhatquant = predict(engr,Xtest,type="quantiles")
#'   ord = order(Yhat)
#'   matplot(Yhat[ord], Yhatquant[ord,], type="l", col=2,lty=1,xlab="prediction", ylab="observation")
#'   points(Yhat[ord],Ytest[ord],pch=20,cex=0.5)
#' 
#'   ## sampling from estimated model
#'   Ysample = predict(engr,Xtest,type="sample",nsample=1)
#'   
#' }
#' 
#' 
#' @export
predict.engression <- function(object,  Xtest, type=c("mean","sample","quantile")[1],trim=0.05, quantiles=0.1*(1:9), nsample=200, drop=TRUE, ...){

    if (is.data.frame(Xtest)) Xtest = dftomat(Xtest)
    if (is.vector(Xtest) && is.numeric(Xtest)) Xtest <- matrix(Xtest, ncol = 1)

    if(object$standardize){
        Xtest  = sweep(sweep(Xtest,2,object$muX,FUN="-"),2,object$sddX,FUN="/")
    }

    Yhat1 = object$engressor(Xtest)  
    Yhat = array(dim=c(dim(Yhat1)[1], dim(Yhat1)[2], nsample))
    for (sam in 1:nsample)  Yhat[, ,sam] = if(!object$standardize) object$engressor(Xtest) else sweep(sweep(object$engressor(Xtest),2,object$sddY,FUN="*"),2,object$muY,FUN="+")

    if(type=="sample") dimnames(Yhat)[[3]] = paste("sample_",1:nsample,sep="") 
    if(type=="mean") Yhat = apply(Yhat,1:(length(dim(Yhat))-1), mean,trim=trim)
    if(type %in% c("quantile","quantiles")){
        if(length(quantiles)==1){
            Yhat = apply(Yhat,1:(length(dim(Yhat))-1), quantile, quantiles)
        }else{
            Yhat = aperm( apply(Yhat,1:(length(dim(Yhat))-1), quantile, quantiles), if(length(dim(Yhat)==3)) c(2,3,1) else c(2,1) )
        } 
    }

    return(if(drop) drop(Yhat) else Yhat)

}
