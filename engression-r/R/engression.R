#' Engression Function
#'
#' This function fits an engression model to the data. It allows for
#' the tuning of several parameters related to model complexity.
#' Variables are per default internally standardized (predictions are on original scale).
#'
#' @param X A matrix or data frame representing the predictors.
#' @param Y A matrix or vector representing the target variable(s).
#' @param noise_dim The dimension of the noise introduced in the model (default: 5).
#' @param hidden_dim The size of the hidden layer in the model (default: 100).
#' @param num_layer The number of layers in the model (default: 3).
#' @param dropout The dropout rate to be used in the model in case no batch normalization is used (default: 0.01)
#' @param batch_norm A boolean indicating whether to use batch-normalization (default: TRUE).
#' @param num_epochs The number of epochs to be used in training (default: 1000).
#' @param lr The learning rate to be used in training (default: 10^-3).
#' @param beta The beta scaling factor for energy loss (default: 1).
#' @param silent A boolean indicating whether to suppress output during model training (default: FALSE).
#' @param standardize A boolean indicating whether to standardize the input data (default: TRUE).
#'
#' @return An engression model object with class "engression".
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
#'   ## plot of realized values against first variable
#'   oldpar <- par()
#'   par(mfrow=c(1,2))
#'   plot(Xtest[,1], Ytest, xlab="Variable 1", ylab="Observation")
#'   ## plot of sampled values against first variable
#'   plot(Xtest[,1], Ysample, xlab="Variable 1", ylab="Sample from engression model")   
#'   par(oldpar)
#' }
#' 
#' @export

engression <- function(X,Y,  noise_dim=5, hidden_dim=100, num_layer=3, dropout=0.05, batch_norm=TRUE, num_epochs=1000,lr=10^(-3),beta=1, silent=FALSE, standardize=TRUE){

    if (is.data.frame(X)) {
        if (any(sapply(X, is.factor)))   warning("Data frame contains factor variables. Mapping to numeric values. Dummy variables would need to be created explicitly by the user.")
        X = dftomat(X)
    }

    if (is.vector(X) && !is.numeric(X)) X <- as.numeric(X)
    if (is.vector(X) && is.numeric(X)) X <- matrix(X, ncol = 1)
    if(is.vector(Y)) Y= matrix(Y, ncol=1)
    for (k in 1:ncol(Y)) Y[,k] = as.numeric(Y[,k])

    if(dropout<=0 & noise_dim==0){
        warning("dropout and noise_dim cannot both be equal to 0 as model needs to be stochastic. setting dropout to 0.5")
        dropout = 0.5
    }

    muX = apply(X,2,mean)
    sddX = apply(X,2,sd)
    if(any(sddX<=0)){
        warning("predictor variable(s) ", colnames(X)[which(sddX<=0)]," are constant on training data -- results might be unreliable")
        sddX = pmax(sddX, 10^(03))
    }
    muY = apply(Y,2,mean)
    sddY = apply(Y,2,sd)
    if(any(sddY<=0)){
        warning("target variable(s) ", colnames(Y)[which(sddY<=0)]," are constant on training data -- results might be unreliable")
    }

    if(standardize){
        X  = sweep(sweep(X,2,muX,FUN="-"),2,sddX,FUN="/")
        Y = sweep(sweep(Y,2,muY,FUN="-"),2,sddY,FUN="/")
    }
    eng = engressionfit(X,Y, noise_dim=noise_dim,hidden_dim=hidden_dim,num_layer=num_layer,dropout=dropout, batch_norm=batch_norm, num_epochs=num_epochs,lr=lr,beta=beta, silent=silent)
    engressor = list(engressor = eng$engressor, lossvec= eng$lossvec,  muX=muX, sddX=sddX,muY=muY, sddY=sddY, standardize=standardize, noise_dim=noise_dim,hidden_dim=hidden_dim,num_layer=num_layer,dropout=dropout, batch_norm=batch_norm, num_epochs=num_epochs,lr=lr)
    class(engressor) = "engression"
    return(engressor)
}
