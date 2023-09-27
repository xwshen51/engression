#' Bagged Engression Function
#'
#' This function fits a bagged engression model to the data by fitting multiple
#' engression models to subsamples of the data. It allows for the tuning of several parameters 
#' related to model complexity.
#'
#' @param X A matrix or data frame representing the predictors.
#' @param Y A matrix or vector representing the target variable(s).
#' @param K The number of bagged models to fit (default: 5).
#' @param keepoutbag A boolean indicating whether to keep the out-of-bag samples and training data (default: TRUE).
#' @param noise_dim The dimension of the noise introduced in the model (default: 10).
#' @param hidden_dim The size of the hidden layer in the model (default: 100).
#' @param num_layer The number of layers in the model (default: 3).
#' @param dropout The dropout rate to be used in the model (default: 0.05).
#' @param batch_norm A boolean indicating whether to use batch-normalization (default: TRUE).
#' @param num_epochs The number of epochs to be used in training (default: 1000).
#' @param lr The learning rate to be used in training (default: 10^-3).
#' @param beta The beta scaling factor for energy loss (default: 1).
#' @param silent A boolean indicating whether to suppress output during model training (default: FALSE).
#' @param standardize A boolean indicating whether to standardize the input data (default: TRUE).
#'
#' @return A bagged engression model object with class "engressionBagged".
#'
#' @examples
#' \donttest{
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
#'   plot(Yhat_oob, Y,xlab="prediction", ylab="observation")
#' 
#'   ## quantile prediction
#'   Yhatquant = predict(engb,Xtest,type="quantiles")
#'   ord = order(Yhat)
#'   matplot(Yhat[ord], Yhatquant[ord,], type="l", col=2,lty=1,xlab="prediction", ylab="observation")
#'   points(Yhat[ord],Ytest[ord],pch=20,cex=0.5)
#' 
#'   ## sampling from estimated model
#'   Ysample = predict(engb,Xtest,type="sample",nsample=1)
#' 
#'   ## plot of realized values against first variable
#'   oldpar <- par()
#'   par(mfrow=c(1,2))
#'   plot(Xtest[,1], Ytest, xlab="Variable 1", ylab="Observation")
#'   ## plot of sampled values against first variable
#'   plot(Xtest[,1], Ysample[,1], xlab="Variable 1", ylab="Sample from engression model")  
#'   par(oldpar)
#' }
#'
#' @export
#' 
engressionBagged <- function(X,Y, K=5, keepoutbag=TRUE, noise_dim=10, hidden_dim=100, num_layer=3, dropout=0.05, batch_norm=TRUE, num_epochs=1000,lr=10^(-3),beta=1, silent=FALSE, standardize=TRUE){

    if (is.data.frame(X)) {
        if (any(sapply(X, is.factor)))   warning("Data frame contains factor variables. Mapping to numeric values. Dummy variables would need to be created explicitly by the user.")
        X <- dftomat(X)
    }
    if (is.vector(X) && is.numeric(X)) X <- matrix(X, ncol = 1)
    if(is.vector(Y)) Y= matrix(Y, ncol=1)
    for (k in 1:ncol(Y)) Y[,k] = as.numeric(Y[,k])


    if(dropout<=0 & noise_dim==0){
        warning("dropout and noise_dim cannot both be equal to 0 as model needs to be stochastic. setting dropout to 0.5")
        dropout = 0.5
    }

    inbagno = min(K-1,ceiling(K*0.8))
    inbag = matrix(nrow=nrow(X), ncol=inbagno)
    for (i in 1:nrow(X)) inbag[i,] = sort(sample(1:K,inbagno))

    models = list()
    for (k in 1:K){
        if(k==1) pr="st"
        if(k==2) pr="nd"
        if(k==3) pr="rd"
        if(k>=4) pr="th"
        if(!silent) cat(paste("\n fitting ",k,"-", pr," out of ",K," engression models \n",sep=""))
        useinbag = which(apply(inbag==k,1,any))
        models[[k]] = engression(X[useinbag,],Y[useinbag],  noise_dim=noise_dim, hidden_dim=hidden_dim, num_layer=num_layer, dropout=dropout, batch_norm=batch_norm, num_epochs=num_epochs,lr=lr,beta=beta, silent=silent, standardize=standardize)
    }

    engBagged = list(models= models, inbag=if(keepoutbag) inbag else NULL, Xtrain=if(keepoutbag) X else NULL, noise_dim=noise_dim,hidden_dim=hidden_dim,num_layer=num_layer,dropout=dropout, batch_norm=batch_norm, num_epochs=num_epochs,lr=lr, standardize=standardize)
    class(engBagged) = "engressionBagged"
    print(engBagged)
    return(engBagged)
}
