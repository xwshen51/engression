#' Print a Bagged Engression Model Object
#'
#' This function displays a summary of a bagged Engression model object. The 
#' summary includes details about the individual models as well as the overall 
#' ensemble.
#'
#' @param x A trained bagged engression model object returned from 
#'   the engressionBagged function.
#' @param ... additional arguments (currently ignored)
#'
#' @return This function does not return anything. It prints a summary of the 
#'   model, including the architecture of the individual models, the number 
#'   of models in the bagged ensemble, and the loss values achieved at several 
#'   epochs during training.
#'
#' @examples
#' \dontrun{
#'   n = 1000
#'   p = 5
#'   X = matrix(rnorm(n*p),ncol=p)
#'   Y = (X[,1]+rnorm(n)*0.1)^2 + (X[,2]+rnorm(n)*0.1) + rnorm(n)*0.1
#'   
#'   ## fit bagged engression object
#'   engb = engressionBagged(X,Y,K=3)
#'   print(engb)
#' 
#' }
#' 
#' @export
print.engressionBagged <- function(x, ...){
    cat("\n bagged engression object with", length(x$models), "models")
    cat("\n \t  noise dimensions: ",x$noise_dim)
    cat("\n \t  hidden dimensions: ",x$hidden_dim)
    cat("\n \t  number of layers: ",x$num_layer)
    cat("\n \t  dropout rate: ",x$dropout)
    cat("\n \t  number of epochs: ",x$num_epochs)
    cat("\n \t  learning rate: ",x$lr)
    cat("\n \t  standardization: ",x$standardize)

    avloss = Reduce("+",lapply(x$models, function(x) x$lossvec))/length(x$models)
    m = nrow(avloss)
    printat = pmax(1,floor((seq(1,m, length=11))))
    pr = cbind(printat, avloss[printat,])
    colnames(pr) = c("epoch", colnames(avloss))
    cat("\n average training loss : \n")
    print(pr)
    cat("\n prediction-loss E(|Y-Yhat|) and variance-loss E(|Yhat-Yhat'|)should ideally be equally large --\n consider training for more epochs if there is a mismatch \n\n")

}
