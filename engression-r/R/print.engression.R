#' Print an Engression Model Object
#'
#' This function is a utility that displays a summary of a fitted Engression model object.
#'
#' @param x A trained engression model returned from the engressionfit function.
#' @param ... additional arguments (currently ignored)
#'
#' @return This function does not return anything. It prints a summary of the model, 
#' including information about its architecture and training process, and the loss 
#' values achieved at several epochs during training.
#'
#' @examples
#' \donttest{
#'   n = 1000
#'   p = 5
#' 
#'   X = matrix(rnorm(n*p),ncol=p)
#'   Y = (X[,1]+rnorm(n)*0.1)^2 + (X[,2]+rnorm(n)*0.1) + rnorm(n)*0.1
#'   
#'   ## fit engression object
#'   engr = engression(X,Y)
#'   print(engr)
#' }
#' 
#' @export
print.engression <- function(x, ...){
    cat("\n engression object with ")
    cat("\n \t  noise dimensions: ",x$noise_dim)
    cat("\n \t  hidden dimensions: ",x$hidden_dim)
    cat("\n \t  number of layers: ",x$num_layer)
    cat("\n \t  dropout rate: ",x$dropout)
    cat("\n \t  batch normalization: ",x$batch_norm)
    cat("\n \t  number of epochs: ",x$num_epochs)
    cat("\n \t  learning rate: ",x$lr)
    cat("\n \t  standardization: ",x$standardize)

    m = nrow(x$lossvec)
    printat = pmax(1, floor(seq(1,m, length=11)))
    pr = cbind(printat, x$lossvec[printat,])
    colnames(pr) = c("epoch", colnames(x$lossvec))
    cat("\n training loss: \n")
    print(pr)
    cat("\n prediction-loss E(|Y-Yhat|) and variance-loss E(|Yhat-Yhat'|)should ideally be equally large --\n consider training for more epochs if there is a mismatch \n\n")

}
