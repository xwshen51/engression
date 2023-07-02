#' Convert Data Frame to Numeric Matrix
#'
#' This function converts a data frame into a numeric matrix. If the data frame 
#' contains factor or character variables, they are first converted to numeric.
#' 
#' @param X A data frame to be converted to a numeric matrix.
#' 
#' @return A numeric matrix corresponding to the input data frame.
#' 
#' 
#' @keywords internal
#' 
dftomat <- function(X){
    X <- data.frame(lapply(X, function(x){
    if (is.factor(x)){
        as.numeric(as.character(x))
    }else if(is.character(x)){
        as.numeric(as.factor(x))
    }else{
        as.numeric(x)
    }
    }))
    X = as.matrix(X)
    return(X)
}    
