#' Energy Loss Calculation (Extended Output)
#'
#' This function calculates the energy loss for given tensors, similar to `energyloss()`. The loss is calculated
#' as the mean of the L2 norms between `yt` and `mxt` and between `yt` and `mxpt`,
#' subtracted by half the mean of the L2 norm between `mxt` and `mxpt`. Unlike `energyloss()`, this function
#' also returns the prediction loss s1 = E(|yt-mxt|) and variance loss s2 = E(|mxt-mxpt'|) as part of the output.
#' 
#' @param yt A tensor representing the target values.
#' @param mxt A tensor representing the model's stochastic predictions.
#' @param mxpt A tensor representing another draw of the model's stochastic predictions.
#' 
#' @return A vector containing the calculated energy loss, `s1`, and `s2`.
#' 
#' 
#' @keywords internal
energylossall <- function(yt,mxt,mxpt){
  s1 = torch_mean(torch_norm(yt - mxt, 2, dim=1)) / 2 + torch_mean(torch_norm(yt - mxpt, 2, dim=1)) / 2
  s2 = torch_mean(torch_norm(mxt - mxpt, 2, dim=1))
  return (c((s1 - s2/2),s1,s2))
}
