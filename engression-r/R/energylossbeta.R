#' Energy Loss Calculation with Beta Scaling
#'
#' This function calculates the energy loss for given tensors. The loss is calculated
#' as the mean of the L2 norms between `yt` and `mxt` and between `yt` and `mxpt`,
#' each raised to the power of `beta`, subtracted by half the mean of the L2 norm between `mxt` and `mxpt`,
#' also raised to the power of `beta`.
#' 
#' @param yt A tensor representing the target values.
#' @param mxt A tensor representing the model's stochastic predictions.
#' @param mxpt A tensor representing another draw of the model's stochastic predictions.
#' @param beta A numeric value for scaling the energy loss. 
#' 
#' @return A scalar representing the calculated energy loss.
#' 
#' @keywords internal
#' 
energylossbeta <- function(yt,mxt,mxpt,beta){
  s1 = torch_pow(torch_mean(torch_norm(yt - mxt, 2, dim=1)),beta) / 2 + torch_pow(torch_mean(torch_norm(yt - mxpt, 2, dim=1)),beta) / 2
  s2 = torch_pow(torch_mean(torch_norm(mxt - mxpt, 2, dim=1)),beta)
  return (s1 - s2/2)
}
