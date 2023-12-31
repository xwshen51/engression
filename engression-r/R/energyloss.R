#' Energy Loss Calculation
#'
#' This function calculates the energy loss for given tensors. The loss is calculated
#' as the mean of the L2 norms between `yt` and `mxt` and between `yt` and `mxpt`,
#' subtracted by half the mean of the L2 norm between `mxt` and `mxpt`.
#' 
#' @param yt A tensor representing the target values.
#' @param mxt A tensor representing the model's stochastic predictions.
#' @param mxpt A tensor representing another draw of the model's stochastic predictions.
#' 
#' @return A scalar representing the calculated energy loss.
#' 
#' 
#' @keywords internal
#' 
energyloss <- function(yt,mxt,mxpt){
  s1 = torch_mean(torch_norm(yt - mxt, 2, dim=2)) / 2 + torch_mean(torch_norm(yt - mxpt, 2, dim=2)) / 2
  s2 = torch_mean(torch_norm(mxt - mxpt, 2, dim=2))
  return (s1 - s2/2)
}
