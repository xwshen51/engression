# Engression

Engression is a neural network-based distributional regression method proposed in the paper "[*Engression: Extrapolation through the Lens of Distributional Regression?*](https://arxiv.org/abs/2307.00835)" by Xinwei Shen and Nicolai Meinshausen (2023). This repository contains the software implementations of engression in both R and Python. 

Consider targets $Y\in\mathbb{R}^k$ and predictors $X\in\mathbb{R}^d$; both variables can be univariate or multivariate, continuous or discrete. Engression can be used to 
* estimate the conditional mean $\mathbb{E}[Y|X=x]$ (as in least-squares regression), 
* estimate the conditional quantiles of $Y$ given $X=x$ (as in quantile regression), and 
* sample from the fitted conditional distribution of $Y$ given $X=x$ (as a generative model).

The results in the paper show the advantages of engression over existing regression approaches in terms of extrapolation. 
 

## Installation

### Python package
The latest release of the Python package can be installed via pip:
```sh
pip install engression
```

The development version can be installed from github:

```sh
pip install -e "git+https://github.com/xwshen51/engression#egg=engression&subdirectory=engression-python" 
```

### R package

The latest release of the R package can be installed through CRAN:

```R
install.packages("engression")
```

The development version can be installed from github:

```R
devtools::install_github("xwshen51/engression", subdir = "engression-r")
```


## Usage Example

### Python
Below is one simple demonstration. See [this tutorial](https://github.com/xwshen51/engression/blob/main/engression-python/examples/example_simu.ipynb) for more details on simulated data and [this tutorial](https://github.com/xwshen51/engression/blob/main/engression-python/examples/example_air.ipynb) for a real data example. We demonstrate in [another tutorial](https://github.com/xwshen51/engression/blob/main/engression-python/examples/example_bag.ipynb) how to fit a bagged engression model, which also helps with hyperparameter tuning.
```python
from engression import engression
from engression.data.simulator import preanm_simulator

## Simulate data
x, y = preanm_simulator("square", n=10000, x_lower=0, x_upper=2, noise_std=1, train=True, device=device)
x_eval, y_eval_med, y_eval_mean = preanm_simulator("square", n=1000, x_lower=0, x_upper=4, noise_std=1, train=False, device=device)

## Fit an engression model
engressor = engression(x, y, lr=0.01, num_epoches=500, batch_size=1000, device="cuda")
## Summarize model information
engressor.summary()

## Evaluation
print("L2 loss:", engressor.eval_loss(x_eval, y_eval_mean, loss_type="l2"))
print("correlation between predicted and true means:", engressor.eval_loss(x_eval, y_eval_mean, loss_type="cor"))

## Predictions
y_pred_mean = engressor.predict(x_eval, target="mean") ## for the conditional mean
y_pred_med = engressor.predict(x_eval, target="median") ## for the conditional median
y_pred_quant = engressor.predict(x_eval, target=[0.025, 0.5, 0.975]) ## for the conditional 2.5% and 97.5% quantiles
```

### R
```R
require(engression)
n = 1000
p = 5

X = matrix(rnorm(n*p),ncol=p)
Y = (X[,1]+rnorm(n)*0.1)^2 + (X[,2]+rnorm(n)*0.1) + rnorm(n)*0.1
Xtest = matrix(rnorm(n*p),ncol=p)
Ytest = (Xtest[,1]+rnorm(n)*0.1)^2 + (Xtest[,2]+rnorm(n)*0.1) + rnorm(n)*0.1

## fit engression object
engr = engression(X,Y)
print(engr)

## prediction on test data
Yhat = predict(engr,Xtest,type="mean")
cat("\n correlation between predicted and realized values:  ", signif(cor(Yhat, Ytest),3))
plot(Yhat, Ytest,xlab="prediction", ylab="observation")

## quantile prediction
Yhatquant = predict(engr,Xtest,type="quantiles")
ord = order(Yhat)
matplot(Yhat[ord], Yhatquant[ord,], type="l", col=2,lty=1,xlab="prediction", ylab="observation")
points(Yhat[ord],Ytest[ord],pch=20,cex=0.5)

## sampling from estimated model
Ysample = predict(engr,Xtest,type="sample",nsample=1)
par(mfrow=c(1,2))
## plot of realized values against first variable
plot(Xtest[,1], Ytest, xlab="Variable 1", ylab="Observation")
## plot of sampled values against first variable
plot(Xtest[,1], Ysample, xlab="Variable 1", ylab="Sample from engression model")   
```


## Contact information
If you meet any problems with the code, please submit an issue or contact [Xinwei Shen](mailto:xinwei.shen@stat.math.ethz.ch).
