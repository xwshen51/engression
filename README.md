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
Below is one simple demonstration. See [this tutorial](https://github.com/xwshen51/engression/blob/main/engression-python/examples/example_simu.ipynb) for more details on simulated data and [this tutorial](https://github.com/xwshen51/engression/blob/main/engression-python/examples/example_air.ipynb) for a real data example.
```python
from engression import engression
from engression.data.simulator import preanm_simulator

## Simulate data
x, y = preanm_simulator("square", n=10000, x_lower=0, x_upper=2, noise_std=1, train=True, device="cpu")
x_eval, y_eval_med, y_eval_mean = preanm_simulator("square", n=1000, x_lower=0, x_upper=4, noise_std=1, train=False, device="cpu")

## Fit an engression model
engressor = engression(x, y, lr=0.01, num_epochs=500, batch_size=1000, device="cpu")
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

Validation data can be passed to choose the number of training iterations: `engression(x, y, x_val=x_val, y_val=y_val)` computes the energy loss on them about every 50 iterations and after the last epoch, and keeps the parameters with the lowest loss; `engressor.summary()` shows the epoch they come from.

### Generalized engression for other types of responses (Python)

Generalized engression models (GEMs), proposed in the paper "[*Generalized Engression Models*](https://arxiv.org/abs/2610.01823)", extend engression to responses that are not continuous: binary labels, categorical and ordinal variables, rankings, and vectors that mix several types. A GEM is the generative model $Y=h(g(X,\varepsilon)+\sigma(X)\odot\eta)$, where $g$ is the engression network with noise $\varepsilon$, the link $h$ maps its continuous output to the type of the response, and $\eta$ is Gaussian noise with a learned scale $\sigma(X)$. It is fitted by the same energy loss, and the same call serves every type; only the argument `data_type` changes. For continuous responses, the method is identical to engression.
```python
from engression import engression

fit = engression(x, y)                                     # continuous response
fit = engression(x, y, data_type="multilabel")             # several binary labels
fit = engression(x, y, data_type="ranking")                # a ranking of the columns of y
fit = engression(x, y, data_type={"continuous": 0,         # mixed response: one link
                                  "multilabel": 7,         # per block, each given by
                                  "multiclass": 12})       # its first column

y_draws = fit.sample(x_new, sample_size=1000)              # draws of Y given x_new
```

The response `y` is a tensor with one row per observation, coded as follows.

| `data_type` | response | coding of `y` | link |
|---|---|---|---|
| `"continuous"` | real values | the values | identity |
| `"multilabel"` | several binary labels | one column per label, all coded as 0/1 or all as -1/+1 | sign |
| `"multiclass"` | one categorical variable | indicator (one-hot) vectors, one column per class | largest coordinate |
| `"ordinal:L"` | ordinal variables with `L` levels | one column per variable, with levels 1, ..., `L` | rounding to the nearest level |
| `"ranking"` | a ranking of $k$ items | rank vectors: column $i$ is the position of item $i$, where 1 is the top | ranks of the coordinates |

For a mixed response, `data_type` is a dictionary with the data types as keys and the first column of each block as values; a list of pairs, such as `[("multiclass", 0), ("multiclass", 3)]`, is used when a type appears more than once.

As for a continuous response, `fit.sample` draws from the fitted conditional distribution, in the coding of `y`, and `fit.predict` and `fit.eval_loss` are computed from such draws. In particular, `fit.predict(x_new, target="mean")` gives the probabilities of the labels coded as 0/1 and of the classes, and the mean level or position otherwise. When `standardize=True`, only the continuous columns of `y` are standardized.

Remarks on fitting a GEM:
* GEMs are new in version 1.1.0, and their defaults may change in later versions.
* `control_variate=True` reduces the variance of the gradient estimates for binary labels and is recommended when there are many labels.
* `sigma_dim="vector"` lets each coordinate of the response have its own scale $\sigma(X)$ instead of a common one, for instance for a mixed response. `sigma_min` is the lower bound of the scale on the discrete coordinates. It caps the probability of an interior level of an ordinal variable, at 98.8% with the default 0.2.
* `classification=True` is obsolete and gives a warning. It is a different model, in which the network outputs class probabilities; for a categorical response, use `data_type="multiclass"` instead.

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
If you meet any problems with the code, please submit an issue or contact [Xinwei Shen](mailto:xwshen@uw.edu).


## Citation
If you would refer to or extend our work, please cite the following paper:
```
@article{10.1093/jrsssb/qkae108,
    author = {Shen, Xinwei and Meinshausen, Nicolai},
    title = {Engression: extrapolation through the lens of distributional regression},
    journal = {Journal of the Royal Statistical Society Series B: Statistical Methodology},
    pages = {qkae108},
    year = {2024},
    month = {11},
    issn = {1369-7412},
    doi = {10.1093/jrsssb/qkae108},
    url = {https://doi.org/10.1093/jrsssb/qkae108},
    eprint = {https://academic.oup.com/jrsssb/advance-article-pdf/doi/10.1093/jrsssb/qkae108/60827977/qkae108.pdf},
}
```