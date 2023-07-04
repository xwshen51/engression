# Engression

Engression is a nonlinear regression methodology proposed in the paper "[*Engression: Extrapolation for Nonlinear Regression?*](https://arxiv.org/abs/2307.00835)" by Xinwei Shen and Nicolai Meinshausen. 
This directory contains the Python implementation of engression.

## Installation
The latest release of the Python package can be installed through pip:
```sh
pip install engression
```

The development version can be installed from github:

```sh
pip install -e "git+https://github.com/xwshen51/engression#egg=engression&subdirectory=engression-python" 
```


## Usage Example

### Python

Below is one simple demonstration. See [this tutorial](https://github.com/xwshen51/engression/blob/main/engression-python/examples/example_simu.ipynb) for more details on simulated data and [this tutorial](https://github.com/xwshen51/engression/blob/main/engression-python/examples/example_air.ipynb) for a real data example.
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


## Contact information
If you meet any problems with the code, please submit an issue or contact [Xinwei Shen](mailto:xinwei.shen@stat.math.ethz.ch).