# Engression

Engression is a nonlinear regression methodology proposed in the paper "[*Engression: Extrapolation for Nonlinear Regression?*]()" by Xinwei Shen and Nicolai Meinshausen. 
This repository contains the software implementations of engression in both R and Python.

## Installation
### R package

The latest release of the R package can be installed through CRAN (coming soon):

```R
install.packages("engression")
```

The development version can be installed from github

```R
devtools::install_github("xwshen51/engression", subdir = "engression-r")
```

### Python package
The latest release of the Python package can be installed through pip:
```sh
pip install engression
```

The development version can be installed from github

```sh
$ git clone https://github.com/xwshen51/engression.git  # Download the package 
$ cd engression/engression-python
$ pip install -r requirements.txt  # Install the requirements
$ python setup.py install develop --user
```


## Usage Example
### R
```R
require(engression)

```

### Python
```python
from engression import engression
from engression.data.simulator import preanm_simulator

# Simulate data
x, y = preanm_simulator("square", n=10000, x_lower=0, x_upper=2, noise_std=1, train=True, device=device)
x_eval, y_eval_med, y_eval_mean = preanm_simulator("square", n=1000, x_lower=0, x_upper=4, noise_std=1, train=False, device=device)

# Build an engression model and train
engressor = engression(x, y, lr=0.01, num_epoches=500, batch_size=1000, device="cuda")
engressor.summary()

# Evaluation
print("L2 loss:", engressor.eval_loss(x_eval, y_eval_mean, loss_type="l2"))
print("correlation between predicted and true means:", engressor.eval_loss(x_eval, y_eval_mean, loss_type="cor"))

# Predictions
y_pred = engressor.predict(x_eval, target="mean")
```


## Contact information
If you meet any problems with the code, please submit an issue or contact [Xinwei Shen](mailto:xinwei.shen@stat.math.ethz.ch).