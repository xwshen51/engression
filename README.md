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
$ cd engression
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
import engression

```


## Contact information
If you meet any problems with the code, please submit an issue or contact Xinwei Shen (`xinwei.shen@stat.math.ethz.ch`).