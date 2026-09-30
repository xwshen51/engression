# Changelog

## 1.0.0

A stable release, with bug fixes and a few new features, chiefly the choice of the training epoch on validation data. Fits with the default settings are unchanged at a fixed seed. Fits change for networks that shared weights, with `out_act`, for data with more than two dimensions, and when a fitted model is trained again on more data; the reported training loss changes when `beta` is not 1.

- New: `engression` and `Engressor.train` take validation data, `x_val` and `y_val`. The energy loss on them is computed about every 50 iterations, at the end of an epoch, and after the last epoch, and the parameters with the lowest loss are kept; their epoch is in the attribute `best_epoch` and printed by `summary`. On the CPU and on CUDA devices, validation does not change the random numbers of training, so that the kept parameters are those that a fit without validation data has after that epoch.
- New: the data can be numpy arrays, lists or tensors of any numeric type, which `engression`, `Engressor.train`, `predict`, `sample`, `eval_loss` and `plot` convert to float32 tensors. Numpy arrays, float64 tensors and integer responses raised errors.
- New: `from engression import Engressor` works, and `engression.__version__` gives the version.
- The intermediate layers of `StoNet`, the network of `engression` and `Engressor`, and of `ResMLP` no longer share their weights. Before, a `StoNet` with `num_layer >= 4`, or `num_layer >= 7` with `resblock=True` (an odd `num_layer` is rounded up), and a `ResMLP` with `num_layer >= 7` applied one set of weights in all intermediate layers, and so had fewer parameters than intended. Models saved by earlier versions, whole or with `state_dict()`, still load and give the same predictions. For such networks, an optimizer state saved by an earlier version does not load, and a `state_dict()` saved by this version loads into earlier versions without an error but gives other predictions.
- `out_act` now acts on the response on its original scale. Before, with the default `standardize=True`, it acted on the standardized response, so that `out_act="relu"`, for example, gave draws no lower than the mean of the training responses instead of no lower than zero. Fits with `out_act` change.
- `engression(x, y)` works for `x` and `y` with one dimension or with more than two, which it flattens to one row per observation, as `Engressor.train` does. A one-dimensional `x` or `y` raised an error, and for an `x` of shape `(n, 2, 5)`, for example, the network was built for 2 inputs instead of 10.
- A full-batch fit no longer records the sample size as the batch size, which made a later `train` on more data use mini-batches; `summary` prints "full batch".
- The training loss reported after fitting is the energy loss with the `beta` of the fit; it was computed with `beta=1`.
- `eval_loss(..., loss_type="energy")` and `loss_func.energy_loss` are exact. For a `beta` that is not an integer, the term E(|Yhat-Yhat'|) counted the distance of each draw to itself as `1e-5 ** beta` instead of 0. This made the loss too small, by 0.16 for `beta=0.1` with the default two draws, and by 0.0016 for `beta=0.5`. With more than 25 draws per observation, the distances between draws lost precision for responses far from zero. With one draw per observation, it raises an error instead of returning nan. Fits were not affected.
- After running out of GPU memory, `sample(..., expand_dim=False)` returned the draws in the wrong order. As a result, `eval_loss(..., loss_type="energy")` compared some responses with draws for other observations, and `plot(..., target="sample")` drew points at the wrong `x`.
- A covariate that is constant in the training data is centered but no longer scaled. It was divided by 1e-5, so that a slightly different value in new data, say 1.1 where the training data had 1, entered the network as 10,000 and gave absurd predictions.
- `preanm_simulator(..., train=False)` computes the true conditional mean with the noise of `noise_dist`; it used Gaussian noise.
- `sample`, `predict` and `eval_loss` no longer run forever without output when the network raises an error other than running out of memory, for example for float64 input; the error is raised.
- `device="cuda"` on a machine without CUDA runs on the CPU, as the printed message says, instead of failing; with `device="mps"`, the message says MPS.
- `sample` works for models fitted with `classification=True`, and `sample(..., sample_size=1, expand_dim=False)` keeps the response dimension of a univariate response, so that `plot(..., target="sample", sample_size=1)` works.
- `eval_loss(..., verbose=True)` works for `loss_type` `"l2"`, `"l1"` and `"cor"`.
- Mini-batch training no longer fails when `print_times_per_epoch` exceeds the number of batches minus one, or, with batch normalization, when the last batch has one observation; that observation is then left out of the epoch. For this, `data.loader.make_dataloader` has a new argument `drop_last`.
- `plot(..., save_dir=...)` saves the figure, and `plot` with training data accepts one-dimensional test data.
- `summary` works before training.
- `Engressor.train` follows the `verbose` argument of the `Engressor` unless given its own; it printed unless called with `verbose=False`. A model fitted by `engression(..., verbose=False)` and trained again prints only with `verbose=True`.
- A `StoNet` with `num_layer=1` prints that it has 2 layers, i.e. one hidden layer, as with `num_layer=2`; the network is unchanged.
- `models.StoNetBase.compute_cdf` accepts one-dimensional `x` and `y`.
- The source package on PyPI (the `.tar.gz` file) includes `requirements.txt`, so that installing from it works. The usual `pip install` uses the wheel and was not affected.
- The package declares that it needs Python 3.9 or later, and its PyPI page shows classifiers: development status, audience and topic.
