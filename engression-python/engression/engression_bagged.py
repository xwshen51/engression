import torch
import numpy as np
import matplotlib.pyplot as plt

from .engression import Engressor
from .loss_func import *
from .utils import *


def engression_bagged(x, y, 
                      num_layer=2, hidden_dim=100, noise_dim=100,
                      lr=0.001, num_epoches=500, batch_size=None, 
                      device="cpu", standardize=True,
                      ensemble_size=10, val_loss_type="energy"):
    """This function fits a bagged engression model to the data by aggregating multiple engression models fitted on subsamples of the data. It calculates validation losses that helps with hyperparameter tuning.

    Args:
        x (torch.Tensor): training data of predictors.
        y (torch.Tensor): training data of responses.
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        lr (float, optional): learning rate. Defaults to 0.001.
        num_epoches (int, optional): number of epochs. Defaults to 500.
        batch_size (int, optional): batch size. Defaults to None.
        device (str, torch.device, optional): device. Defaults to "cpu". Choices = ["cpu", "gpu", "cuda"].
        standardize (bool, optional):  whether to standardize data for training. Defaults to True.
        ensemble_size (int, optional): number of ensemble models. Defaults to 10.
        val_loss_type (str, optional): loss type for validation. Defaults to "energy". Choices: ["l1", "l2", "energy"].

    Returns:
        BaggedEngressor object: a fitted bagged engression model.
    """
    engressor = BaggedEngressor(in_dim=x.shape[1], out_dim=y.shape[1], 
                                num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim, 
                                lr=lr, num_epoches=num_epoches, batch_size=batch_size, 
                                device=device, standardize=standardize, 
                                ensemble_size=ensemble_size, val_loss_type=val_loss_type)
    engressor.train(x, y)
    return engressor


class BaggedEngressor(object):
    """Bagged engressor.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        lr (float, optional): learning rate. Defaults to 0.001.
        num_epoches (int, optional): number of epoches. Defaults to 500.
        batch_size (int, optional): batch size. Defaults to None, referring to the full batch.
        device (str or torch.device, optional): device. Defaults to "cpu".
        standardize (bool, optional): whether to standardize data. Defaults to True.
        ensemble_size (int, optional): number of models for ensemble. Defaults to 10.
        val_loss_type (str, optional): loss type for validation. Defaults to "energy". Choices: ["l1", "l2", "energy"].
    """
    def __init__(self, 
                 in_dim, out_dim, num_layer=2, hidden_dim=100, noise_dim=100,
                 lr=0.001, num_epoches=500, batch_size=None, 
                 device="cpu", standardize=True,
                 ensemble_size=10, val_loss_type="energy"): 
        super().__init__()
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.lr = lr
        self.num_epoches = num_epoches
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.num_models_for_each_sample = ensemble_size // 2 
        # As such, each model has around 50% training data and the remaining 50% as validation data. 
        
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.standardize = standardize
        self.split_mask = None
        
        # Build ensemble models.
        self.models = []        
        for i in range(ensemble_size):
            check_device = i == 0 
            self.models.append(Engressor(in_dim, out_dim, num_layer, hidden_dim, noise_dim, 
                                         lr=lr, standardize=standardize, device=device, check_device=check_device))
        self.val_loss_type = val_loss_type
        
        self.train_losses = [] # training losses of each model
        self.val_losses = [] # validation losses of each model
        self.val_loss_final = None # validation loss of aggregated model on the entire training data set

    def train_mode(self):
        for i in range(self.ensemble_size):
            self.models[i].train_mode()
        
    def eval_mode(self):
        for i in range(self.ensemble_size):
            self.models[i].eval_mode()

    def summary(self):
        """Print the model architecture and hyperparameters."""
        print("Engression model with\n" +
              "\t number of layers: {}\n".format(self.num_layer) +
              "\t hidden dimensions: {}\n".format(self.hidden_dim) +
              "\t noise dimensions: {}\n".format(self.noise_dim) +
              "\t number of epochs: {}\n".format(self.num_epoches) +
              "\t batch size: {}\n".format(self.batch_size) +
              "\t learning rate: {}\n".format(self.lr) +
              "\t standardization: {}\n".format(self.standardize) +
              "\t training mode: {}\n".format(self.models[0].model.training) +
              "\t device: {}\n".format(self.device) + 
              "\t ensemble size: {}\n".format(self.ensemble_size))
        print("Validation {} loss: {:.4f}".format(self.val_loss_type, self.val_loss_final))
        
    def train(self, x, y, num_epoches=None, batch_size=None, standardize=True, val_loss_type="energy", val_sample_size=100):
        """Fit multiple models on subsamples of the training data.

        Args:
            x (torch.Tensor): training data of predictors.
            y (torch.Tensor): trainging data of responses.
            num_epoches (int, optional): number of training epochs. Defaults to None.
            batch_size (int, optional): batch size for mini-batch SGD. Defaults to 512.
            print_every_nepoch (int, optional): print losses every print_every_nepoch number of epochs. Defaults to 100.
            print_times_per_epoch (int, optional): print losses for print_times_per_epoch times per epoch. Defaults to 1.
            standardize (bool, optional): whether to standardize the data. Defaults to True.
                - standardization scheme for bagging: 
                    for each model, all data are standardized using the mean and standard deviation of the training data for fitting this model alone.
            val_loss_type (str, optional): loss type for validation. Defaults to "energy". Choices: ["l1", "l2", "energy"].
            val_sample_size (int, optional): generated sample size for evaluating the validation loss. Defaults to 100.
        """
        self.train_mode()
        if num_epoches is None:
            num_epoches = self.num_epoches
        if batch_size is None:
            batch_size = self.batch_size if self.batch_size is not None else x.size(0)
        if val_loss_type == "":
            val_loss_type = self.val_loss_type
        if standardize:
            self.standardize = standardize

        x = vectorize(x)
        y = vectorize(y)
        x = x.to(self.device)
        y = y.to(self.device)
            
        # Mask matrix for splitting training and validation data.
        data_size = x.shape[0]
        rng = np.random.default_rng(21875667591346)
        self.split_mask = torch.from_numpy(rng.multivariate_hypergeometric([1]*self.ensemble_size, self.num_models_for_each_sample, size=data_size))
        
        # Training.
        for i in range(self.ensemble_size):
            verbose = i == 0
            model = self.models[i]
            train_idx = self.split_mask[:,i] == 1
            x_train = x[train_idx]
            y_train = y[train_idx]
            x_val = x[~train_idx]
            y_val = y[~train_idx]
            if verbose:
                print("Model 1 training details:\n")
            model.train(x_train, y_train, num_epoches=num_epoches, batch_size=batch_size, standardize=self.standardize, verbose=verbose)
            if verbose:
                print("\n")
            val_loss = model.eval_loss(x_val, y_val, loss_type="energy", sample_size=2)
            train_loss = model.eval_loss(x_train, y_train, loss_type="energy", sample_size=2)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print("[Model {}] train_loss: {:.4f}, val_loss: {:.4f}".format(i + 1, train_loss, val_loss))
        
        # Validation.
        self.val_loss_final = self.validate_bagged(x, y, loss_type=val_loss_type, sample_size=val_sample_size)
        print("\nValidation {} loss of the bagged engression model: {:.4f}".format(val_loss_type, self.val_loss_final))
    
    def validate_bagged(self, x, y, loss_type="energy", sample_size=100):
        """Validation loss of the bagged model (on training data set). The loss for each data point is computed using the models that are not fitted on this data point. Note that this is a better validation metric than simply averaging the validation losses of each model. 

        Args:
            x (torch.Tensor): training data for predictors.
            y (torch.Tensor): training data for responses.
            loss_type (str, optional): type of the loss. Defaults to "energy".
            sample_size (int, optional): generated sample size to evaluate the loss. Defaults to 100.

        Returns:
            float: validation loss of the bagged model.
        """
        self.eval_mode()
        x = vectorize(x)
        y = vectorize(y)
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Samples from the bagged model on the validation data.
        sample_size_per_model = sample_size // self.num_models_for_each_sample
        y_samples = torch.zeros((y.shape[0], y.shape[1], sample_size_per_model, self.ensemble_size), device=self.device)
        for i in range(self.ensemble_size):
            model = self.models[i]
            val_idx = self.split_mask[:,i] == 0
            y_samples[val_idx, :, :, i] = model.sample(x[val_idx], sample_size=sample_size_per_model)
        val_idx = (1 - self.split_mask).nonzero(as_tuple=True)
        y_samples = y_samples[val_idx[0], :, :, val_idx[1]]
        y_samples = y_samples.reshape(y.shape[0], self.num_models_for_each_sample, y.shape[1], sample_size_per_model).permute(0, 2, 3, 1)
        y_samples = y_samples.reshape(y.shape[0], y.shape[1], -1)
        
        # Compute the loss (on the original scale).
        if loss_type == "energy":
            y_samples = list(torch.split(y_samples, 1, dim=2))
            loss = energy_loss(y, y_samples, verbose=False)
        elif loss_type == "l2":
            y_pred = y_samples.mean(dim=len(y_samples.shape) - 1)
            loss = (y - y_pred).pow(2).mean()
        elif loss_type == "l1":
            y_pred = y_samples.median(dim=len(y_samples.shape) - 1).values
            loss = (y - y_pred).abs().mean()
        return loss.item()
    
    def sample(self, x, sample_size=100):
        """Sample new response data from all ensemble models. 

        Args:
            x (torch.Tensor): test data of predictors.
            sample_size (int, optional): generated sample sizes for each x. Defaults to 100.
            expand_dim (bool, optional): whether to expand the sample dimension. Defaults to True.

        Returns:
            torch.Tensor of shape (data_size, response_dim, sample_size).
                - [:,:,i] consists of the i-th sample of all x.
                - [i,:,:] consists of all samples of x_i.
        """
        self.eval_mode()
        x = vectorize(x)
        x = x.to(self.device)
        sample_size_per_model = sample_size // self.ensemble_size
        y_samples = []
        for i in range(self.ensemble_size):
            model = self.models[i]
            if i == self.ensemble_size:
                sample_size_per_model += sample_size % self.ensemble_size
            y_samples.append(model.sample(x, sample_size=sample_size_per_model))
        y_samples = torch.cat(y_samples, dim=-1)
        return y_samples
    
    def predict(self, x, target="mean", sample_size=100):
        """Point prediction.

        Args:
            x (torch.Tensor): data of predictors.
            target (str or float or list, optional): a quantity of interest to predict. float refers to the quantiles. Defaults to "mean".
            sample_size (int, optional): generated sample sizes for each x. Defaults to 100.

        Returns:
            torch.Tensor or list of torch.Tensor: point predictions.
        """
        self.eval_mode()
        x = vectorize(x)
        x = x.to(self.device)
        y_samples = self.sample(x, sample_size=sample_size)
        if not isinstance(target, list):
            target = [target]
        y_pred = []
        extremes = []
        for t in target:
            if t == "mean":
                y_pred.append(y_samples.mean(dim=len(y_samples.shape) - 1))
            else:
                if t == "median":
                    t = 0.5
                assert isinstance(t, float)
                y_pred.append(y_samples.quantile(t, dim=len(y_samples.shape) - 1))
                if min(t, 1 - t) * sample_size < 10:
                    extremes.append(t)
        
        if len(extremes) > 0:
            print("Warning: the estimate for quantiles at {} with a sample size of {} could be inaccurate. Please increase the `sample_size`.".format(extremes, sample_size))

        if len(y_pred) == 1:
            return y_pred[0]
        else:
            return y_pred
        
    def eval_loss(self, x, y, loss_type="l2", sample_size=100):
        """Evaluate the bagged model.

        Args:
            x (torch.Tensor): data of predictors.
            y (torch.Tensor): data of responses.
            loss_type (str, optional): loss type. Defaults to "l2". Choices: ["l2", "l1", "energy", "cor"].
            sample_size (int, optional): generated sample sizes for each x. Defaults to 100.

        Returns:
            float: evaluation loss.
        """
        self.eval_mode()
        x = vectorize(x)
        y = vectorize(y)
        x = x.to(self.device)
        y = y.to(self.device)
        if loss_type == "l2":
            y_pred = self.predict(x, target="mean", sample_size=sample_size)
            loss = (y - y_pred).pow(2).mean()
        elif loss_type == "cor":
            y_pred = self.predict(x, target="mean", sample_size=sample_size)
            loss = cor(y, y_pred)
        elif loss_type == "l1":
            y_pred = self.predict(x, target=0.5, sample_size=sample_size)
            loss = (y - y_pred).abs().mean()
        else:
            assert loss_type == "energy"
            y_samples = self.sample(x, sample_size=sample_size, expand_dim=False)
            loss = energy_loss(y, y_samples, verbose=False)
        return loss.item()
    
    def plot(self, x, y, x_idx=0, y_idx=0, target="mean", sample_size=100, save_dir=None,
             alpha=0.8, ymin=None, ymax=None):
        """Plot true data and predictions.

        Args:
            x (torch.Tensor): data of predictors
            y (torch.Tensor): data of responses
            x_idx (int, optional): index of the predictor to plot (if there are multiple). Defaults to 0.
            y_idx (int, optional): index of the response to plot (if there are multiple). Defaults to 0.
            target (str or float, optional): target quantity. Defaults to "mean". Choice: ["mean", "median", "sample", float].
            sample_size (int, optional): generated sample size used for estimation. Defaults to 100.
            save_dir (str, optional): directory to save the plot. Defaults to None.
            alpha (float, optional): transparency of the sampled data points. Defaults to 0.8.
            ymin (float, optional): minimum value of y in the plot. Defaults to None.
            ymax (float, optional): maximum value of y in the plot. Defaults to None.
        """
        x = vectorize(x)
        y = vectorize(y)
        x = x.to(self.device)
        y = y.to(self.device)
        plt.scatter(x[:,x_idx].cpu(), y[:,y_idx].cpu(), s=1, label="true data", color="silver")
        if target != "sample":
            y_pred = self.predict(x, target=target, sample_size=sample_size)
            plt.scatter(x[:,x_idx].cpu(), y_pred[:,y_idx].cpu(), s=1, label="predictions", color="lightskyblue")
        else:
            y_samples = self.sample(x, sample_size=sample_size, expand_dim=False)
            x_rep = x.repeat(sample_size, 1)
            plt.scatter(x_rep[:,x_idx].cpu(), y_samples[:,y_idx].cpu(), s=1, label="samples", color="lightskyblue", alpha=alpha)
        plt.legend(markerscale=2)
        plt.ylim(ymin, ymax)
        if x.shape[1] == 1:
            plt.xlabel(r"$x$")
        else:
            plt.xlabel(r"$x_{}$".format(x_idx))
        if y.shape[1] == 1:
            plt.ylabel(r"$y$")
        else:
            plt.ylabel(r"$y_{}$".format(y_idx))
        if save_dir is not None:
            make_folder(save_dir)
            plt.savefig(save_dir, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
