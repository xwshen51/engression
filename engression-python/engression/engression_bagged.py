# coming soon.

import torch
import numpy as np
import matplotlib.pyplot as plt

from .engression import Engressor
from .loss_func import *
from .models import StoNet
from .data.loader import make_dataloader
from .utils import *


class BaggedEngressor(object):
    def __init__(self, 
                 in_dim, out_dim, num_layer=2, hidden_dim=100, noise_dim=100,
                 lr=0.001, num_epoches=500, batch_size=None, device="cpu", standardize=True,
                 ensemble_size=10, val_loss_type="energy"): 
        """Engressor with bagging.

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
        super().__init__()
        self.lr = lr
        self.num_epoches = num_epoches
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.num_models_for_each_sample = ensemble_size // 2
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.standardize = standardize
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.split_mask = None
        
        # Build ensemble models.
        self.models = []        
        for _ in range(ensemble_size):
            self.models.append(Engressor(in_dim, out_dim, num_layer, hidden_dim, noise_dim, 
                                         lr=lr, device=device, standardize=standardize))
        self.val_loss_type = val_loss_type

    def train_mode(self):
        for i in range(self.ensemble_size):
            self.models[i].train_mode()
        
    def eval_mode(self):
        for i in range(self.ensemble_size):
            self.models[i].eval_mode()
        
    def train(self, x, y, num_epoches=None, batch_size=None, standardize=True, val_loss_type="", val_sample_size=100, verbose=True):
        """_summary_

        Args:
            x (_type_): _description_
            y (_type_): _description_
            num_epoches (_type_, optional): _description_. Defaults to None.
            batch_size (_type_, optional): _description_. Defaults to None.
            standardize (bool, optional): _description_. Defaults to True.
                - Standardization scheme for bagging: 
                    for each model, all data are standardized using the mean and standard deviation of the training data for this model alone.
            val_loss_type (str, optional): _description_. Defaults to "".
            val_sample_size (int, optional): _description_. Defaults to 100.
            verbose (bool, optional): _description_. Defaults to True.
        """
        self.train_mode()
        if num_epoches is None:
            num_epoches = self.num_epoches
        if batch_size is None:
            batch_size = self.batch_size
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
            model = self.models[i]
            train_idx = self.split_mask[:,i]
            x_train = x[train_idx]
            y_train = y[train_idx]
            x_val = x[~train_idx]
            y_val = y[~train_idx]
            model.train(x_train, y_train, num_epoches=num_epoches, batch_size=batch_size, standardize=self.standardize, verbose=False)
            val_loss = model.eval_loss(x_val, y_val, loss_type="energy", sample_size=2)
            train_loss = model.eval_loss(x_train, y_train, loss_type="energy", sample_size=2)
            standardize_str = "(standardized)" if self.standardize else ""
            print("[Model {}] train_loss{}: {:.4f}, val_loss{}: {:.4f}".format(i + 1, standardize_str, train_loss, standardize_str, val_loss))
        
        # Validation.
        val_loss_final = self.validate_bagged(x, y, loss_type=val_loss_type, sample_size=val_sample_size)
        print("Final validation {} loss: {:.4f}".format(val_loss_type, val_loss_final))
    
    def validate_bagged(self, x, y, loss_type="energy", sample_size=100):
        """Evaluate the bagged model on the validation data.

        Args:
            x (torch.Tensor): training data for predictors.
            y (torch.Tensor): training data for responses.
            loss_type (str, optional): type of the loss. Defaults to "energy".
            sample_size (int, optional): _description_. Defaults to 100.

        Returns:
            float: loss of the bagged model on the validation data.
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
        
        # Compute the loss (on the original scale).
        if loss_type == "energy":
            loss = energy_loss(y, y_samples, verbose=False)
        elif loss_type == "l2":
            y_pred = y_samples.mean(dim=len(y_samples.shape) - 1)
            loss = (y - y_pred).pow(2).mean()
        elif loss_type == "l1":
            y_pred = y_samples.median(dim=len(y_samples.shape) - 1)
            loss = (y - y_pred).abs().mean()
        return loss.item()
    
    def sample(self, x, sample_size=100):
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
        
    def eval_loss(self, x, y, loss_type="energy", sample_size=100):
        """Evaluate the bagged model.

        Args:
            x (_type_): _description_
            y (_type_): _description_
            loss_type (str, optional): _description_. Defaults to "energy".
            sample_size (int, optional): _description_. Defaults to 100.

        Returns:
            _type_: _description_
        """
        self.eval_mode()
        x = vectorize(x)
        y = vectorize(y)
        x = x.to(self.device)
        y = y.to(self.device)
        if loss_type == "l2":
            y_pred = self.predict(x, target="mean", sample_size=sample_size)
            loss = (y - y_pred).pow(2).mean()
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
            sample_size (int, optional): sample size used for estimation. Defaults to 100.
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


def engression_bagged(x, y, 
                       num_layer=2, hidden_dim=100, noise_dim=100,
                       lr=0.001, num_epoches=500, batch_size=None, device="cpu",
                       nfolds=10, loss_type="energy"):
    engressor = BaggedEngressor(in_dim=x.shape[1], out_dim=y.shape[1], num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim, 
                            lr=lr, num_epoches=num_epoches, batch_size=batch_size, device=device, nfolds=nfolds)
    engressor.train(x, y)
    print("Average validation {} loss: {:.4f}".format(loss_type, np.mean(engressor.val_losses)))
    return engressor
