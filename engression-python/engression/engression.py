import torch
import matplotlib.pyplot as plt

from .loss_func import *
from .models import StoNet
from .data.loader import make_dataloader
from .utils import *


class Engressor(object):

    def __init__(self, 
                 in_dim, out_dim, num_layer=2, hidden_dim=100, noise_dim=100,
                 lr=0.001, num_epoches=500, batch_size=None, device="cpu", standardize=True): 
        """Engressor class

        Args:
            in_dim (int): input dimension
            out_dim (int): output dimension
            num_layer (int, optional): number of layers. Defaults to 2.
            hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
            noise_dim (int, optional): noise dimension. Defaults to 100.
            lr (float, optional): learning rate. Defaults to 0.001.
            num_epoches (int, optional): number of epoches. Defaults to 500.
            batch_size (int, optional): batch size. Defaults to None, referring to the full batch.
            device (str or torch.device, optional): device. Defaults to "cpu". Choices = ["cpu", "gpu", "cuda"].
            standardize (bool, optional): whether to standardize data for training. Defaults to True.
        """
        super().__init__()
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.lr = lr
        self.num_epoches = num_epoches
        self.batch_size = batch_size
        if isinstance(device, str):
            if device == "gpu" or device == "cuda":
                device = torch.device("cuda")
            else:
                device = torch.device(device)
        self.device = device
        check_for_gpu(self.device)
        self.standardize = standardize
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        
        self.model = StoNet(in_dim, out_dim, num_layer, hidden_dim, noise_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.tr_loss = None
            
    def train_mode(self):
        self.model.train()
        
    def eval_mode(self):
        self.model.eval()
        
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
              "\t training mode: {}\n".format(self.model.training) +
              "\t device: {}\n".format(self.device))
        print("Training loss (original scale):\n" +
              "\t energy-loss: {:.2f}, \n\tE(|Y-Yhat|): {:.2f}, \n\tE(|Yhat-Yhat'|): {:.2f}".format(
                  self.tr_loss[0], self.tr_loss[1], self.tr_loss[2]))

        
    def _standardize_data_and_record_stats(self, x, y):
        """Standardize the data and record the mean and standard deviation of the training data.

        Args:
            x (torch.Tensor): training data of predictors.
            y (torch.Tensor): training data of responses.

        Returns:
            torch.Tensor: standardized data.
        """
        self.x_mean = torch.mean(x, dim=0)
        self.x_std = torch.std(x, dim=0)
        self.y_mean = torch.mean(y, dim=0)
        self.y_std = torch.std(y, dim=0)
        return (x - self.x_mean) / self.x_std, (y - self.y_mean) / self.y_std

    def standardize_data(self, x, y=None):
        """Standardize the data, if self.standardize is True.

        Args:
            x (torch.Tensor): training data of predictors.
            y (torch.Tensor, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: standardized or original data.
        """
        if y is None:
            if self.standardize:
                return (x - self.x_mean) / self.x_std
            else:
                return x
        else:
            if self.standardize:
                return (x - self.x_mean) / self.x_std, (y - self.y_mean) / self.y_std
            else:
                return x, y
    
    def unstandardize_data(self, y, x=None):
        """Transform the predictions back to the original scale, if self.standardize is True.

        Args:
            y (torch.Tensor): data in the standardized scale

        Returns:
            torch.Tensor: data in the original scale
        """
        if x is None:
            if self.standardize:
                return y * self.y_std + self.y_mean
            else:
                return y
        else:
            if self.standardize:
                return x * self.x_std + self.x_mean, y * self.y_std + self.y_mean
            else:
                return x, y
        
    def train(self, x, y, num_epoches=None, batch_size=512, print_every_nepoch=100, print_times_per_epoch=1, standardize=True, verbose=True):
        """Training function.

        Args:
            x (torch.Tensor): training data of predictors.
            y (torch.Tensor): trainging data of responses.
            num_epoches (int, optional): number of training epochs. Defaults to None.
            batch_size (int, optional): batch size for mini-batch SGD. Defaults to 512.
            print_every_nepoch (int, optional): print losses every print_every_nepoch number of epochs. Defaults to 100.
            print_times_per_epoch (int, optional): print losses for print_times_per_epoch times per epoch. Defaults to 1.
            standardize (bool, optional): standardize the data. Defaults to True.
            verbose (bool, optional): whether to print losses and info. Defaults to True.
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
        if self.standardize and verbose: 
            print("Data is standardized for training only; the printed training losses are on the standardized scale. \n" +
                  "However during evaluation, the predictions, evaluation metrics, and plots will be on the original scale.\n")
            x, y = self._standardize_data_and_record_stats(x, y)
        
        if batch_size >= x.size(0)//2:
            if verbose:
                print("Batch is larger than half of the sample size. Training based on full-batch gradient descent.")
            self.batch_size = x.size(0)
            for epoch_idx in range(num_epoches):
                self.model.zero_grad()
                y_sample1 = self.model(x)
                y_sample2 = self.model(x)
                loss, loss1, loss2 = energy_loss_two_sample(y, y_sample1, y_sample2, verbose=True)
                loss.backward()
                self.optimizer.step()
                if (epoch_idx == 0 or  (epoch_idx + 1) % print_every_nepoch == 0) and verbose:
                    print("[Epoch {} ({:.0f}%)] energy-loss: {:.4f},  E(|Y-Yhat|): {:.4f},  E(|Yhat-Yhat'|): {:.4f}".format(
                        epoch_idx + 1, 100 * epoch_idx / num_epoches, loss.item(), loss1.item(), loss2.item()))
        else:
            train_loader = make_dataloader(x, y, batch_size=batch_size, shuffle=True)
            if verbose:
                print("Training based on mini-batch gradient descent with a batch size of {}.".format(batch_size))
            for epoch_idx in range(num_epoches):
                for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                    self.model.zero_grad()
                    y_sample1 = self.model(x_batch)
                    y_sample2 = self.model(x_batch)
                    loss, loss1, loss2 = energy_loss_two_sample(y_batch, y_sample1, y_sample2, verbose=True)
                    loss.backward()
                    self.optimizer.step()
                    if (epoch_idx == 0 or (epoch_idx + 1) % print_every_nepoch == 0) and verbose:
                        if (batch_idx + 1) % (len(train_loader) // print_times_per_epoch) == 0:
                            print("[Epoch {} ({:.0f}%), batch {}]: energy-loss: {:.4f},  E(|Y-Yhat|): {:.4f},  E(|Yhat-Yhat'|): {:.4f}".format(
                                epoch_idx + 1, 100 * epoch_idx / num_epoches, batch_idx + 1, loss.item(), loss1.item(), loss2.item()))

        # Evaluate performance on the training data (on the original scale)
        self.model.eval()
        x, y = self.unstandardize_data(y, x)
        self.tr_loss = self.eval_loss(x, y, loss_type="energy", verbose=True)
        
        if verbose:
            print("\nTraining loss on the original (non-standardized) scale:\n" +
                "\tEnergy-loss: {:.4f},  E(|Y-Yhat|): {:.4f},  E(|Yhat-Yhat'|): {:.4f}".format(
                    self.tr_loss[0], self.tr_loss[1], self.tr_loss[2]))
            
        if verbose:
            print("\nPrediction-loss E(|Y-Yhat|) and variance-loss E(|Yhat-Yhat'|) should ideally be equally large" +
                "\n-- consider training for more epochs or adjusting hyperparameters if there is a mismatch ")
    
    def predict(self, x, target="mean", sample_size=100):
        """Point prediction.

        Args:
            x (torch.Tensor): data of predictors.
            target (str or float or list, optional): single-valued functional to predict. float refers to the quantiles. Defaults to ["mean"].
            sample_size (int, optional): sample sizes for each x. Defaults to 100.

        Returns:
            torch.Tensor or list of torch.Tensor: point predictions.
        """
        self.eval_mode()  
        x = vectorize(x)
        x = x.to(self.device)
        x = self.standardize_data(x)
        y_pred = self.model.predict(x, target, sample_size)
        y_pred = self.unstandardize_data(y_pred)
        return y_pred
    
    def sample(self, x, sample_size=100, expand_dim=True):
        """Sample new response data.

        Args:
            x (torch.Tensor): test data of predictors.
            target (str or float or list, optional): single-valued functional to predict. float refers to the quantiles. Defaults to ["mean"].
            sample_size (int, optional): sample sizes for each x. Defaults to 100.

        Returns:
            torch.Tensor or list of torch.Tensor: samples.
                - [:,:,i] gives the i-th sample of all x.
                - [i,:,:] gives all samples of x_i.
        """
        self.eval_mode()
        x = vectorize(x)
        x = x.to(self.device)
        x = self.standardize_data(x)
        y_samples = self.model.sample(x, sample_size, expand_dim=expand_dim)
        y_samples = self.unstandardize_data(y_samples)
        return y_samples
    
    def eval_loss(self, x, y, loss_type="l2", sample_size=None, verbose=False):
        """Compute the loss for evaluation.

        Args:
            x (torch.Tensor): data of predictors.
            y (torch.Tensor): data of responses.
            loss_type (str, optional): loss type. Defaults to "l2". Choices: ["l2", "l1", "energy", "cor"].
            sample_size (int, optional): sample sizes for each x. Defaults to 100.
        
        Returns:
            float: evaluation loss.
        """
        if sample_size is None:
            sample_size = 2 if loss_type == "energy" else 100
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
            loss = energy_loss(y, y_samples, verbose=verbose)
        if not verbose:
            return loss.item()
        else:
            loss, loss1, loss2 = loss
            return loss.item(), loss1.item(), loss2.item()
    
    def plot(self, x_te, y_te, x_tr=None, y_tr=None, x_idx=0, y_idx=0, 
             target="mean", sample_size=100, save_dir=None,
             alpha=0.8, ymin=None, ymax=None):
        """Plot true data and predictions.

        Args:
            x_te (torch.Tensor): test data of predictors
            y_te (torch.Tensor): test data of responses
            x_tr (torch.Tensor): training data of predictors
            y_tr (torch.Tensor): training data of responses
            x_idx (int, optional): index of the predictor to plot (if there are multiple). Defaults to 0.
            y_idx (int, optional): index of the response to plot (if there are multiple). Defaults to 0.
            target (str or float, optional): target quantity. Defaults to "mean". Choice: ["mean", "median", "sample", float].
            sample_size (int, optional): sample sizes for each x. Defaults to 100.
            save_dir (str, optional): directory to save the plot. Defaults to None.
            alpha (float, optional): transparency of the sampled data points. Defaults to 0.8.
            ymin (float, optional): minimum value of y in the plot. Defaults to None.
            ymax (float, optional): maximum value of y in the plot. Defaults to None.
        """
        if x_tr is not None and y_tr is not None:
            # Plot training data as well.
            x_tr = vectorize(x_tr)
            y_tr = vectorize(y_tr)
            plt.scatter(x_tr[:,x_idx].cpu(), y_tr[:,y_idx].cpu(), s=1, label="training data", color="silver")
            plt.scatter(x_te[:,x_idx].cpu(), y_te[:,y_idx].cpu(), s=1, label="test data", color="gold")
            x = torch.cat((x_tr, x_te), dim=0)
            y = torch.cat((y_tr, y_te), dim=0)
        else:
            # Plot only the test data.
            x_te = vectorize(x_te)
            y_te = vectorize(y_te)
            plt.scatter(x_te[:,x_idx].cpu(), y_te[:,y_idx].cpu(), s=1, label="true data", color="silver")
            x = x_te
            y = y_te
        x = x.to(self.device)
        y = y.to(self.device)
        
        if target != "sample":
            y_pred = self.predict(x, target=target, sample_size=sample_size)
            plt.scatter(x[:,x_idx].cpu(), y_pred[:,y_idx].cpu(), s=1, label="predictions", color="lightskyblue")
        else:
            y_sample = self.sample(x, sample_size=sample_size, expand_dim=False)
            x_rep = x.repeat(sample_size, 1)
            plt.scatter(x_rep[:,x_idx].cpu(), y_sample[:,y_idx].cpu(), s=1, label="samples", color="lightskyblue", alpha=alpha)
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


def engression(x, y, 
               num_layer=2, hidden_dim=100, noise_dim=100,
               lr=0.001, num_epoches=500, batch_size=None, 
               print_every_nepoch=100, print_times_per_epoch=1,
               device="cpu", standardize=True,
               verbose=True): 
    """engression function.

    Args:
        x (torch.Tensor): training data of predictors.
        y (torch.Tensor): training data of responses.
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        lr (float, optional): learning rate. Defaults to 0.001.
        num_epoches (int, optional): number of epochs. Defaults to 500.
        batch_size (int, optional): batch size. Defaults to None.
        print_every_nepoch (int, optional): print losses every print_every_nepoch number of epochs. Defaults to 100.
        print_times_per_epoch (int, optional): print losses for print_times_per_epoch times per epoch. Defaults to 1.
        device (str, torch.device, optional): device. Defaults to "cpu". Choices = ["cpu", "gpu", "cuda"].
        standardize (bool, optional):  whether to standardize data for training. Defaults to True.
        verbose (bool, optional): whether to print losses and info. Defaults to True.

    Returns:
        Engressor object: a fitted engression model.
    """
    engressor = Engressor(in_dim=x.shape[1], out_dim=y.shape[1], num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim, 
                          lr=lr, num_epoches=num_epoches, batch_size=batch_size, device=device, standardize=standardize)
    engressor.train(x, y, num_epoches=num_epoches, batch_size=batch_size, 
                    print_every_nepoch=print_every_nepoch, print_times_per_epoch=print_times_per_epoch, 
                    standardize=standardize, verbose=verbose)
    return engressor
