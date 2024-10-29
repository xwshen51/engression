import torch
from .utils import vectorize
from torch.linalg import vector_norm


def energy_loss(x_true, x_est, beta=1, verbose=True):
    """Loss function based on the energy score.

    Args:
        x_true (torch.Tensor): iid samples from the true distribution of shape (data_size, data_dim)
        x_est (list of torch.Tensor): 
            - a list of length sample_size, where each element is a tensor of shape (data_size, data_dim) that contains one sample for each data point from the estimated distribution, or 
            - a tensor of shape (data_size*sample_size, response_dim) such that x_est[data_size*(i-1):data_size*i,:] contains one sample for each data point, for i = 1, ..., sample_size.
        beta (float): power parameter in the energy score.
        verbose (bool): whether to return two terms of the loss.

    Returns:
        loss (torch.Tensor): energy loss.
    """
    EPS = 0 if float(beta).is_integer() else 1e-5
    x_true = vectorize(x_true).unsqueeze(1)
    if not isinstance(x_est, list):
        x_est = list(torch.split(x_est, x_true.shape[0], dim=0))
    m = len(x_est)
    x_est = [vectorize(x_est[i]).unsqueeze(1) for i in range(m)]
    x_est = torch.cat(x_est, dim=1)
        
    s1 = (vector_norm(x_est - x_true, 2, dim=2) + EPS).pow(beta).mean()
    s2 = (torch.cdist(x_est, x_est, 2) + EPS).pow(beta).mean() * m / (m - 1)
    if verbose:
        return torch.cat([(s1 - s2 / 2).reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return (s1 - s2 / 2)
    

def energy_loss_two_sample(x0, x, xp, x0p=None, beta=1, verbose=True, weights=None):
    """Loss function based on the energy score (estimated based on two samples).
    
    Args:
        x0 (torch.Tensor): an iid sample from the true distribution.
        x (torch.Tensor): an iid sample from the estimated distribution.
        xp (torch.Tensor): another iid sample from the estimated distribution.
        xp0 (torch.Tensor): another iid sample from the true distribution.
        beta (float): power parameter in the energy score.
        verbose (bool):  whether to return two terms of the loss.
    
    Returns:
        loss (torch.Tensor): energy loss.
    """
    EPS = 0 if float(beta).is_integer() else 1e-5
    x0 = vectorize(x0)
    x = vectorize(x)
    xp = vectorize(xp)
    if weights is None:
        weights = 1 / x0.size(0)
    if x0p is None:
        s1 = ((vector_norm(x - x0, 2, dim=1) + EPS).pow(beta) * weights).sum() / 2 + ((vector_norm(xp - x0, 2, dim=1) + EPS).pow(beta) * weights).sum() / 2
        s2 = ((vector_norm(x - xp, 2, dim=1) + EPS).pow(beta) * weights).sum() 
        loss = s1 - s2/2
    else:
        x0p = vectorize(x0p)
        s1 = ((vector_norm(x - x0, 2, dim=1) + EPS).pow(beta).sum() + (vector_norm(xp - x0, 2, dim=1) + EPS).pow(beta).sum() + 
              (vector_norm(x - x0p, 2, dim=1) + EPS).pow(beta).sum() + (vector_norm(xp - x0p, 2, dim=1) + EPS).pow(beta).sum()) / 4
        s2 = (vector_norm(x - xp, 2, dim=1) + EPS).pow(beta).sum() 
        s3 = (vector_norm(x0 - x0p, 2, dim=1) + EPS).pow(beta).sum() 
        loss = s1 - s2/2 - s3/2
    if verbose:
        return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return loss
