import torch
from .utils import vectorize

def energy_loss(x_true, x_est, beta=1, verbose=False):
    """Loss function based on the energy score.

    Args:
        x_true (torch.Tensor): iid samples from the true distribution.
        x_est (list of torch.Tensor): a list of iid samples from the estimated distribution. ## todo: not a list!!!
        beta (float): power parameter in the energy score.
        verbose (bool): whether to return two terms of the loss.

    Returns:
        loss (torch.Tensor): energy loss.
    """
    x_true = vectorize(x_true).unsqueeze(1)
    if not isinstance(x_est, list):
        x_est = list(torch.split(x_est, x_true.shape[0], dim=0))
    m = len(x_est)
    x_est = [vectorize(x_est[i]).unsqueeze(1) for i in range(m)]
    x_est = torch.cat(x_est, dim=1)
        
    s1 = torch.norm(x_est - x_true, 2, dim=2).pow(beta).mean()
    s2 = torch.cdist(x_est, x_est, 2).pow(beta).mean() * m / (m - 1)
    if verbose:
        return torch.cat([(s1 - s2 / 2).reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return (s1 - s2 / 2)
    

def energy_loss_two_sample(x0, x, xp, beta=1, verbose=False):
    """Loss function based on the energy score (estimated based on two samples).
    
    Args:
        x0 (torch.Tensor): iid samples from the true distribution.
        x (torch.Tensor): iid samples from the estimated distribution.
        xp (torch.Tensor): iid samples from the estimated distribution.
        beta (float): power parameter in the energy score.
        verbose (bool):  whether to return two terms of the loss.
    
    Returns:
        loss (torch.Tensor): energy loss.
    """
    x0 = vectorize(x0)
    x = vectorize(x)
    xp = vectorize(xp)
    s1 = torch.norm(x - x0, 2, dim=1).pow(beta).mean() / 2 + torch.norm(xp - x0, 2, dim=1).pow(beta).mean() / 2
    s2 = torch.norm(x - xp, 2, dim=1).pow(beta).mean() 
    if verbose:
        return torch.cat([(s1 - s2/2).reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return (s1 - s2/2)
