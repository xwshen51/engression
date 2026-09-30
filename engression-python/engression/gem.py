import torch

from .links import link, is_continuous
from .loss_func import energy_loss_two_sample
from .models import StoNetBase, StoNet, Net


class GEMNet(StoNetBase):
    """Generalized engression model (GEM): Y = h(g(X, eps) + sigma(X) * eta), where g is a stochastic neural network,
    sigma is a neural network with positive outputs, eta is standard Gaussian noise, and h is the link of the data type of the response.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        blocks (list of Block): blocks of the response with their data types, as returned by `links.parse_data_type`.
        num_layer (int, optional): number of layers of g. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer of g. Defaults to 100.
        noise_dim (int, optional): noise dimension of g. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer to g. Defaults to False.
        resblock (bool, optional): whether to use residual blocks in g. Defaults to False.
        sigma_dim (str, optional): "scalar" for a scale sigma common to all coordinates of the response or "vector" for coordinate-wise scales. Defaults to "scalar".
        sigma_min (float, optional): lower bound of sigma on the discrete coordinates of the response. Defaults to 0.2.
        sigma_num_layer (int, optional): number of layers of sigma. Defaults to 2.
        sigma_hidden_dim (int, optional): number of neurons per layer of sigma. Defaults to 100.
    """
    def __init__(self, in_dim, out_dim, blocks, num_layer=2, hidden_dim=100,
                 noise_dim=100, add_bn=False, resblock=False,
                 sigma_dim="scalar", sigma_min=0.2, sigma_num_layer=2, sigma_hidden_dim=100):
        super().__init__()
        if sigma_dim not in ["scalar", "vector"]:
            raise ValueError("Unknown `sigma_dim` '{}'. Choices: ['scalar', 'vector'].".format(sigma_dim))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.blocks = blocks
        self.sigma_dim = sigma_dim
        self.g_net = StoNet(in_dim, out_dim, num_layer, hidden_dim, noise_dim, add_bn, None, resblock)
        self.sigma_net = Net(in_dim, 1 if sigma_dim == "scalar" else out_dim, sigma_num_layer, sigma_hidden_dim, out_act="softplus")
        # The gradient estimate for a discrete coordinate is divided by sigma, so sigma is bounded away from zero there.
        self.register_buffer("sigma_min", sigma_min * (~is_continuous(blocks)).float())
        # The levels of an ordinal response are 1, ..., L, so g starts from the middle level instead of zero.
        g_init = torch.zeros(out_dim)
        for block in blocks:
            if block.name == "ordinal":
                g_init[block.start:block.end] = (block.num_level + 1) / 2
        self.register_buffer("g_init", g_init)

    def g(self, x):
        """Sample from the generator g, which gives the latent vector before the perturbation and the link.

        Args:
            x (torch.Tensor): input data of shape (data_size, in_dim).

        Returns:
            torch.Tensor: samples of shape (data_size, out_dim).
        """
        return self.g_net(x) + self.g_init

    def sigma(self, x):
        """Scale of the perturbation eta.

        Args:
            x (torch.Tensor): input data of shape (data_size, in_dim).

        Returns:
            torch.Tensor: scales of shape (data_size, out_dim).
        """
        return self.sigma_net(x) + self.sigma_min

    def forward(self, x, return_latent=False):
        """Sample from the model.

        Args:
            x (torch.Tensor): input data of shape (data_size, in_dim).
            return_latent (bool, optional): whether to return also the quantities that the sample is computed from. Defaults to False.

        Returns:
            torch.Tensor: samples of shape (data_size, out_dim), one for each data point.
            If return_latent, a tuple of the samples, the outputs of g, the scales sigma, and the noise eta, all of shape (data_size, out_dim).
        """
        g = self.g(x)
        sigma = self.sigma(x)
        eta = torch.randn_like(g)
        y = link(g + sigma * eta, self.blocks)
        if return_latent:
            return y, g, sigma, eta
        else:
            return y


def _distance(x, xp, blocks, beta=1, control_variate=False):
    """Distance ||x - xp||^beta between the rows of two samples, as used in the gradient estimate of `gem_loss_two_sample`.

    Returns:
        torch.Tensor of shape (data_size, 1), or of shape (data_size, response_dim) with control variate,
        in which case the columns of the binary labels are centered: when a binary coordinate of x flips,
        the distance takes two values, whose midpoint, which does not depend on that coordinate, is subtracted.
    """
    EPS = 0 if float(beta).is_integer() else 1e-5
    diff2 = (x - xp).pow(2)
    dist2 = diff2.sum(dim=1, keepdim=True)
    dist = (dist2.sqrt() + EPS).pow(beta)
    if control_variate:
        dist = dist.repeat(1, x.size(1))
        for block in blocks:
            if block.name == "multilabel":
                # squared distance without coordinate j; with labels in {-1, +1}, coordinate j adds 0 or 4 to it
                rest = (dist2 - diff2[:, block.start:block.end]).clamp(min=0)
                dist[:, block.start:block.end] -= ((rest.sqrt() + EPS).pow(beta) + ((rest + 4).sqrt() + EPS).pow(beta)) / 2
    return dist


def gem_loss_two_sample(y, draw1, draw2, blocks, beta=1, control_variate=False):
    """Loss function of generalized engression based on the energy score (estimated based on two samples).

    The links of the discrete data types are not differentiable, so the gradient is obtained as follows.
        - For the continuous coordinates of the response, the loss is differentiated through the samples, as in engression.
        - For the discrete coordinates, the gradients of the loss with respect to the outputs of g and sigma are estimated
          from the values of the loss and the noise eta: with c = ||y - x||^beta - ||x - xp||^beta the cost of the sample x,
          these are c * eta / sigma / 2 and c * (eta^2 - 1) / sigma / 2, averaged over the data points.
    The estimates for the discrete coordinates enter the returned surrogate loss as coefficients of the outputs of g and sigma,
    so that one call of `backward` on the surrogate loss gives an unbiased estimate of the gradient of the energy loss
    with respect to all the parameters of the model.

    Args:
        y (torch.Tensor): an iid sample from the true distribution, with binary labels coded as -1/+1.
        draw1 (tuple of torch.Tensor): an iid sample from the estimated distribution, together with the outputs of g and sigma and the noise eta, as returned by `GEMNet(x, return_latent=True)`.
        draw2 (tuple of torch.Tensor): another iid sample from the estimated distribution, in the same form.
        blocks (list of Block): blocks of the response with their data types.
        beta (float): power parameter in the energy score.
        control_variate (bool): whether to use the control variate for binary labels, which reduces the variance of the gradient estimate.

    Returns:
        surrogate (torch.Tensor): surrogate loss, only to be differentiated.
        loss (torch.Tensor): energy loss and its two terms.
    """
    x, g, sigma, eta = draw1
    xp, gp, sigmap, etap = draw2
    loss = energy_loss_two_sample(y, x, xp, beta=beta, verbose=True)
    with torch.no_grad():
        discrete = ~is_continuous(blocks, device=x.device)
        zero = torch.zeros_like(x)
        dist = _distance(x, xp, blocks, beta, control_variate)
        cost = (_distance(x, y, blocks, beta, control_variate) - dist) / (2 * x.size(0))
        costp = (_distance(xp, y, blocks, beta, control_variate) - dist) / (2 * x.size(0))
        grad_g = torch.where(discrete, cost * eta / sigma, zero)
        grad_gp = torch.where(discrete, costp * etap / sigmap, zero)
        grad_sigma = torch.where(discrete, cost * (eta.pow(2) - 1) / sigma, zero)
        grad_sigmap = torch.where(discrete, costp * (etap.pow(2) - 1) / sigmap, zero)
    surrogate = (loss[0] + (grad_g * g).sum() + (grad_gp * gp).sum() +
                 (grad_sigma * sigma).sum() + (grad_sigmap * sigmap).sum())
    return surrogate, loss.detach()
