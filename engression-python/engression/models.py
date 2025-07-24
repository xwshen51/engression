import torch
import torch.nn as nn
from .data.loader import make_dataloader


class StoLayer(nn.Module):    
    """A stochastic layer.

    Args:
        in_dim (int): input dimension 
        out_dim (int): output dimension
        noise_dim (int, optional): noise dimension. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
    """
    def __init__(self, in_dim, out_dim, noise_dim=100, add_bn=False, out_act=None, noise_std=1, verbose=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.noise_dim = noise_dim
        self.add_bn = add_bn
        self.noise_std = noise_std
        self.verbose = verbose
        
        layer = [nn.Linear(in_dim + noise_dim, out_dim)]
        if add_bn:
            layer += [nn.BatchNorm1d(out_dim)]
        self.layer = nn.Sequential(*layer)
        if out_act == "softmax" and out_dim == 1:
            out_act = "sigmoid"
        self.out_act = get_act_func(out_act)
    
    def forward(self, x):
        device = next(self.layer.parameters()).device
        if isinstance(x, int):
            # For unconditional generation, x is the batch size.
            assert self.in_dim == 0
            out = torch.randn(x, self.noise_dim, device=device) * self.noise_std
        else:
            if x.size(1) < self.in_dim and self.verbose:
                print("Warning: covariate dimension does not aligned with the specified input dimension; filling in the remaining dimension with noise.")
            eps = torch.randn(x.size(0), self.noise_dim + self.in_dim - x.size(1), device=device) * self.noise_std
            out = torch.cat([x, eps], dim=1)
        out = self.layer(out)
        if self.out_act is not None:
            out = self.out_act(out)
        return out


def get_act_func(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "sigmoid":
        return nn.Sigmoid() 
    elif name == "tanh":
        return nn.Tanh() 
    elif name == "softmax":
        return nn.Softmax(dim=1)
    elif name == "elu":
        return nn.ELU(inplace=True)
    elif name == "softplus":
        return nn.Softplus()
    else:
        return None


class StoResBlock(nn.Module):
    """A stochastic residual net block.

    Args:
        dim (int, optional): input dimension. Defaults to 100.
        hidden_dim (int, optional): hidden dimension (default to dim). Defaults to None.
        out_dim (int, optional): output dimension (default to dim). Defaults to None.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        add_bn (bool, optional): whether to add batch normalization. Defaults to True.
        out_act (str, optional): output activation function. Defaults to None.
    """
    def __init__(self, dim=100, hidden_dim=None, out_dim=None, noise_dim=100, add_bn=False, out_act=None, noise_std=1):
        super().__init__()
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        if hidden_dim is None:
            hidden_dim = dim
        if out_dim is None:
            out_dim = dim
        self.layer1 = [nn.Linear(dim + noise_dim, hidden_dim)]
        self.add_bn = add_bn
        if add_bn:
            self.layer1.append(nn.BatchNorm1d(hidden_dim))
        self.layer1.append(nn.ReLU())
        self.layer1 = nn.Sequential(*self.layer1)
        self.layer2 = nn.Linear(hidden_dim + noise_dim, out_dim)
        if add_bn and out_act == "relu": # for intermediate blocks
            self.layer2 = nn.Sequential(*[self.layer2, nn.BatchNorm1d(out_dim)])
        if out_dim != dim:
            self.layer3 = nn.Linear(dim, out_dim)
        self.dim = dim
        self.out_dim = out_dim
        self.noise_dim = noise_dim
        if out_act == "softmax" and out_dim == 1:
            out_act = "sigmoid"
        self.out_act = get_act_func(out_act)

    def forward(self, x):
        if self.noise_dim > 0:
            eps = torch.randn(x.size(0), self.noise_dim, device=x.device) * self.noise_std
            out = self.layer1(torch.cat([x, eps], dim=1))
            eps = torch.randn(x.size(0), self.noise_dim, device=x.device) * self.noise_std
            out = self.layer2(torch.cat([out, eps], dim=1))
        else:
            out = self.layer2(self.layer1(x))
        if self.out_dim != self.dim:
            out2 = self.layer3(x)
            out = out + out2
        else:
            out += x
        if self.out_act is not None:
            out = self.out_act(out)
        return out


class FiLMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, condition_dim, 
                 hidden_dim=512, noise_dim=0, add_bn=False, resblock=False, 
                 out_act=None, film_pos='out', film_level=1):
        super().__init__()
        self.film_pos = film_pos
        self.film_level = film_level
        film_out_dim = out_dim if film_pos == 'out' else in_dim
        if film_level > 1:
            self.condition_layer = nn.Linear(condition_dim, film_out_dim * 2)
        elif film_level == 1:
            self.condition_layer = nn.Linear(condition_dim, film_out_dim)
        if resblock:
            self.net = StoLayer(in_dim, out_dim, noise_dim, add_bn, out_act)
        else:
            self.net = StoResBlock(in_dim, hidden_dim, out_dim, noise_dim, add_bn, out_act)
        
    def forward(self, x, condition):
        out = self.net(x) if self.film_pos == 'out' else x
        if self.film_level > 1:
            gamma, beta = self.condition_layer(condition).chunk(2, dim=1)         
            out = gamma * out + beta
        elif self.film_level == 1:
            beta = self.condition_layer(condition)
            out = out + beta
        if self.film_pos == 'in':
            out = self.net(out)
        return out


# class FiLMBlockIn(nn.Module):
#     def __init__(self, in_dim, out_dim, condition_dim, 
#                  hidden_dim=512, noise_dim=0, add_bn=False, resblock=False, 
#                  out_act=None, film_level=1):
#         super().__init__()
#         self.condition_layer = nn.Linear(condition_dim, in_dim * 2)
#         if resblock:
#             self.net = StoLayer(in_dim, out_dim, noise_dim, add_bn, out_act)
#         else:
#             self.net = StoResBlock(in_dim, hidden_dim, out_dim, noise_dim, add_bn, out_act)
        
#     def forward(self, x, condition):
#         gamma, beta = self.condition_layer(condition).chunk(2, dim=1)         
#         out = self.net(gamma * x + beta)
#         return out


class StoNetBase(nn.Module):
    def __init__(self, forward_sampling=True):
        super().__init__()
        self.sampling_func = self.forward if forward_sampling else self.sampling_func
    
    @torch.no_grad()
    def predict(self, x, target=["mean"], sample_size=100):
        """Point prediction.

        Args:
            x (torch.Tensor): input data
            target (str or float or list, optional): quantities to predict. float refers to the quantiles. Defaults to ["mean"].
            sample_size (int, optional): sample sizes for each x. Defaults to 100.

        Returns:
            torch.Tensor or list of torch.Tensor: point predictions
                - [:,:,i] gives the i-th sample of all x.
                - [i,:,:] gives all samples of x_i.
            
        Here we do not call `sample` but directly call `forward`.
        """
        samples = self.sample(x=x, sample_size=sample_size, expand_dim=True)
        if not isinstance(target, list):
            target = [target]
        results = []
        extremes = []
        for t in target:
            if t == "mean":
                results.append(samples.mean(dim=len(samples.shape) - 1))
            else:
                if t == "median":
                    t = 0.5
                assert isinstance(t, float)
                results.append(samples.quantile(t, dim=len(samples.shape) - 1))
                if min(t, 1 - t) * sample_size < 10:
                    extremes.append(t)
        
        if len(extremes) > 0:
            print("Warning: the estimate for quantiles at {} with a sample size of {} could be inaccurate. Please increase the `sample_size`.".format(extremes, sample_size))

        if len(results) == 1:
            return results[0]
        else:
            return results

    def sample_onebatch(self, x, sample_size=100, expand_dim=True, require_grad=False):
        """Sampling new response data (for one batch of data).

        Args:
            x (torch.Tensor): new data of predictors of shape [data_size, covariate_dim]
            sample_size (int, optional): new sample size. Defaults to 100.
            expand_dim (bool, optional): whether to expand the sample dimension. Defaults to True.

        Returns:
            torch.Tensor of shape (data_size, response_dim, sample_size) if expand_dim else (data_size*sample_size, response_dim), where response_dim could have multiple channels.
        """
        data_size = x.size(0) ## input data size
        if not require_grad:
            with torch.no_grad():
                ## repeat the data for sample_size times, get a tensor [data, data, ..., data]
                x_rep = x.repeat(sample_size, 1)
                ## samples of shape (data_size*sample_size, response_dim) such that samples[data_size*(i-1):data_size*i,:] contains one sample for each data point, for i = 1, ..., sample_size
                samples = self.sampling_func(x_rep).detach()
        else:
            x_rep = x.repeat(sample_size, 1)
            samples = self.sampling_func(x_rep)
        if not expand_dim:# or sample_size == 1:
            return samples
        else:
            expand_dim = len(samples.shape)
            samples = samples.unsqueeze(expand_dim) ## (data_size*sample_size, response_dim, 1)
            ## a list of length data_size, each element is a tensor of shape (data_size, response_dim, 1)
            samples = list(torch.split(samples, data_size)) 
            samples = torch.cat(samples, dim=expand_dim) ## (data_size, response_dim, sample_size)
            return samples
            # without expanding dimensions:
            # samples.reshape(-1, *samples.shape[1:-1])
    
    def sample_batch(self, x, sample_size=100, expand_dim=True, batch_size=None):
        """Sampling with mini-batches; only used when out-of-memory.

        Args:
            x (torch.Tensor): new data of predictors of shape [data_size, covariate_dim]
            sample_size (int, optional): new sample size. Defaults to 100.
            expand_dim (bool, optional): whether to expand the sample dimension. Defaults to True.
            batch_size (int, optional): batch size. Defaults to None.

        Returns:
            torch.Tensor of shape (data_size, response_dim, sample_size) if expand_dim else (data_size*sample_size, response_dim), where response_dim could have multiple channels.
        """
        if batch_size is not None and batch_size < x.shape[0]:
            test_loader = make_dataloader(x, batch_size=batch_size, shuffle=False)
            samples = []
            for (x_batch,) in test_loader:
                samples.append(self.sample_onebatch(x_batch, sample_size, expand_dim))
            samples = torch.cat(samples, dim=0)
        else:
            samples = self.sample_onebatch(x, sample_size, expand_dim)
        return samples
    
    def sample(self, x, sample_size=100, expand_dim=True, verbose=True):
        """Sampling that adaptively adjusts the batch size according to the GPU memory."""
        batch_size = x.shape[0]
        while True:
            try:
                samples = self.sample_batch(x, sample_size, expand_dim, batch_size)
                break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_size = batch_size // 2
                    if verbose:
                        print("Out of memory; reduce the batch size to {}".format(batch_size))
        return samples
    
    
class StoNet(StoNetBase):
    """Stochastic neural network.

    Args:
        in_dim (int): input dimension 
        out_dim (int): output dimension
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to False.
        out_act (str, optional): output activation function. Defaults to None.
        resblock (bool, optional): whether to use residual blocks. Defaults to False.
    """
    def __init__(self, in_dim, out_dim, num_layer=2, hidden_dim=100, 
                 noise_dim=100, add_bn=False, out_act=None, resblock=False, 
                 noise_all_layer=True, out_bias=True, verbose=True, forward_sampling=True):
        super().__init__(forward_sampling=forward_sampling)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.add_bn = add_bn
        self.noise_all_layer = noise_all_layer
        self.out_bias = out_bias
        if out_act == "softmax" and out_dim == 1:
            out_act = "sigmoid"
        self.out_act = get_act_func(out_act)
        
        self.num_blocks = None
        if resblock:
            if num_layer % 2 != 0:
                num_layer += 1
                print("The number of layers must be an even number for residual blocks. Changed to {}".format(str(num_layer)))
            num_blocks = num_layer // 2
            self.num_blocks = num_blocks
        self.resblock = resblock
        self.num_layer = num_layer
        
        if self.resblock: 
            if self.num_blocks == 1:
                self.net = StoResBlock(dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, 
                                       noise_dim=noise_dim, add_bn=add_bn, out_act=out_act)
            else:
                self.input_layer = StoResBlock(dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, 
                                               noise_dim=noise_dim, add_bn=add_bn, out_act="relu")
                if not noise_all_layer:
                    noise_dim = 0
                self.inter_layer = nn.Sequential(*[StoResBlock(dim=hidden_dim, noise_dim=noise_dim, add_bn=add_bn, out_act="relu")]*(self.num_blocks - 2))
                self.out_layer = StoResBlock(dim=hidden_dim, hidden_dim=hidden_dim, out_dim=out_dim, 
                                             noise_dim=noise_dim, add_bn=add_bn, out_act=out_act) # output layer with concatinated noise
        else:
            self.input_layer = StoLayer(in_dim=in_dim, out_dim=hidden_dim, noise_dim=noise_dim, add_bn=add_bn, out_act="relu", verbose=verbose)
            if not noise_all_layer:
                noise_dim = 0
            self.inter_layer = nn.Sequential(*[StoLayer(in_dim=hidden_dim, out_dim=hidden_dim, noise_dim=noise_dim, add_bn=add_bn, out_act="relu")]*(num_layer - 2))
            # self.out_layer = StoLayer(in_dim=hidden_dim, out_dim=out_dim, noise_dim=noise_dim, add_bn=False, out_act=out_act) # output layer with concatinated noise
            self.out_layer = nn.Linear(hidden_dim, out_dim, bias=out_bias)
            if self.out_act is not None:
                self.out_layer = nn.Sequential(*[self.out_layer, self.out_act])
            
    def forward(self, x):
        if self.num_blocks == 1:
            return self.net(x)
        else:
            return self.out_layer(self.inter_layer(self.input_layer(x)))


class CondStoNet(StoNetBase):
    """Conditional stochastic neural network.

    Args:
        in_dim (int): input dimension 
        out_dim (int): output dimension
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
        out_act (str, optional): output activation function. Defaults to None.
        resblock (bool, optional): whether to use residual blocks. Defaults to False.
        condition_dim
    """
    def __init__(self, in_dim, out_dim, condition_dim, num_layer=2, hidden_dim=100, 
                 noise_dim=100, add_bn=False, out_act=None, resblock=False, 
                 noise_all_layer=True, film_pos='out', film_level=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.add_bn = add_bn
        self.noise_all_layer = noise_all_layer
        
        self.num_blocks = None
        if resblock:
            if num_layer % 2 != 0:
                num_layer += 1
                print("The number of layers must be an even number for residual blocks. Changed to {}".format(str(num_layer)))
            self.num_blocks = num_layer // 2
        self.resblock = resblock
        self.num_layer = num_layer
        
        if resblock:
            num_layer = self.num_blocks
        if self.num_blocks == 1:
            self.net = nn.ModuleList([FiLMBlock(in_dim=in_dim, out_dim=out_dim, condition_dim=condition_dim, hidden_dim=hidden_dim, noise_dim=noise_dim, add_bn=add_bn, resblock=resblock, out_act=out_act, film_pos=film_pos, film_level=film_level)])
        else:
            layers = [FiLMBlock(in_dim=in_dim, out_dim=hidden_dim, condition_dim=condition_dim, hidden_dim=hidden_dim, noise_dim=noise_dim, add_bn=add_bn, resblock=resblock, out_act="relu", film_pos=film_pos, film_level=film_level)]
            if not noise_all_layer:
                noise_dim = 0
            for i in range(num_layer - 2):
                layers.append(FiLMBlock(in_dim=hidden_dim, out_dim=hidden_dim, condition_dim=condition_dim, noise_dim=noise_dim, add_bn=add_bn, resblock=resblock, out_act="relu", film_pos=film_pos, film_level=film_level))
            layers.append(FiLMBlock(in_dim=hidden_dim, out_dim=out_dim, condition_dim=condition_dim, hidden_dim=hidden_dim, noise_dim=noise_dim, add_bn=add_bn, resblock=resblock, out_act=out_act, film_pos=film_pos, film_level=film_level))
            self.net = nn.ModuleList(layers)
            
    def forward(self, x, condition):
        out = x
        for layer in self.net:
            out = layer(out, condition)
        return out


class Net(nn.Module):
    """Deterministic neural network.

    Args:
        in_dim (int, optional): input dimension. Defaults to 1.
        out_dim (int, optional): output dimension. Defaults to 1.
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to False.
        sigmoid (bool, optional): whether to add sigmoid or softmax at the end. Defaults to False.
    """
    def __init__(self, in_dim=1, out_dim=1, num_layer=2, hidden_dim=100, 
                 add_bn=False, sigmoid=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.add_bn = add_bn
        self.sigmoid = sigmoid
        
        net = [nn.Linear(in_dim, hidden_dim)]
        if add_bn:
            net += [nn.BatchNorm1d(hidden_dim)]
        net += [nn.ReLU(inplace=True)]
        for _ in range(num_layer - 2):
            net += [nn.Linear(hidden_dim, hidden_dim)]
            if add_bn:
                net += [nn.BatchNorm1d(hidden_dim)]
            net += [nn.ReLU(inplace=True)]
        net.append(nn.Linear(hidden_dim, out_dim))
        if sigmoid:
            out_act = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)
            net.append(out_act)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ResMLPBlock(nn.Module):
    """MLP residual net block.

    Args:
        dim (int): dimension of input and output.
    """
    def __init__(self, dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        out += x
        return self.relu(out)


class ResMLP(nn.Module):
    """Residual MLP.

    Args:
        in_dim (int, optional): input dimension. Defaults to 1.
        out_dim (int, optional): output dimension. Defaults to 1.
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
    """
    def __init__(self, in_dim=1, out_dim=1, num_layer=2, hidden_dim=100, add_bn=False, sigmoid=False):
        super().__init__()
        out_act = "sigmoid" if sigmoid else None
        if num_layer % 2 != 0:
            num_layer += 1
            print("The number of layers must be an even number for residual blocks. Added one layer.")
        num_blocks = num_layer // 2
        self.num_blocks = num_blocks
        if num_blocks == 1:
            self.net = StoResBlock(dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, 
                                   noise_dim=0, add_bn=add_bn, out_act=out_act)
        else:
            self.input_layer = StoResBlock(dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, 
                                           noise_dim=0, add_bn=add_bn, out_act="relu")
            self.inter_layer = nn.Sequential(*[StoResBlock(dim=hidden_dim, noise_dim=0, add_bn=add_bn, out_act="relu")]*(self.num_blocks - 2))
            self.out_layer = StoResBlock(dim=hidden_dim, hidden_dim=hidden_dim, out_dim=out_dim, 
                                         noise_dim=0, add_bn=add_bn, out_act=out_act)

    def forward(self, x):
        if self.num_blocks == 1:
            return self.net(x)
        else:
            return self.out_layer(self.inter_layer(self.input_layer(x)))