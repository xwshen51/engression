import torch
import torch.nn as nn


class StoLayer(nn.Module):    
    """A stochastic layer.

    Args:
        in_dim (int): input dimension 
        out_dim (int): output dimension
        noise_dim (int, optional): noise dimension. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
    """
    def __init__(self, in_dim, out_dim, noise_dim=100, add_bn=True, out_act=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.noise_dim = noise_dim
        self.add_bn = add_bn
        
        layer = [nn.Linear(in_dim + noise_dim, out_dim)]
        if add_bn:
            layer += [nn.BatchNorm1d(out_dim)]
        self.layer = nn.Sequential(*layer)
        if out_act == "relu":
            self.out_act = nn.ReLU(inplace=True)
        elif out_act == "sigmoid":
            self.out_act = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)
        else:
            self.out_act = None
    
    def forward(self, x):
        eps = torch.randn(x.size(0), self.noise_dim, device=x.device)
        out = torch.cat([x, eps], dim=1)
        out = self.layer(out)
        if self.out_act is not None:
            out = self.out_act(out)
        return out


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
    def __init__(self, dim=100, hidden_dim=None, out_dim=None, noise_dim=100, add_bn=True, out_act=None):
        super().__init__()
        self.noise_dim = noise_dim
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
        if out_act == "relu":
            self.out_act = nn.ReLU(inplace=True)
        elif out_act == "sigmoid":
            self.out_act = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)
        else:
            self.out_act = None

    def forward(self, x):
        if self.noise_dim > 0:
            eps = torch.randn(x.size(0), self.noise_dim, device=x.device)
            out = self.layer1(torch.cat([x, eps], dim=1))
            eps = torch.randn(x.size(0), self.noise_dim, device=x.device)
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
    
    
class StoNet(nn.Module):
    """Stochastic neural network.

    Args:
        in_dim (int): input dimension 
        out_dim (int): output dimension
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
        sigmoid (bool, optional): whether to add a sigmoid function at the model output. Defaults to False.
        resblock (bool, optional): whether to use residual blocks. Defaults to False.
    """
    def __init__(self, in_dim, out_dim, num_layer=2, hidden_dim=100, 
                 noise_dim=100, add_bn=True, sigmoid=False, resblock=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.add_bn = add_bn
        self.sigmoid = sigmoid
        out_act = "sigmoid" if sigmoid else None
        
        self.num_blocks = None
        if resblock:
            if num_layer % 2 != 0:
                num_layer += 1
                # print("The number of layers must be an even number for residual blocks. Added one layer.")
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
                self.inter_layer = nn.Sequential(*[StoResBlock(dim=hidden_dim, noise_dim=noise_dim, add_bn=add_bn, out_act="relu")]*(self.num_blocks - 2))
                self.out_layer = StoResBlock(dim=hidden_dim, hidden_dim=hidden_dim, out_dim=out_dim, 
                                             noise_dim=noise_dim, add_bn=add_bn, out_act=out_act) # output layer with concatinated noise
        else:
            self.input_layer = StoLayer(in_dim=in_dim, out_dim=hidden_dim, noise_dim=noise_dim, add_bn=add_bn, out_act="relu")
            self.inter_layer = nn.Sequential(*[StoLayer(in_dim=hidden_dim, out_dim=hidden_dim, noise_dim=noise_dim, add_bn=add_bn, out_act="relu")]*(num_layer - 2))
            # self.out_layer = StoLayer(in_dim=hidden_dim, out_dim=out_dim, noise_dim=noise_dim, add_bn=False, out_act=out_act) # output layer with concatinated noise
            self.out_layer = nn.Linear(hidden_dim, out_dim)
            if sigmoid:
                out_act = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)
                self.out_layer = nn.Sequential(*[self.out_layer, out_act])
                
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
    
    def sample(self, x, sample_size=100, expand_dim=True):
        """Sample new response data.

        Args:
            x (torch.Tensor): new data of predictors of shape [data_size, covariate_dim]
            sample_size (int, optional): new sample size. Defaults to 100.
            expand_dim (bool, optional): whether to expand the sample dimension. Defaults to True.

        Returns:
            torch.Tensor of shape (data_size, response_dim, sample_size) if expand_dim else (data_size*sample_size, response_dim), where response_dim could have multiple channels.
        """
        data_size = x.size(0) ## input data size
        with torch.no_grad():
            ## repeat the data for sample_size times, get a tensor [data, data, ..., data]
            x_rep = x.repeat(sample_size, 1)
            ## samples of shape (data_size*sample_size, response_dim) such that samples[data_size*(i-1):data_size*i,:] contains one sample for each data point, for i = 1, ..., sample_size
            samples = self.forward(x=x_rep).detach()
        if not expand_dim:
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
        
    def forward(self, x):
        if self.num_blocks == 1:
            return self.net(x)
        else:
            return self.out_layer(self.inter_layer(self.input_layer(x)))


class Net(nn.Module):
    """Deterministic neural network.

    Args:
        in_dim (int, optional): input dimension. Defaults to 1.
        out_dim (int, optional): output dimension. Defaults to 1.
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
        sigmoid (bool, optional): whether to add sigmoid or softmax at the end. Defaults to False.
    """
    def __init__(self, in_dim=1, out_dim=1, num_layer=2, hidden_dim=100, 
                 add_bn=True, sigmoid=False):
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
    def __init__(self, in_dim=1, out_dim=1, num_layer=2, hidden_dim=100, add_bn=True, sigmoid=False):
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