import torch
import torch.nn as nn


class StoLayer(nn.Module):    
    """A stochastic layer.

    Args:
        in_dim (int): input dimension 
        out_dim (int): output dimension
        noise_dim (int, optional): noise dimension. Defaults to 100.
    """
    def __init__(self, in_dim, out_dim, noise_dim=100):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.noise_dim = noise_dim
        layer = [
            nn.Linear(in_dim + noise_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        ]
        self.layer = nn.Sequential(*layer)
    
    def forward(self, x, inject_noise=True):
        eps = torch.randn(x.size(0), self.noise_dim, device=x.device) if inject_noise else torch.zeros(x.size(0), self.noise_dim, device=x.device)
        x = torch.cat([x, eps], dim=1)
        return self.layer(x)
    
    
class StoNet(nn.Module):
    """Stochastic neural network.

    Args:
        in_dim (int): input dimension 
        out_dim (int): output dimension
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        noise_dim (int, optional): noise dimension. Defaults to 100.
    """
    def __init__(self, in_dim, out_dim, num_layer=2, hidden_dim=100, noise_dim=100):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        
        self.input_layer = StoLayer(in_dim, hidden_dim, noise_dim)
        if num_layer > 2:
            inter_layer = [StoLayer(hidden_dim, hidden_dim, noise_dim)]
            for i in range(num_layer - 3):
                inter_layer.append(StoLayer(hidden_dim, hidden_dim, noise_dim))
            self.inter_layer = nn.Sequential(*inter_layer)
        self.out_layer = nn.Linear(hidden_dim, out_dim)
                
    def predict(self, x, target=["mean"], sample_size=100):
        """Point prediction.

        Args:
            x (torch.Tensor): _description_
            target (str or float or list, optional): single-valued functional to predict. float refers to the quantiles. Defaults to ["mean"].
            sample_size (int, optional): sample sizes for each x. Defaults to 100.

        Returns:
            torch.Tensor or list of torch.Tensor: point predictions
                - [:,:,i] gives the i-th sample of all x.
                - [i,:,:] gives all samples of x_i.
            
        Here we do not call `sample` but directly call `forward`.
        """
        # eval_size = x.size(0)
        # # Sampling
        # with torch.no_grad():
        #     x_rep = torch.repeat_interleave(x, (sample_size * torch.ones(eval_size, device=x.device)).long(), dim=0)
        #     y_pred = self.forward(x=x_rep).detach()
        #     y_pred = list(torch.split(y_pred, sample_size))
            
        # if not isinstance(target, list):
        #     target = [target]
        # results = []
        # extremes = []
        # for t in target:
        #     if t == "mean":
        #         results.append(torch.cat([y_pred[i].mean().unsqueeze(0) for i in range(eval_size)], dim=0).unsqueeze(1))
        #     else:
        #         if t == "median":
        #             t = 0.5
        #         assert isinstance(t, float)
        #         results.append(torch.cat([y_pred[i].quantile(t).unsqueeze(0) for i in range(eval_size)], dim=0).unsqueeze(1))
        #         if min(t, 1-t) * sample_size < 10:
        #             extremes.append(t)
        
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
            torch.Tensor of shape (data_size, response_dim, sample_size), where response_dim could have multiple channels.
        """
        eval_size = x.size(0)
        with torch.no_grad():
            # x_rep = torch.repeat_interleave(x, (sample_size * torch.ones(eval_size, device=x.device)).long(), dim=0)
            x_rep = x.repeat(sample_size, 1)
            samples = self.forward(x=x_rep).detach()
        if not expand_dim:
            return samples
        else:
            expand_dim = len(samples.shape)
            samples = samples.unsqueeze(expand_dim)
            samples = list(torch.split(samples, eval_size))
            samples = torch.cat(samples, dim=expand_dim)
            return samples
            # without expanding dimensions:
            # samples.reshape(-1, *samples.shape[1:-1])
        
    def forward(self, x, inject_noise=True):
        x = self.input_layer(x, inject_noise)
        for i in range(self.num_layer - 2):
            x = self.inter_layer[i](x, inject_noise)
        x = self.out_layer(x)
        return x


class Net(nn.Module):
    """Deterministic neural network.

    Args:
        in_dim (int, optional): input dimension. Defaults to 1.
        out_dim (int, optional): output dimension. Defaults to 1.
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
    """
    def __init__(self, in_dim=1, out_dim=1, num_layer=2, hidden_dim=100):
        super().__init__()
        net = [
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layer - 2):
            net += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        net.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
