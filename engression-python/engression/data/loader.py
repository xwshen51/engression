import torch
from torch.utils.data import TensorDataset, DataLoader

def make_dataloader(x, y=None, batch_size=128, shuffle=True, num_workers=0):
    """Make dataloader.

    Args:
        x (torch.Tensor): data of predictors.
        y (torch.Tensor): data of responses.
        batch_size (int, optional): batch size. Defaults to 128.
        shuffle (bool, optional): whether to shuffle data. Defaults to True.
        num_workers (int, optional): number of workers. Defaults to 0.

    Returns:
        DataLoader: data loader
    """
    if y is None:
        dataset = TensorDataset(x)
    else:
        dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def partition_data(x_full, y_full, cut_quantile=0.3, split_train="smaller"):
    """Partition data into training and test sets.

    Args:
        x_full (torch.Tensor): full data of x.
        y_full (torch.Tensor): full data of y.
        cut_quantile (float, optional): quantile of the cutting point of x. Defaults to 0.3.
        split_train (str, optional): which subset is used for for training. choices=["smaller", "larger"]. Defaults to "smaller".

    Returns:
        tuple of torch.Tensors: training and test data.
    """
    # Split data into training and test sets.
    x_cut = torch.quantile(x_full, cut_quantile)
    train_idx = x_full <= x_cut if split_train == "smaller" else x_full >= x_cut
    x_tr = x_full[train_idx]
    y_tr = y_full[train_idx]
    x_te = x_full[~train_idx]
    y_te = y_full[~train_idx]
    
    # Standardize data based on training statistics.
    x_tr_mean = x_tr.mean()
    x_tr_std = x_tr.std()
    y_tr_mean = y_tr.mean()
    y_tr_std = y_tr.std()
    x_tr = (x_tr - x_tr_mean)/x_tr_std
    y_tr = (y_tr - y_tr_mean)/y_tr_std
    x_te = (x_te - x_tr_mean)/x_tr_std
    y_te = (y_te - y_tr_mean)/y_tr_std
    x_full_normal = (x_full - x_tr_mean)/x_tr_std
    return x_tr.unsqueeze(1), y_tr.unsqueeze(1), x_te.unsqueeze(1), y_te.unsqueeze(1), x_full_normal