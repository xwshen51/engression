import os
import contextlib
import numpy as np
import torch

def to_tensor(x):
    """Convert data to a tensor of the default floating-point type of PyTorch (float32 unless changed), which the networks use.

    Args:
        x (torch.Tensor, numpy.ndarray or array-like): data of any numeric type.

    Returns:
        torch.Tensor: data as a floating-point tensor, on the device of x if it is a tensor.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(np.ascontiguousarray(x))
    return x.to(torch.get_default_dtype())

def vectorize(x, multichannel=False):
    """Vectorize data in any shape.

    Args:
        x (torch.Tensor): input data
        multichannel (bool, optional): whether to keep the multiple channels (in the second dimension). Defaults to False.

    Returns:
        torch.Tensor: data of shape (sample_size, dimension) or (sample_size, num_channel, dimension) if multichannel is True.
    """
    if len(x.shape) == 1:
        return x.unsqueeze(1)
    if len(x.shape) == 2:
        return x
    else:
        if not multichannel: # one channel
            return x.reshape(x.shape[0], -1)
        else: # multi-channel
            return x.reshape(x.shape[0], x.shape[1], -1)
        
def cor(x, y):
    """Compute the correlation between two signals.

    Args:
        x (torch.Tensor): input data
        y (torch.Tensor): input data

    Returns:
        torch.Tensor: correlation between x and y
    """
    x = vectorize(x)
    y = vectorize(y)
    x = x - x.mean(0)
    y = y - y.mean(0)
    return ((x * y).mean()) / (x.std(unbiased=False) * y.std(unbiased=False))

def make_folder(name):
    """Make a folder.

    Args:
        name (str): folder name.
    """
    if not os.path.exists(name):
        print('Creating folder: {}'.format(name))
        os.makedirs(name)

def check_for_gpu(device):
    """Check if a CUDA device is available.

    Args:
        device (torch.device): current set device.
    """
    if device.type == "cuda":
        if torch.cuda.is_available():
            print("GPU is available, running on GPU.\n")
        else:
            print("GPU is NOT available, running instead on CPU.\n")
    elif device.type == "mps":
        print("Running on MPS.\n")
    else:
        if torch.cuda.is_available():
            print("Warning: You have a CUDA device, so you may consider using GPU for potential acceleration\n by setting device to 'cuda'.\n")
        else:
            print("Running on CPU.\n")


@contextlib.contextmanager
def keep_rng_state(device):
    """Restore the states of the random number generators of the CPU and of a CUDA device when leaving the context, 
    so that the random numbers drawn in the context do not change those drawn after it. The generator of an MPS device 
    is left alone, since some versions of PyTorch cannot restore its state.

    Args:
        device (torch.device): device.
    """
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
