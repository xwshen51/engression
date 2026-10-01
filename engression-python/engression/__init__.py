from .engression import engression, Engressor

__version__ = "0.1.15"

try:
    # pylint: disable=wrong-import-position
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "No module named 'torch', and engression depends on PyTorch (aka 'torch')."
        "Visit https://pytorch.org/ for installation instructions.")
