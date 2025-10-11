from .engression import engression

try:
    # pylint: disable=wrong-import-position
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "No module named 'torch', and engression depends on PyTorch (aka 'torch')."
        "Visit https://pytorch.org/ for installation instructions.")
