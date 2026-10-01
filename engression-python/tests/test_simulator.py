"""Tests of the data simulator."""
import math

import torch

from engression.data.simulator import preanm_simulator


def test_true_mean():
    """The true conditional mean returned for evaluation used Gaussian noise whatever the noise distribution."""
    torch.manual_seed(0)
    _, _, mean_uniform = preanm_simulator("square", n=20, x_lower=-1, x_upper=-1, noise_dist="uniform", train=False)
    _, _, mean_gaussian = preanm_simulator("square", n=20, x_lower=-1, x_upper=-1, noise_dist="gaussian", train=False)
    # E[max(e - 1, 0)^2 / 2] for e uniform on (-sqrt(3), sqrt(3)) and for e standard Gaussian
    exact_uniform = (math.sqrt(3) - 1) ** 3 / (12 * math.sqrt(3))
    exact_gaussian = (2 * (1 - 0.5 * (1 + math.erf(1 / math.sqrt(2)))) - math.exp(-0.5) / math.sqrt(2 * math.pi)) / 2
    assert abs(mean_uniform.mean().item() - exact_uniform) < 0.002
    assert abs(mean_gaussian.mean().item() - exact_gaussian) < 0.002
