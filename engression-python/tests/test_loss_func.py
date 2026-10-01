"""Tests of the energy loss against exact computations."""
import pytest
import torch

from engression.loss_func import energy_loss


def exact_terms(y, samples, beta):
    """The two terms of the energy loss for samples of shape (data_size, response_dim, sample_size), the second averaged
    over the pairs of different samples."""
    m = samples.size(2)
    s1 = (samples - y.unsqueeze(2)).norm(dim=1).pow(beta).mean()
    s2 = (samples.unsqueeze(3) - samples.unsqueeze(2)).norm(dim=1).pow(beta).sum() / (y.size(0) * m * (m - 1))
    return s1, s2


@pytest.mark.parametrize("beta", [0.1, 0.5, 1, 1.5])
@pytest.mark.parametrize("sample_size", [2, 5])
def test_beta(beta, sample_size):
    """For a beta that is not an integer, the pairs of a sample with itself used to add 1e-5 ** beta to the second term,
    0.32 for beta = 0.1."""
    torch.manual_seed(0)
    y, samples = torch.randn(300, 2), torch.randn(300, 2, sample_size)
    loss = energy_loss(y, [samples[:, :, i] for i in range(sample_size)], beta=beta, verbose=True)
    s1, s2 = exact_terms(y, samples, beta)
    assert loss[1].item() == pytest.approx(s1.item(), rel=1e-4) and loss[2].item() == pytest.approx(s2.item(), rel=1e-4)
    assert loss[0].item() == pytest.approx((s1 - s2 / 2).item(), rel=1e-4)


def test_many_samples_far_from_origin():
    """With more than 25 samples, torch.cdist computed the distances by a matrix product, which lost precision for
    samples far from the origin: here the second term came out 4% too small."""
    torch.manual_seed(0)
    y, samples = torch.randn(200, 1) + 3000, torch.randn(200, 1, 50) + 3000
    loss = energy_loss(y, [samples[:, :, i] for i in range(50)], beta=1, verbose=True)
    s1, s2 = exact_terms(y, samples, 1)
    assert loss[2].item() == pytest.approx(s2.item(), rel=1e-4)


def test_one_sample():
    """With one sample for each data point, the second term is not defined; the loss used to be nan."""
    with pytest.raises(ValueError):
        energy_loss(torch.randn(10, 2), torch.randn(10, 2))
