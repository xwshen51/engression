"""Tests of the gradient estimate of generalized engression (`gem_loss_two_sample`).

For discrete responses, the expected loss given the outputs (mu, mu', s) of the networks is a finite sum over the outcomes of the
two samples, with probabilities given by Gaussian integrals, so its exact gradient is available and the estimate is compared with it.
"""
import itertools
import math

import pytest
import torch

from engression.gem import gem_loss_two_sample
from engression.links import link, parse_data_type

@pytest.fixture(autouse=True)
def double_precision():
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


def make_draw(mu, s, blocks, size, generator):
    """One sample of size `size` from the model with the outputs of g and sigma fixed at (mu, s) for all data points."""
    g = mu.repeat(size, 1).requires_grad_()
    sigma = s.repeat(size, 1).requires_grad_()
    eta = torch.randn(size, mu.numel(), generator=generator)
    return link(g + sigma * eta, blocks), g, sigma, eta


def estimate_gradient(y, mu, mup, s, blocks, size, seed, **kwargs):
    """Monte Carlo average of the gradient estimates with respect to (mu, mu', s) and its standard errors."""
    generator = torch.Generator().manual_seed(seed)
    draw1 = make_draw(mu, s, blocks, size, generator)
    draw2 = make_draw(mup, s, blocks, size, generator)
    surrogate, _ = gem_loss_two_sample(y.repeat(size, 1), draw1, draw2, blocks, **kwargs)
    surrogate.backward()
    # a row of `.grad` is the estimate from one data point divided by the number of data points
    grads = [draw1[1].grad * size, draw2[1].grad * size, (draw1[2].grad + draw2[2].grad) * size]
    return [grad.mean(dim=0) for grad in grads], [grad.std(dim=0) / math.sqrt(size) for grad in grads]


def cdf(t):
    return 0.5 * (1 + torch.erf(t / math.sqrt(2)))


def integrate(f, lower=-15., upper=15., num=6001):
    t = torch.linspace(lower, upper, num).unsqueeze(1)
    return torch.trapezoid(f(t), t, dim=0)


def outcome_probs(mu, s, name, num_level=None):
    """Outcomes of one block and their probabilities under the link of h(mu + s * eta) with standard Gaussian eta."""
    k = mu.numel()
    s = s.expand(k)
    density = lambda t, i: torch.exp(-0.5 * ((t - mu[i]) / s[i]) ** 2) / (s[i] * math.sqrt(2 * math.pi))
    if name == "multilabel":
        outcomes = torch.tensor(list(itertools.product([-1., 1.], repeat=k)))
        probs = torch.stack([cdf(a * mu / s).prod() for a in outcomes])
    elif name == "ordinal":
        levels = torch.arange(1, num_level + 1).to(mu.dtype)
        upper = torch.cat([cdf((levels[:-1].unsqueeze(1) + 0.5 - mu) / s), torch.ones(1, k)])
        lower = torch.cat([torch.zeros(1, k), upper[:-1]])
        level_probs = upper - lower
        index = list(itertools.product(range(num_level), repeat=k))
        outcomes = torch.tensor([[levels[l] for l in idx] for idx in index])
        probs = torch.stack([torch.stack([level_probs[l, j] for j, l in enumerate(idx)]).prod() for idx in index])
    elif name == "multiclass":
        outcomes = torch.eye(k)
        probs = torch.cat([integrate(lambda t: density(t, i) * torch.stack(
            [cdf((t - mu[l]) / s[l]) for l in range(k) if l != i]).prod(dim=0)) for i in range(k)])
    elif name == "ranking":
        assert k == 3
        outcomes, probs = [], []
        for a, b, c in itertools.permutations(range(3)):
            # z_a > z_b > z_c: items a, b, c take the positions 1, 2, 3
            rank = torch.zeros(3)
            rank[a], rank[b], rank[c] = 1, 2, 3
            outcomes.append(rank)
            probs.append(integrate(lambda t: density(t, b) * (1 - cdf((t - mu[a]) / s[a])) * cdf((t - mu[c]) / s[c])))
        outcomes, probs = torch.stack(outcomes), torch.cat(probs)
    return outcomes, probs


def exact_gradient(y, mu, mup, s, blocks, beta=1):
    """Gradient with respect to (mu, mu', s) of the expected loss over (eta, eta') of a response with discrete blocks only."""
    mu, mup, s = mu.clone().requires_grad_(), mup.clone().requires_grad_(), s.clone().requires_grad_()
    probs = []
    for m in [mu, mup]:
        outcomes, prob = torch.zeros(1, 0), torch.ones(1)
        for block in blocks:
            s_block = s if s.numel() == 1 else s[block.start:block.end]
            o, p = outcome_probs(m[block.start:block.end], s_block, block.name, block.num_level)
            outcomes = torch.cat([outcomes.repeat_interleave(len(o), dim=0), o.repeat(len(outcomes), 1)], dim=1)
            prob = (prob.unsqueeze(1) * p.unsqueeze(0)).flatten()
        probs.append(prob)
    assert torch.allclose(probs[0].sum(), torch.tensor(1.)) and torch.allclose(probs[1].sum(), torch.tensor(1.))
    dist_y = (outcomes - y).norm(dim=1).pow(beta)
    dist = torch.cdist(outcomes, outcomes).pow(beta)
    loss = (probs[0] * dist_y).sum() / 2 + (probs[1] * dist_y).sum() / 2 - (probs[0].unsqueeze(1) * probs[1].unsqueeze(0) * dist).sum() / 2
    return torch.autograd.grad(loss, [mu, mup, s])


def test_matches_formula():
    """The gradients with respect to the outputs of g and sigma are those of Proposition 1 on the discrete coordinates and
    the derivatives of the joint norms on the continuous coordinates."""
    torch.manual_seed(0)
    n = 50
    blocks = parse_data_type({"continuous": 0, "multilabel": 2, "multiclass": 5, "ordinal:4": 8, "ranking": 9}, 12)
    y = link(torch.randn(n, 12) * 2 + 1, blocks)
    for sigma_dim in [1, 12]:
        g, gp = torch.randn(n, 12, requires_grad=True), torch.randn(n, 12, requires_grad=True)
        sigma, sigmap = [(torch.rand(n, sigma_dim) + 0.2).requires_grad_() for _ in range(2)]
        eta, etap = torch.randn(n, 12), torch.randn(n, 12)
        x, xp = link(g + sigma * eta, blocks), link(gp + sigmap * etap, blocks)
        surrogate, loss = gem_loss_two_sample(y, (x, g, sigma, eta), (xp, gp, sigmap, etap), blocks)
        surrogate.backward()

        x, xp = x.detach(), xp.detach()
        norm_y, normp_y, norm = (x - y).norm(dim=1, keepdim=True), (xp - y).norm(dim=1, keepdim=True), (x - xp).norm(dim=1, keepdim=True)
        assert torch.allclose(loss[0], (norm_y / 2 + normp_y / 2 - norm / 2).mean())
        c, cp = norm_y - norm, normp_y - norm
        G, Gp = c * eta / sigma / 2, cp * etap / sigmap / 2
        S, Sp = c * (eta ** 2 - 1) / sigma / 2, cp * (etap ** 2 - 1) / sigmap / 2
        # continuous coordinates
        G[:, :2] = ((x - y) / norm_y - (x - xp) / norm)[:, :2] / 2
        Gp[:, :2] = ((xp - y) / normp_y - (xp - x) / norm)[:, :2] / 2
        S[:, :2], Sp[:, :2] = (eta * G)[:, :2], (etap * Gp)[:, :2]
        if sigma_dim == 1:
            S, Sp = S.sum(dim=1, keepdim=True), Sp.sum(dim=1, keepdim=True)
        assert torch.allclose(g.grad, G.detach() / n, atol=1e-12)
        assert torch.allclose(gp.grad, Gp.detach() / n, atol=1e-12)
        assert torch.allclose(sigma.grad, S.detach() / n, atol=1e-12)
        assert torch.allclose(sigmap.grad, Sp.detach() / n, atol=1e-12)


def test_control_variate_centers_the_costs():
    """With the control variate, the cost of a binary coordinate is half the change of the norms when the coordinate flips,
    at most 2 in absolute value; the other coordinates keep the plain cost."""
    torch.manual_seed(0)
    n = 200
    blocks = parse_data_type({"multilabel": 0, "multiclass": 6}, 9)
    y = link(torch.randn(n, 9), blocks)
    g, gp = torch.randn(n, 9, requires_grad=True), torch.randn(n, 9, requires_grad=True)
    sigma, sigmap = torch.ones(n, 9, requires_grad=True), torch.ones(n, 9, requires_grad=True)
    eta, etap = torch.randn(n, 9), torch.randn(n, 9)
    x, xp = link(g + sigma * eta, blocks), link(gp + sigmap * etap, blocks)
    surrogate, _ = gem_loss_two_sample(y, (x, g, sigma, eta), (xp, gp, sigmap, etap), blocks, control_variate=True)
    surrogate.backward()
    cost = g.grad * 2 * n / eta   # since sigma = 1

    def flip(v, j):
        v = v.clone()
        v[:, j] = -v[:, j]
        return v
    for j in range(6):
        change = ((x - y).norm(dim=1) - (flip(x, j) - y).norm(dim=1)) - ((x - xp).norm(dim=1) - (flip(x, j) - xp).norm(dim=1))
        assert torch.allclose(cost[:, j], change / 2, atol=1e-10)
    assert cost[:, :6].abs().max() <= 2
    plain = (x - y).norm(dim=1) - (x - xp).norm(dim=1)
    assert torch.allclose(cost[:, 6:], plain.unsqueeze(1).repeat(1, 3), atol=1e-10)


def test_control_variate_without_binary_labels():
    torch.manual_seed(0)
    blocks = parse_data_type({"continuous": 0, "multiclass": 2, "ranking": 5}, 8)
    y = link(torch.randn(30, 8), blocks)
    grads = []
    for control_variate in [False, True]:
        generator = torch.Generator().manual_seed(1)
        draw1 = make_draw(torch.zeros(8), torch.ones(1), blocks, 30, generator)
        draw2 = make_draw(torch.zeros(8), torch.ones(1), blocks, 30, generator)
        surrogate, _ = gem_loss_two_sample(y, draw1, draw2, blocks, control_variate=control_variate)
        surrogate.backward()
        grads.append([draw1[1].grad, draw2[1].grad, draw1[2].grad, draw2[2].grad])
    assert all(torch.equal(a, b) for a, b in zip(*grads))


CASES = {
    "multilabel": ("multilabel", [1., -1., 1.], [0.3, -0.8, 1.2], [-0.5, 0.4, 0.9], [0.7, 1.1, 0.5]),
    "ordinal": ("ordinal:4", [2., 4.], [1.7, 3.1], [2.6, 2.2], [0.8, 0.6]),
    "multiclass": ("multiclass", [0., 1., 0.], [0.2, -0.3, 0.5], [-0.4, 0.6, 0.1], [0.9, 0.6, 1.2]),
    "ranking": ("ranking", [2., 1., 3.], [0.4, -0.2, 0.1], [-0.3, 0.5, 0.2], [0.8, 1.1, 0.6]),
    "mixed": ({"multilabel": 0, "ordinal:3": 2, "multiclass": 3}, [1., -1., 2., 0., 0., 1.],
              [0.3, -0.6, 1.8, 0.2, -0.3, 0.5], [-0.5, 0.2, 2.4, -0.4, 0.6, 0.1], [0.7, 1.1, 0.8, 0.9, 0.6, 1.2]),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("sigma_dim", ["scalar", "vector"])
@pytest.mark.parametrize("control_variate", [False, True])
@pytest.mark.parametrize("beta", [1, 0.5])
def test_unbiased(case, sigma_dim, control_variate, beta):
    """The Monte Carlo average of the gradient estimates agrees with the exact gradient of the expected loss."""
    data_type, y, mu, mup, s = CASES[case]
    y, mu, mup, s = torch.tensor([y]), torch.tensor(mu), torch.tensor(mup), torch.tensor(s)
    if sigma_dim == "scalar":
        s = s[:1]
    blocks = parse_data_type(data_type, mu.numel())
    exact = exact_gradient(y, mu, mup, s, blocks, beta)
    estimates, std_errors = estimate_gradient(y, mu, mup, s, blocks, size=200000, seed=0, beta=beta, control_variate=control_variate)
    for grad, estimate, std_error in zip(exact, estimates, std_errors):
        assert grad.abs().max() > 1e-3
        assert ((estimate - grad).abs() < 5 * std_error).all()
        assert (std_error < 0.02).all()


@pytest.mark.parametrize("sigma_dim", ["scalar", "vector"])
def test_unbiased_with_continuous_coordinates(sigma_dim):
    """On a mixed response, differentiating through the continuous coordinates has the same mean as estimating the gradient
    from the values of the loss on all coordinates."""
    blocks = parse_data_type({"continuous": 0, "multilabel": 2, "multiclass": 4}, 7)
    y = torch.tensor([[0.4, -1.0, 1., -1., 0., 1., 0.]])
    mu = torch.tensor([0.1, 0.5, 0.3, -0.8, 0.2, -0.3, 0.5])
    mup = torch.tensor([-0.6, 0.2, -0.5, 0.4, -0.4, 0.6, 0.1])
    s = torch.tensor([0.9, 0.7, 0.7, 1.1, 0.9, 0.6, 1.2])
    if sigma_dim == "scalar":
        s = s[:1]
    size = 1000000
    estimates, std_errors = estimate_gradient(y, mu, mup, s, blocks, size=size, seed=0)

    generator = torch.Generator().manual_seed(1)
    eta, etap = torch.randn(size, 7, generator=generator), torch.randn(size, 7, generator=generator)
    x, xp = link(mu + s * eta, blocks), link(mup + s * etap, blocks)
    norm = (x - xp).norm(dim=1, keepdim=True)
    c, cp = (x - y).norm(dim=1, keepdim=True) - norm, (xp - y).norm(dim=1, keepdim=True) - norm
    grads = [c * eta / s / 2, cp * etap / s / 2, (c * (eta ** 2 - 1) + cp * (etap ** 2 - 1)) / s / 2]
    if sigma_dim == "scalar":
        grads[2] = grads[2].sum(dim=1, keepdim=True)
    for grad, estimate, std_error in zip(grads, estimates, std_errors):
        std_error_diff = (grad.var(dim=0) / size + std_error ** 2).sqrt()
        assert ((estimate - grad.mean(dim=0)).abs() < 5 * std_error_diff).all()
        assert (std_error_diff < 0.01).all()
    # differentiating through the continuous coordinates reduces the variance
    for grad, std_error in zip(grads[:2], std_errors[:2]):
        assert (std_error[:2] < grad[:, :2].std(dim=0) / math.sqrt(size) / 2).all()


def test_control_variate_reduces_variance():
    torch.manual_seed(0)
    k = 50
    blocks = parse_data_type("multilabel", k)
    y = link(torch.randn(1, k), blocks)
    mu, mup, s = torch.randn(k) * 0.5, torch.randn(k) * 0.5, torch.ones(1)
    _, plain = estimate_gradient(y, mu, mup, s, blocks, size=20000, seed=0)
    _, centered = estimate_gradient(y, mu, mup, s, blocks, size=20000, seed=0, control_variate=True)
    for std_error_plain, std_error_centered in zip(plain, centered):
        assert (std_error_centered < std_error_plain / 3).all()
