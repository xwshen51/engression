"""Tests of generalized engression through `engression` and `Engressor`."""
import math

import pytest
import torch

from engression import engression
from engression.engression import Engressor
from engression.gem import GEMNet


def simulate(data_type, n=400, seed=0):
    """Responses read off a latent Gaussian vector whose mean depends on x."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=generator)
    z = x[:, :1] * torch.tensor([1., -1., 0.5, 0.]) + torch.randn(n, 4, generator=generator)
    if data_type == "multilabel":
        y = (z > 0).float()
    elif data_type == "multiclass":
        y = torch.nn.functional.one_hot(z.argmax(dim=1), 4).float()
    elif data_type == "ordinal:5":
        y = (z[:, :2] + 3).round().clamp(1, 5)
    elif data_type == "ranking":
        y = z.argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    else:
        y = torch.cat([z[:, :1] * 3 + 10, (z[:, 1:3] > 0).float(), (z[:, 3:] + 2).round().clamp(1, 3),
                       torch.nn.functional.one_hot(z[:, :3].argmax(dim=1), 3).float()], dim=1)
    return x, y


MIXED = {"continuous": 0, "multilabel": 1, "ordinal:3": 3, "multiclass": 4}
VALUES = {"multilabel": {0., 1.}, "multiclass": {0., 1.}, "ordinal:5": {1., 2., 3., 4., 5.}, "ranking": {1., 2., 3., 4.}}


@pytest.mark.parametrize("data_type", list(VALUES))
@pytest.mark.parametrize("batch_size", [None, 64])
@pytest.mark.parametrize("sigma_dim", ["scalar", "vector"])
@pytest.mark.parametrize("control_variate", [False, True])
def test_fit_and_sample(data_type, batch_size, sigma_dim, control_variate):
    x, y = simulate(data_type)
    engressor = engression(x, y, data_type=data_type, num_epochs=3, batch_size=batch_size,
                           sigma_dim=sigma_dim, control_variate=control_variate, verbose=False)
    assert type(engressor.model) is GEMNet
    assert all(param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
               for param in engressor.model.parameters())
    assert len(engressor.tr_loss) == 3 and all(math.isfinite(loss) for loss in engressor.tr_loss)

    y_samples = engressor.sample(x[:9], sample_size=50)
    assert y_samples.shape == (9, y.size(1), 50)
    assert set(y_samples.unique().tolist()) <= VALUES[data_type]
    if data_type == "multiclass":
        assert torch.equal(y_samples.sum(dim=1), torch.ones(9, 50))
    if data_type == "ranking":
        assert torch.equal(y_samples.sort(dim=1).values, torch.arange(1., 5.).view(1, 4, 1).repeat(9, 1, 50))
    assert engressor.sample(x[:9], sample_size=50, expand_dim=False).shape == (450, y.size(1))
    assert engressor.sample(x[:9], sample_size=1).shape == (9, y.size(1))

    y_pred = engressor.predict(x[:9], target="mean")
    assert y_pred.shape == (9, y.size(1))
    if data_type == "multiclass":
        assert torch.allclose(y_pred.sum(dim=1), torch.ones(9))
    assert set(engressor.predict(x[:9], target="median").unique().tolist()) <= VALUES[data_type] | {1.5, 2.5, 3.5, 4.5, 0.5}
    assert all(math.isfinite(engressor.eval_loss(x, y, loss_type=loss_type)) for loss_type in ["l2", "l1", "cor", "energy"])


@pytest.mark.parametrize("num_layer", [2, 3, 4, 5])
@pytest.mark.parametrize("sigma_dim", ["scalar", "vector"])
def test_no_shared_parameters(num_layer, sigma_dim):
    """No parameter appears twice in the network, and all of them are trained."""
    x, y = simulate("mixed")
    engressor = engression(x, y, data_type=MIXED, num_layer=num_layer, hidden_dim=16, sigma_dim=sigma_dim, 
                           num_epochs=2, verbose=False)
    params = [param for param in engressor.model.state_dict(keep_vars=True).values() if isinstance(param, torch.nn.Parameter)]
    assert len({id(param) for param in params}) == len(params)
    assert len(engressor.model.g_net.inter_layer) == max(num_layer - 2, 0)
    assert all(param.grad is not None and param.grad.abs().sum() > 0 for param in params)


@pytest.mark.parametrize("sigma_dim", ["scalar", "vector"])
@pytest.mark.parametrize("control_variate", [False, True])
@pytest.mark.parametrize("standardize", [True, False])
def test_mixed(sigma_dim, control_variate, standardize):
    x, y = simulate("mixed")
    engressor = engression(x, y, data_type=MIXED, num_epochs=3, sigma_dim=sigma_dim, control_variate=control_variate,
                           standardize=standardize, verbose=False)
    assert all(param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
               for param in engressor.model.parameters())
    if standardize:
        # only the continuous column is standardized
        assert torch.allclose(engressor.y_mean, torch.cat([y[:, :1].mean(dim=0), torch.zeros(6)]))
        assert torch.allclose(engressor.y_std, torch.cat([y[:, :1].std(dim=0), torch.ones(6)]))
        assert torch.equal(engressor.y_mean[1:], torch.zeros(6)) and torch.equal(engressor.y_std[1:], torch.ones(6))
    assert engressor.y_zero_one.tolist() == [False, True, True, False, False, False, False]
    y_samples = engressor.sample(x[:9], sample_size=200)
    assert y_samples.shape == (9, 7, 200)
    if standardize:
        assert 5 < y_samples[:, 0].mean() < 15
    assert set(y_samples[:, 1:3].unique().tolist()) <= {0., 1.}
    assert set(y_samples[:, 3].unique().tolist()) <= {1., 2., 3.}
    assert torch.equal(y_samples[:, 4:].sum(dim=1), torch.ones(9, 200))
    assert math.isfinite(engressor.eval_loss(x, y, loss_type="energy"))
    engressor.summary()


def test_coding_of_labels():
    """Labels are returned in the coding of the training data, and the two codings give the same fit."""
    x, y = simulate("multilabel")
    engressors = []
    for y_train in [y, 2 * y - 1, y.bool(), y.long()]:
        torch.manual_seed(0)
        engressors.append(engression(x, y_train, data_type="multilabel", num_epochs=3, verbose=False))
    for engressor in engressors[1:]:
        for param, param_ref in zip(engressor.model.parameters(), engressors[0].model.parameters()):
            assert torch.equal(param, param_ref)
    samples, preds, losses = [], [], []
    for engressor, y_eval in zip(engressors, [y, 2 * y - 1, y.bool(), y.long()]):
        torch.manual_seed(1)
        samples.append(engressor.sample(x[:9], sample_size=20))
        torch.manual_seed(1)
        preds.append(engressor.predict(x[:9], target="mean"))
        torch.manual_seed(1)
        losses.append(engressor.eval_loss(x, y_eval, loss_type="energy"))
    assert set(samples[0].unique().tolist()) == {0., 1.} and set(samples[1].unique().tolist()) == {-1., 1.}
    assert torch.equal(samples[1], 2 * samples[0] - 1) and torch.equal(samples[2], samples[0]) and torch.equal(samples[3], samples[0])
    assert torch.allclose(preds[1], 2 * preds[0] - 1) and (preds[0] >= 0).all() and (preds[0] <= 1).all()
    # the distance between two labels is 1 in the 0/1 coding and 2 in the -1/+1 coding
    assert losses[1] == pytest.approx(2 * losses[0]) and losses[2] == pytest.approx(losses[0]) and losses[3] == pytest.approx(losses[0])
    assert engressors[0].tr_loss[0] == pytest.approx(engressors[1].tr_loss[0] / 2)


def test_integer_responses():
    for data_type in ["multiclass", "ordinal:5", "ranking"]:
        x, y = simulate(data_type)
        engressor = engression(x, y.long(), data_type=data_type, num_epochs=2, verbose=False)
        assert math.isfinite(engressor.eval_loss(x, y.long(), loss_type="energy"))
        assert engressor.sample(x[:9], sample_size=3).is_floating_point()


def test_engressor_class():
    x, y = simulate("mixed")
    engressor = Engressor(in_dim=2, out_dim=7, data_type=MIXED, num_layer=3, hidden_dim=32, noise_dim=7, add_bn=False,
                          lr=1e-3, check_device=False)
    engressor.train(x, y, num_epochs=3, verbose=False)
    engressor.train(x, y, num_epochs=2, batch_size=50, verbose=False)
    assert engressor.sample(x[:9], sample_size=5).shape == (9, 7, 5)
    data_type = [("multiclass", 0), ("multiclass", 2)]
    engressor = Engressor(in_dim=2, out_dim=4, data_type=data_type, check_device=False)
    y = torch.cat([torch.nn.functional.one_hot(torch.randint(0, 2, (400,)), 2) for _ in range(2)], dim=1)
    engressor.train(x, y, num_epochs=2, verbose=False)
    y_samples = engressor.sample(x[:9], sample_size=5)
    assert torch.equal(y_samples[:, :2].sum(dim=1), torch.ones(9, 5)) and torch.equal(y_samples[:, 2:].sum(dim=1), torch.ones(9, 5))


def test_defaults():
    """Without BN layers and with the noise dimension equal to the dimension of the response, unless the response is continuous."""
    x, y = simulate("multilabel")
    engressor = engression(x, y, data_type="multilabel", num_epochs=2, verbose=False)
    assert engressor.noise_dim == 4 and engressor.add_bn is False
    assert engressor.model.g_net.input_layer.layer[0].in_features == 2 + 4
    assert not any(isinstance(module, torch.nn.BatchNorm1d) for module in engressor.model.modules())
    engressor = engression(x, y, data_type="multilabel", noise_dim=20, add_bn=True, num_epochs=2, verbose=False)
    assert engressor.noise_dim == 20 and engressor.add_bn is True
    assert any(isinstance(module, torch.nn.BatchNorm1d) for module in engressor.model.modules())
    engressor = engression(x, y, num_epochs=2, verbose=False)
    assert engressor.noise_dim == 100 and engressor.add_bn is True
    assert any(isinstance(module, torch.nn.BatchNorm1d) for module in engressor.model.modules())


def test_default_learning_rate_and_epochs(monkeypatch):
    """A GEM is trained with the learning rate 0.001 for 2000 epochs, a continuous response with 0.0001 for 500 epochs, as before."""
    for data_type, lr, num_epochs in [("multilabel", 0.001, 2000), ({"continuous": 0, "ranking": 1}, 0.001, 2000),
                                      (None, 0.0001, 500), ("continuous", 0.0001, 500)]:
        engressor = Engressor(2, 4, data_type=data_type, check_device=False, verbose=False)
        assert engressor.lr == lr and engressor.num_epochs == num_epochs
        assert engressor.optimizer.param_groups[0]["lr"] == lr
    engressor = Engressor(2, 4, data_type="multilabel", lr=0.01, num_epochs=7, check_device=False, verbose=False)
    assert engressor.lr == 0.01 and engressor.num_epochs == 7 and engressor.optimizer.param_groups[0]["lr"] == 0.01
    monkeypatch.setattr(Engressor, "train", lambda self, *args, **kwargs: None)
    x, y = simulate("multilabel")
    engressor = engression(x, y, data_type="multilabel", verbose=False)
    assert engressor.lr == 0.001 and engressor.num_epochs == 2000
    engressor = engression(x, y, verbose=False)
    assert engressor.lr == 0.0001 and engressor.num_epochs == 500


def test_ordinal_starts_from_middle_level():
    engressor = Engressor(in_dim=2, out_dim=3, data_type={"continuous": 0, "ordinal:5": 1}, check_device=False)
    engressor.model.eval()
    y_samples = engressor.model(torch.randn(1000, 2))
    assert y_samples[:, 0].abs().mean() < 1
    assert 2.5 < y_samples[:, 1:].mean() < 3.5


def test_errors():
    x, y = simulate("multilabel")
    with pytest.raises(ValueError):
        engression(x, y, "multilabel", verbose=False)     # the third argument is `classification`
    with pytest.raises(ValueError):
        engression(x, y, data_type="multilabel", classification=True, verbose=False)
    with pytest.raises(ValueError):
        engression(x, y, data_type="multilabel", out_act="sigmoid", verbose=False)
    with pytest.raises(ValueError):
        engression(x, y, data_type="multilabel", sigma_dim="matrix", verbose=False)
    with pytest.raises(ValueError):
        engression(x, y, data_type="binary", verbose=False)
    with pytest.raises(ValueError):
        engression(x, y, data_type="ordinal", verbose=False)
    with pytest.raises(ValueError):
        engression(x, y, data_type="multiclass", verbose=False)     # rows are not indicator vectors
    with pytest.raises(ValueError):
        engression(x, y, data_type="ranking", verbose=False)
    with pytest.raises(ValueError):
        engression(x, y + 2, data_type="multilabel", verbose=False)
    with pytest.raises(ValueError):
        engression(x, y, data_type={"multilabel": 0, "multiclass": 4}, verbose=False)


@pytest.mark.parametrize("control_variate", [False, True])
def test_recovers_probabilities(control_variate):
    """Labels Y_j = 1{a_j x + e_j + u > 0} with a shared noise u, so that the labels are dependent given x."""
    torch.manual_seed(0)
    n, a = 2000, torch.tensor([1.5, -1., 0.])
    x = torch.rand(n, 1) * 4 - 2
    y = (a * x + torch.randn(n, 3) * 0.6 + torch.randn(n, 1) * 0.8 > 0).float()
    engressor = engression(x, y, data_type="multilabel", hidden_dim=50, noise_dim=3, add_bn=False,
                           control_variate=control_variate, lr=1e-3, num_epochs=600, verbose=False)
    x_eval = torch.linspace(-1.5, 1.5, 7).unsqueeze(1)
    y_samples = engressor.sample(x_eval, sample_size=4000)
    prob = 0.5 * (1 + torch.erf(a * x_eval / math.sqrt(2)))       # the noise e_j + u is standard Gaussian
    assert (y_samples.mean(dim=2) - prob).abs().max() < 0.1
    # P(Y_1 = Y_2 = 1 | x = 0) = 1/4 + arcsin(0.64) / (2 pi) = 0.36 for correlation 0.64 between the noises of two labels,
    # against 1/4 for independent labels
    both = (y_samples[3, 0] * y_samples[3, 1]).mean()
    assert abs(both - (0.25 + math.asin(0.64) / (2 * math.pi))) < 0.08


def test_sigma_min():
    """The scale of the perturbation is bounded below on the discrete coordinates only."""
    x, y = simulate("mixed")
    for sigma_dim in ["scalar", "vector"]:
        engressor = engression(x, y, data_type=MIXED, sigma_dim=sigma_dim, sigma_min=0.2, num_epochs=2, verbose=False)
        sigma = engressor.model.sigma(x)
        assert sigma.shape == (400, 7) and (sigma > 0).all()
        assert (sigma[:, 1:] > 0.2).all()
        if sigma_dim == "scalar":
            assert torch.allclose(sigma[:, 1:] - 0.2, sigma[:, :1].repeat(1, 6))


@pytest.mark.parametrize("data_type", list(VALUES) + ["mixed"])
def test_validation(data_type):
    """The kept parameters are those that a fit without validation data has after the same epoch."""
    x, y = simulate(data_type)
    x_val, y_val = simulate(data_type, n=200, seed=1)
    data_type = MIXED if data_type == "mixed" else data_type
    torch.manual_seed(0)
    engressor = engression(x, y, data_type=data_type, lr=1e-2, num_epochs=120, verbose=False, x_val=x_val, y_val=y_val)
    assert engressor.best_epoch in [50, 100, 120]
    torch.manual_seed(0)
    reference = engression(x, y, data_type=data_type, lr=1e-2, num_epochs=engressor.best_epoch, verbose=False).model.state_dict()
    for key, value in engressor.model.state_dict().items():
        assert torch.equal(value, reference[key]), key


def test_validation_coding():
    """The binary labels of the validation data are coded as those of the training data."""
    x, y = simulate("multilabel")
    x_val, y_val = simulate("multilabel", n=200, seed=1)
    with pytest.raises(ValueError, match="coded as those of the training data"):
        engression(x, y, data_type="multilabel", num_epochs=1, verbose=False, x_val=x_val, y_val=2 * y_val - 1)
    with pytest.raises(ValueError, match="coded as those of the training data"):
        engression(x, 2 * y - 1, data_type="multilabel", num_epochs=1, verbose=False, x_val=x_val, y_val=y_val)
    with pytest.raises(ValueError, match="all coded as 0/1 or all as -1/\\+1"):
        engression(x, y, data_type="multilabel", num_epochs=1, verbose=False, x_val=x_val, y_val=3 * y_val)
    engression(x, y.long(), data_type="multilabel", num_epochs=1, verbose=False, x_val=x_val, y_val=y_val.long())


def test_eval_loss_coding():
    """`eval_loss` checks that the response is coded as the training data. Labels coded as -1/+1 for a model trained on
    0/1 labels would otherwise give a wrong loss without an error."""
    x, y = simulate("multilabel")
    engressor = engression(x, y, data_type="multilabel", num_epochs=2, verbose=False)
    for loss_type in ["energy", "l2"]:
        engressor.eval_loss(x, y, loss_type=loss_type)
        with pytest.raises(ValueError, match="coded as those of the training data"):
            engressor.eval_loss(x, 2 * y - 1, loss_type=loss_type)
    with pytest.raises(ValueError, match="all coded as 0/1 or all as -1/\\+1"):
        engressor.eval_loss(x, 3 * y)
    x, y = simulate("multiclass")
    engressor = engression(x, y, data_type="multiclass", num_epochs=2, verbose=False)
    with pytest.raises(ValueError, match="indicator"):
        engressor.eval_loss(x, y.argmax(dim=1, keepdim=True).float().repeat(1, 4))
