import pytest
import torch

from engression import engression
from engression.data.loader import make_dataloader
from engression.loss_func import energy_loss_two_sample
from engression.models import StoNet


def simulate(n=300, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=generator)
    y = torch.cat([x[:, :1] ** 2, 2 * x[:, 1:2]], dim=1) + torch.randn(n, 2, generator=generator)
    return x, y


@pytest.mark.parametrize("num_layer", [2, 3])
@pytest.mark.parametrize("batch_size", [None, 50])
@pytest.mark.parametrize("add_bn", [True, False])
def test_training_unchanged(num_layer, batch_size, add_bn):
    """The fit is identical to that of the training procedure of version 0.1.15, written out below."""
    x, y = simulate()
    torch.manual_seed(0)
    engressor = engression(x, y, num_layer=num_layer, hidden_dim=32, noise_dim=8, add_bn=add_bn, lr=1e-3,
                           num_epochs=5, batch_size=batch_size, verbose=False)

    torch.manual_seed(0)
    model = StoNet(3, 2, num_layer, 32, 8, add_bn, None, False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    x_mean, x_std, y_mean, y_std = x.mean(dim=0), x.std(dim=0), y.mean(dim=0), y.std(dim=0)
    x, y = (x - x_mean) / x_std, (y - y_mean) / y_std
    batches = [(x, y)] if batch_size is None else make_dataloader(x, y, batch_size=batch_size, shuffle=True)
    for _ in range(5):
        for x_batch, y_batch in batches:
            model.zero_grad()
            y_sample1 = model(x_batch)
            y_sample2 = model(x_batch)
            loss, _, _ = energy_loss_two_sample(y_batch, y_sample1, y_sample2, beta=1, verbose=True)
            loss.backward()
            optimizer.step()

    assert torch.equal(engressor.y_mean, y_mean) and torch.equal(engressor.y_std, y_std)
    for param, param_ref in zip(engressor.model.parameters(), model.parameters()):
        assert torch.equal(param, param_ref)


def test_predict_and_sample():
    x, y = simulate()
    engressor = engression(x, y, num_epochs=5, verbose=False)
    assert engressor.predict(x[:7]).shape == (7, 2)
    assert [y_pred.shape for y_pred in engressor.predict(x[:7], target=[0.1, "median"])] == [(7, 2), (7, 2)]
    assert engressor.sample(x[:7], sample_size=10).shape == (7, 2, 10)
    assert engressor.sample(x[:7], sample_size=10, expand_dim=False).shape == (70, 2)
    assert isinstance(engressor.eval_loss(x, y, loss_type="energy"), float)


def test_classification_sample():
    """`sample` used to fail for models fitted with classification=True."""
    x, _ = simulate()
    y = torch.nn.functional.one_hot(torch.randint(0, 4, (300,)), 4).float()
    engressor = engression(x, y, classification=True, num_epochs=5, verbose=False)
    for sample_size in [1, 4, 10]:
        y_samples = engressor.sample(x[:7], sample_size=sample_size)
        assert y_samples.shape == ((7, 4, sample_size) if sample_size > 1 else (7, 4))
        assert torch.allclose(y_samples.sum(dim=1), torch.ones(1))
    assert engressor.predict(x[:7]).shape == (7, 4)


def test_sample_size_one(tmp_path):
    """With sample_size=1, `sample` drops the dimension of the samples; with expand_dim=False, there is no such dimension,
    and it used to drop that of a univariate response instead, so that plot(target="sample", sample_size=1) failed."""
    x, y = simulate()
    engressor = engression(x, y[:, :1], num_epochs=2, verbose=False)
    assert engressor.sample(x[:7], sample_size=1).shape == (7, 1)
    assert engressor.sample(x[:7], sample_size=1, expand_dim=False).shape == (7, 1)
    assert engressor.sample(x[:7], sample_size=3, expand_dim=False).shape == (21, 1)
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    engressor.plot(x[:50], y[:50, :1], target="sample", sample_size=1, save_dir=str(tmp_path / "samples.png"))
    assert (tmp_path / "samples.png").is_file()


def test_eval_loss_verbose():
    """eval_loss(verbose=True) used to fail for the losses other than the energy loss."""
    x, y = simulate()
    engressor = engression(x, y, num_epochs=5, verbose=False)
    for loss_type in ["l2", "l1", "cor"]:
        assert isinstance(engressor.eval_loss(x, y, loss_type=loss_type, verbose=True), float)
    loss = engressor.eval_loss(x, y, loss_type="energy", verbose=True)
    assert len(loss) == 3 and loss[0] == pytest.approx(loss[1] - loss[2] / 2)


@pytest.mark.parametrize("beta", [0.5, 1, 1.5])
def test_training_loss_with_beta(beta):
    """The training loss reported after fitting is the energy loss with the beta of the fit; it used beta = 1."""
    x, y = simulate()
    engressor = engression(x, y, beta=beta, num_epochs=5, verbose=False)
    torch.manual_seed(1)
    engressor.train(x, y, num_epochs=0, verbose=False)
    torch.manual_seed(1)
    assert engressor.tr_loss == pytest.approx(engressor.eval_loss(x, y, loss_type="energy", beta=beta, verbose=True), rel=1e-5)


def test_print_times_per_epoch(capsys):
    """Mini-batch training used to fail when print_times_per_epoch exceeded the number of batches minus one."""
    x, y = simulate(n=200)
    engression(x, y, num_epochs=2, batch_size=50, print_every_nepoch=1, print_times_per_epoch=4)
    assert capsys.readouterr().out.count("[Epoch 2 (50%), batch") == 4


@pytest.mark.parametrize("resblock", [False, True])
def test_batch_norm_last_batch_of_one(resblock):
    """With batch normalization, mini-batch training used to fail when the last batch had one observation."""
    x, y = simulate(n=201)
    engressor = engression(x, y, num_layer=4, resblock=resblock, num_epochs=2, batch_size=50, verbose=False)
    assert engressor.add_bn and all(torch.isfinite(torch.tensor(engressor.tr_loss)))


def test_plot_save(tmp_path):
    """plot(save_dir=...) used to make a folder at the path of the figure and then fail to save it; with training 
    data, it also failed for a one-dimensional response."""
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    x, y = simulate()
    engressor = engression(x, y[:, :1], num_epochs=2, verbose=False)
    path = tmp_path / "figures" / "fit.png"
    engressor.plot(x[:50], y[:50, 0], x_tr=x[50:100], y_tr=y[50:100, 0], save_dir=str(path))
    assert path.is_file()
    engressor.plot(x[:50], y[:50, :1], target="sample", sample_size=2, save_dir=str(tmp_path / "samples.png"))
    assert (tmp_path / "samples.png").is_file()


def test_data_with_more_dimensions():
    """engression() used to build the network for the size of the second dimension of x and y, whereas Engressor.train
    flattens all dimensions after the first, so that the network had the wrong numbers of inputs and outputs."""
    x, y = simulate()
    x3, y3 = x.repeat(1, 2).view(-1, 2, 3), y.view(-1, 1, 2)
    engressor = engression(x3, y3, num_epochs=2, verbose=False)
    assert (engressor.model.in_dim, engressor.model.out_dim) == (6, 2)
    assert engressor.predict(x3[:7]).shape == (7, 2)


def test_one_dimensional_data():
    """engression() used to fail for a one-dimensional x or y, which Engressor.train accepts."""
    x, y = simulate()
    engressor = engression(x[:, 0], y[:, 0], num_epochs=2, verbose=False)
    assert engressor.predict(x[:7, 0]).shape == (7, 1)
    assert engressor.sample(x[:7, 0], sample_size=3).shape == (7, 1, 3)
