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
