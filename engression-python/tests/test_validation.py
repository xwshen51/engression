import pytest
import torch

from engression import engression
from engression.engression import Engressor


def simulate(n, seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=generator)
    y = torch.cat([x[:, :1] ** 2, 2 * x[:, 1:2]], dim=1) + torch.randn(n, 2, generator=generator)
    return x, y


def fit(num_epochs, batch_size=None, x_val=None, y_val=None):
    x, y = simulate(100, seed=0)
    torch.manual_seed(0)
    engressor = Engressor(3, 2, hidden_dim=16, noise_dim=4, lr=1e-2, batch_size=batch_size, verbose=False)
    engressor.train(x, y, num_epochs=num_epochs, verbose=False, x_val=x_val, y_val=y_val)
    return engressor


@pytest.mark.parametrize("batch_size, num_epochs, iterations", [
    (None, 120, [50, 100, 120]),   # full batch: one iteration per epoch
    (20, 23, [50, 100, 115]),      # five iterations per epoch
    (30, 30, [52, 100, 120]),      # four iterations per epoch
])
@pytest.mark.parametrize("kept", [0, -1])
def test_validation_keeps_the_best_parameters(monkeypatch, batch_size, num_epochs, iterations, kept):
    """The validation loss is computed at the end of each epoch in which the number of iterations reaches a multiple of 50
    and at the end of the last epoch, and the parameters with the lowest loss are kept. They are the parameters that a fit
    without validation data has after that epoch, since validation does not change the random numbers of training."""
    values = [1.0] * len(iterations)
    values[kept] = 0.0
    record = []
    validation_loss = Engressor._validation_loss

    def scripted_validation_loss(self, x_val, y_val):
        validation_loss(self, x_val, y_val)   # draws random numbers as usual
        record.append(int(next(iter(self.optimizer.state.values()))["step"]))
        return values[len(record) - 1]

    monkeypatch.setattr(Engressor, "_validation_loss", scripted_validation_loss)
    engressor = fit(num_epochs, batch_size, *simulate(50, seed=1))
    assert record == iterations
    best_epoch = num_epochs * iterations[kept] // iterations[-1]
    assert engressor.best_epoch == best_epoch
    reference = fit(best_epoch, batch_size).model.state_dict()
    for key, value in engressor.model.state_dict().items():
        assert torch.equal(value, reference[key]), key


def test_validation_against_overfitting():
    """With few observations and many epochs, the network overfits, and the kept parameters fit new data better than the last ones."""
    x, y = simulate(20, seed=102)
    x_val, y_val = simulate(500, seed=1)
    x_test, y_test = simulate(5000, seed=2)
    losses = []
    for validation in [False, True]:
        torch.manual_seed(0)
        engressor = engression(x, y, hidden_dim=64, add_bn=False, lr=1e-2, num_epochs=2000, verbose=False,
                               x_val=x_val if validation else None, y_val=y_val if validation else None)
        torch.manual_seed(1)
        losses.append(engressor.eval_loss(x_test, y_test, loss_type="energy", sample_size=20))
    assert engressor.best_epoch < 1000
    assert losses[1] < losses[0] - 0.2


def test_validation_summary(capsys):
    x, y = simulate(100, seed=0)
    x_val, y_val = simulate(50, seed=1)
    engressor = engression(x, y, num_epochs=60, x_val=x_val, y_val=y_val)
    assert engressor.best_epoch in [50, 60]
    message = "parameters after epoch {} gave the lowest energy loss on the validation data".format(engressor.best_epoch)
    assert message in capsys.readouterr().out
    engressor.summary()
    assert "parameters kept from epoch {}".format(engressor.best_epoch) in capsys.readouterr().out
    assert not hasattr(engressor, "_best_state")
    # Training again without validation data keeps the last parameters.
    engressor.train(x, y, num_epochs=1, verbose=False)
    assert engressor.best_epoch is None
    # A model saved by an earlier version has no such attribute.
    del engressor.__dict__["best_epoch"]
    engressor.summary()
    assert "Validation" not in capsys.readouterr().out


def test_validation_data_errors():
    x, y = simulate(100, seed=0)
    with pytest.raises(ValueError, match="both"):
        engression(x, y, num_epochs=1, verbose=False, x_val=x)
    with pytest.raises(ValueError, match="sample sizes"):
        engression(x, y, num_epochs=1, verbose=False, x_val=x, y_val=y[:50])
    with pytest.raises(ValueError, match="columns"):
        engression(x, y, num_epochs=1, verbose=False, x_val=x[:, :2], y_val=y)
