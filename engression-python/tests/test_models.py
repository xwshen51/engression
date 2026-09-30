import pytest
import torch
import torch.nn as nn

from engression.models import StoLayer, StoResBlock, StoNet, CondStoNet, Net, ResMLP


def shared_parameters(model):
    """Groups of names under which one parameter appears more than once in the model."""
    names = {}
    for name, param in model.state_dict(keep_vars=True).items():
        if isinstance(param, nn.Parameter):
            names.setdefault(id(param), []).append(name)
    return [group for group in names.values() if len(group) > 1]


@pytest.mark.parametrize("num_layer", range(1, 9))
@pytest.mark.parametrize("resblock", [False, True])
@pytest.mark.parametrize("add_bn", [False, True])
@pytest.mark.parametrize("noise_all_layer", [True, False])
def test_no_shared_parameters(num_layer, resblock, add_bn, noise_all_layer):
    model = StoNet(3, 2, num_layer=num_layer, hidden_dim=16, noise_dim=4, add_bn=add_bn, resblock=resblock,
                   noise_all_layer=noise_all_layer)
    assert shared_parameters(model) == []
    if not resblock:
        # input layer, num_layer - 2 intermediate layers and output layer
        n_bn = 2 * 16 if add_bn else 0
        noise_dim = 4 if noise_all_layer else 0
        n_param = (3 + 4) * 16 + 16 + n_bn + max(num_layer - 2, 0) * ((16 + noise_dim) * 16 + 16 + n_bn) + 16 * 2 + 2
        assert sum(param.numel() for param in model.parameters()) == n_param


@pytest.mark.parametrize("num_layer", range(2, 9))
def test_no_shared_parameters_other_networks(num_layer):
    assert shared_parameters(ResMLP(3, 2, num_layer=num_layer, hidden_dim=16)) == []
    assert shared_parameters(Net(3, 2, num_layer=num_layer, hidden_dim=16)) == []
    assert shared_parameters(CondStoNet(3, 2, condition_dim=2, num_layer=num_layer, hidden_dim=16, noise_dim=4)) == []


@pytest.mark.parametrize("num_layer", [1, 2, 3])
@pytest.mark.parametrize("add_bn", [False, True])
def test_initialization_as_before(num_layer, add_bn):
    """With at most one intermediate layer, the parameters are initialized as in version 0.1.15, which made an
    intermediate layer also when none was used, so that fits with a fixed seed are unchanged."""
    torch.manual_seed(0)
    model = StoNet(3, 2, num_layer=num_layer, hidden_dim=16, noise_dim=4, add_bn=add_bn)
    torch.manual_seed(0)
    layers = [StoLayer(3, 16, 4, add_bn, "relu"), StoLayer(16, 16, 4, add_bn, "relu"), nn.Linear(16, 2)]
    if num_layer < 3:
        layers.pop(1)
    params = [param for layer in layers for param in layer.parameters()]
    assert len(params) == len(list(model.parameters()))
    assert all(torch.equal(param, param_ref) for param, param_ref in zip(model.parameters(), params))


@pytest.mark.parametrize("num_layer", [4, 6])
def test_initialization_as_before_resblock(num_layer):
    torch.manual_seed(0)
    model = StoNet(3, 2, num_layer=num_layer, hidden_dim=16, noise_dim=4, resblock=True)
    torch.manual_seed(0)
    blocks = [StoResBlock(dim=3, hidden_dim=16, out_dim=16, noise_dim=4, out_act="relu"),
              StoResBlock(dim=16, noise_dim=4, out_act="relu"),
              StoResBlock(dim=16, hidden_dim=16, out_dim=2, noise_dim=4)]
    if num_layer == 4:
        blocks.pop(1)
    params = [param for block in blocks for param in block.parameters()]
    assert len(params) == len(list(model.parameters()))
    assert all(torch.equal(param, param_ref) for param, param_ref in zip(model.parameters(), params))


def test_weights_of_earlier_versions_load():
    """Weights saved by versions up to 0.1.15, whose intermediate layers shared one set of weights, still load."""
    model = StoNet(3, 2, num_layer=5, hidden_dim=16, noise_dim=4)
    model_tied = StoNet(3, 2, num_layer=5, hidden_dim=16, noise_dim=4)
    model_tied.inter_layer = nn.Sequential(*[model_tied.inter_layer[0]] * 3)    # the construction of 0.1.15
    model.load_state_dict(model_tied.state_dict())
    x = torch.randn(10, 3)
    torch.manual_seed(1)
    y = model(x)
    torch.manual_seed(1)
    assert torch.equal(y, model_tied(x))

