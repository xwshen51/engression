import pytest
import torch

from engression.links import (Block, parse_data_type, link, link_block, is_continuous,
                              check_response, encode_response, decode_response)


def test_parse_single_type():
    assert parse_data_type(None, 3) == [Block("continuous", 0, 3, None)]
    assert parse_data_type("continuous", 3) == [Block("continuous", 0, 3, None)]
    assert parse_data_type("multilabel", 4) == [Block("multilabel", 0, 4, None)]
    assert parse_data_type("ordinal:5", 2) == [Block("ordinal", 0, 2, 5)]
    assert parse_data_type("ranking", 4) == [Block("ranking", 0, 4, None)]


def test_parse_mixed():
    blocks = parse_data_type({"continuous": 0, "multilabel": 7, "multiclass": 12}, 15)
    assert blocks == [Block("continuous", 0, 7, None), Block("multilabel", 7, 12, None), Block("multiclass", 12, 15, None)]
    # the order of the keys does not matter
    assert parse_data_type({"multiclass": 12, "continuous": 0, "multilabel": 7}, 15) == blocks
    # a data type that appears twice is given as a list
    blocks = parse_data_type([("multiclass", 0), ("multiclass", 3), ("ordinal:4", 5)], 6)
    assert blocks == [Block("multiclass", 0, 3, None), Block("multiclass", 3, 5, None), Block("ordinal", 5, 6, 4)]


@pytest.mark.parametrize("data_type, out_dim", [
    ("binary", 3),                                  # unknown name
    ("ordinal", 1),                                 # number of levels missing
    ("ordinal:1", 1),                               # fewer than two levels
    ("multilabel:3", 3),                            # argument not used
    ({"continuous": 1, "multilabel": 3}, 5),        # does not start at column 0
    ({"continuous": 0, "multilabel": 5}, 5),        # starts after the last column
    ([("continuous", 0), ("multilabel", 0)], 5),    # two blocks at the same column
    ({"continuous": 0, "multiclass": 4}, 5),        # multiclass block with one column
    ("ranking", 1),                                 # ranking of one item
    ({}, 3),
])
def test_parse_errors(data_type, out_dim):
    with pytest.raises(ValueError):
        parse_data_type(data_type, out_dim)


def test_links():
    z = torch.tensor([[0.3, -1.2, 2.6, 0.1],
                      [-0.7, 4.0, -3.0, 3.9]])
    assert torch.equal(link_block(z, "continuous"), z)
    assert torch.equal(link_block(z, "multilabel"), torch.tensor([[1., -1., 1., 1.], [-1., 1., -1., 1.]]))
    assert torch.equal(link_block(z, "multiclass"), torch.tensor([[0., 0., 1., 0.], [0., 1., 0., 0.]]))
    # rounding to the nearest level, clipped to 1, ..., 3
    assert torch.equal(link_block(z, "ordinal", 3), torch.tensor([[1., 1., 3., 1.], [1., 3., 1., 3.]]))
    assert torch.equal(link_block(torch.tensor([[1.49, 1.51, 2.49, 2.51]]), "ordinal", 3), torch.tensor([[1., 2., 2., 3.]]))
    # the i-th entry is the position of item i; position 1 is for the largest coordinate
    assert torch.equal(link_block(z, "ranking"), torch.tensor([[2., 4., 1., 3.], [3., 1., 4., 2.]]))


def test_link_mixed_and_gradient():
    blocks = parse_data_type({"continuous": 0, "multilabel": 2, "multiclass": 4, "ordinal:3": 7, "ranking": 8}, 11)
    z = torch.randn(20, 11, requires_grad=True)
    y = link(z, blocks)
    assert y.shape == z.shape
    assert torch.equal(y[:, :2], z[:, :2])
    assert set(y[:, 2:4].unique().tolist()) <= {-1., 1.}
    assert torch.equal(y[:, 4:7].sum(dim=1), torch.ones(20))
    assert set(y[:, 7].unique().tolist()) <= {1., 2., 3.}
    assert torch.equal(y[:, 8:].sort(dim=1).values, torch.tensor([1., 2., 3.]).repeat(20, 1))
    # gradients pass through the continuous columns only
    y.sum().backward()
    assert torch.equal(z.grad, is_continuous(blocks).float().repeat(20, 1))


def test_check_and_code_response():
    blocks = parse_data_type({"continuous": 0, "multilabel": 1, "multiclass": 3, "ordinal:4": 6, "ranking": 7}, 10)
    y = torch.tensor([[0.5, 0., 1., 0., 1., 0., 4., 1., 3., 2.],
                      [-2., 1., 1., 1., 0., 0., 1., 3., 2., 1.]])
    zero_one = check_response(y, blocks)
    assert zero_one.tolist() == [False, True, True] + [False] * 7
    y_coded = encode_response(y, zero_one)
    assert torch.equal(y_coded[:, 1:3], torch.tensor([[-1., 1.], [1., 1.]]))
    assert torch.equal(y_coded[:, [0, 3, 4, 5, 6, 7, 8, 9]], y[:, [0, 3, 4, 5, 6, 7, 8, 9]])
    assert torch.equal(decode_response(y_coded, zero_one), y)
    # samples of shape (data_size, response_dim, sample_size)
    y3 = y_coded.unsqueeze(2).repeat(1, 1, 5)
    assert torch.equal(decode_response(y3, zero_one), y.unsqueeze(2).repeat(1, 1, 5))
    # labels coded as -1/+1 are left as they are
    y_pm = y.clone()
    y_pm[:, 1:3] = y_coded[:, 1:3]
    zero_one = check_response(y_pm, blocks)
    assert not zero_one.any()
    assert torch.equal(encode_response(y_pm, zero_one), y_pm)


@pytest.mark.parametrize("column, value", [
    (1, 2.),      # a label that is neither 0/1 nor -1/+1
    (1, -1.),     # 0/1 and -1/+1 mixed in one block
    (3, 1.),      # two classes at once
    (4, 0.5),     # not an indicator
    (6, 0.),      # level below 1
    (6, 5.),      # level above L
    (6, 2.5),     # level that is not an integer
    (7, 2.),      # a tie in a ranking
    (8, 0.),      # positions must be 1, ..., k
    (1, float("nan")),
])
def test_check_response_errors(column, value):
    blocks = parse_data_type({"continuous": 0, "multilabel": 1, "multiclass": 3, "ordinal:4": 6, "ranking": 7}, 10)
    y = torch.tensor([[0.5, 0., 1., 0., 1., 0., 4., 1., 3., 2.],
                      [-2., 0., 1., 1., 0., 0., 1., 3., 2., 1.]])
    check_response(y, blocks)
    y[0, column] = value
    with pytest.raises(ValueError):
        check_response(y, blocks)


def test_check_response_dim():
    with pytest.raises(ValueError):
        check_response(torch.zeros(3, 4), parse_data_type("multilabel", 5))
