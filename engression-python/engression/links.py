from collections import namedtuple

import torch


DATA_TYPES = ["continuous", "multilabel", "multiclass", "ordinal", "ranking"]

Block = namedtuple("Block", ["name", "start", "end", "num_level"])


def parse_data_type(data_type, out_dim):
    """Parse the data type of the response into blocks of columns.

    Args:
        data_type (str, dict or list): data type of the response.
            - None or str: one data type for all columns. Choices: ["continuous", "multilabel", "multiclass", "ordinal:L", "ranking"], where L is the number of levels of an ordinal response.
            - dict: a mixed response, with the data types as keys and the first column of each block as values, e.g. {"continuous": 0, "multilabel": 7, "multiclass": 12}. Each block ends where the next one starts.
            - list: the same as (data type, first column) pairs, e.g. [("multiclass", 0), ("multiclass", 3)], for a data type that appears more than once.
        out_dim (int): number of columns of the response.

    Returns:
        list of Block: blocks (name, start, end, num_level) ordered by their first column; the block consists of the columns start, ..., end - 1.
    """
    if data_type is None:
        data_type = "continuous"
    if isinstance(data_type, str):
        items = [(data_type, 0)]
    elif isinstance(data_type, dict):
        items = list(data_type.items())
    else:
        items = [tuple(item) for item in data_type]
    if len(items) == 0:
        raise ValueError("`data_type` is empty.")
    items = sorted(items, key=lambda item: item[1])
    starts = [int(start) for _, start in items]
    if starts[0] != 0:
        raise ValueError("The first block of `data_type` must start at column 0, but it starts at column {}.".format(starts[0]))
    if len(set(starts)) < len(starts):
        raise ValueError("Two blocks of `data_type` start at the same column: {}.".format(starts))
    if starts[-1] >= out_dim:
        raise ValueError("A block of `data_type` starts at column {}, but the response has only {} columns.".format(starts[-1], out_dim))

    blocks = []
    for i, (spec, start) in enumerate(items):
        start = starts[i]
        end = starts[i + 1] if i + 1 < len(items) else out_dim
        name, _, num_level = str(spec).partition(":")
        if name not in DATA_TYPES:
            raise ValueError("Unknown data type '{}'. Choices: {}.".format(spec, DATA_TYPES))
        if name == "ordinal":
            if not num_level.isdigit() or int(num_level) < 2:
                raise ValueError("An ordinal response needs its number of levels, e.g. 'ordinal:5' for the levels 1, ..., 5.")
            num_level = int(num_level)
        elif num_level != "":
            raise ValueError("Data type '{}' takes no argument; got '{}'.".format(name, spec))
        else:
            num_level = None
        if name in ["multiclass", "ranking"] and end - start < 2:
            raise ValueError("A '{}' block needs at least two columns, but the one at column {} has {}.".format(name, start, end - start))
        blocks.append(Block(name, start, end, num_level))
    return blocks


def link_block(z, name, num_level=None):
    """Link of one data type, which maps latent vectors to outcomes.

    Args:
        z (torch.Tensor): latent vectors of shape (data_size, block_dim).
        name (str): data type. Choices: ["continuous", "multilabel", "multiclass", "ordinal", "ranking"].
        num_level (int, optional): number of levels, for an ordinal response. Defaults to None.

    Returns:
        torch.Tensor: outcomes of shape (data_size, block_dim).
            - continuous: z itself.
            - multilabel: the sign of z, in {-1, +1}.
            - multiclass: the indicator (one-hot) vector of the largest coordinate of z.
            - ordinal: z rounded to the nearest level in {1, ..., num_level}.
            - ranking: the rank vector of z, whose i-th entry is the position of item i, where position 1 is for the largest coordinate.
    """
    if name == "continuous":
        return z
    elif name == "multilabel":
        return (z > 0).to(z.dtype) * 2 - 1
    elif name == "multiclass":
        return torch.zeros_like(z).scatter_(1, z.argmax(dim=1, keepdim=True), 1)
    elif name == "ordinal":
        cuts = torch.arange(1, num_level, device=z.device, dtype=z.dtype) + 0.5
        return 1 + (z.unsqueeze(2) > cuts).sum(dim=2).to(z.dtype)
    elif name == "ranking":
        order = z.argsort(dim=1, descending=True)
        positions = torch.arange(1, z.size(1) + 1, device=z.device, dtype=z.dtype).expand_as(z)
        return torch.zeros_like(z).scatter_(1, order, positions)
    else:
        raise ValueError("Unknown data type '{}'. Choices: {}.".format(name, DATA_TYPES))


def link(z, blocks):
    """Apply to each block of the latent vectors the link of its data type.

    Args:
        z (torch.Tensor): latent vectors of shape (data_size, response_dim).
        blocks (list of Block): blocks of the response, see `parse_data_type`.

    Returns:
        torch.Tensor: outcomes of shape (data_size, response_dim). Gradients pass through the continuous blocks only.
    """
    out = [link_block(z[:, block.start:block.end], block.name, block.num_level) for block in blocks]
    return out[0] if len(out) == 1 else torch.cat(out, dim=1)


def is_continuous(blocks, device=None):
    """Indicate the continuous columns of the response.

    Args:
        blocks (list of Block): blocks of the response.
        device (str or torch.device, optional): device. Defaults to None.

    Returns:
        torch.Tensor: boolean tensor of shape (response_dim,).
    """
    mask = torch.zeros(blocks[-1].end, dtype=torch.bool, device=device)
    for block in blocks:
        if block.name == "continuous":
            mask[block.start:block.end] = True
    return mask


def check_response(y, blocks):
    """Check that the response is coded as its data types require.

    Coding of the response:
        - continuous: real values.
        - multilabel: binary labels, all coded as 0/1 or all as -1/+1 within a block.
        - multiclass: indicator (one-hot) vectors.
        - ordinal: levels 1, ..., L.
        - ranking: rank vectors, whose i-th entry is the position of item i, that is, permutations of 1, ..., k.

    Args:
        y (torch.Tensor): data of responses of shape (data_size, response_dim).
        blocks (list of Block): blocks of the response.

    Returns:
        torch.Tensor: boolean tensor of shape (response_dim,) that indicates the binary columns coded as 0/1.
    """
    if y.size(1) != blocks[-1].end:
        raise ValueError("The response has {} columns, but the model was specified with {}.".format(y.size(1), blocks[-1].end))
    zero_one = torch.zeros(y.size(1), dtype=torch.bool, device=y.device)
    for block in blocks:
        y_block = y[:, block.start:block.end]
        where = "the '{}' block (columns {} to {})".format(block.name, block.start, block.end - 1)
        if block.name == "multilabel":
            if ((y_block == 0) | (y_block == 1)).all():
                zero_one[block.start:block.end] = True
            elif not ((y_block == -1) | (y_block == 1)).all():
                raise ValueError("The labels of {} must be all coded as 0/1 or all as -1/+1.".format(where))
        elif block.name == "multiclass":
            if not (((y_block == 0) | (y_block == 1)).all() and (y_block.sum(dim=1) == 1).all()):
                raise ValueError("The rows of {} must be indicator (one-hot) vectors, ".format(where) +
                                 "e.g. torch.nn.functional.one_hot(labels, num_classes).")
        elif block.name == "ordinal":
            if not ((y_block == y_block.round()).all() and y_block.min() >= 1 and y_block.max() <= block.num_level):
                raise ValueError("The levels of {} must be coded as 1, ..., {}.".format(where, block.num_level))
        elif block.name == "ranking":
            positions = torch.arange(1, y_block.size(1) + 1, device=y.device, dtype=y.dtype)
            if not (y_block.sort(dim=1).values == positions).all():
                raise ValueError("The rows of {} must be rank vectors, that is, permutations of 1, ..., {} ".format(where, y_block.size(1)) +
                                 "whose i-th entry is the position of item i.")
    return zero_one


def encode_response(y, zero_one):
    """Code the binary columns of the response as -1/+1, the coding used in training.

    Args:
        y (torch.Tensor): data of responses of shape (data_size, response_dim).
        zero_one (torch.Tensor): boolean tensor of shape (response_dim,) that indicates the binary columns coded as 0/1.

    Returns:
        torch.Tensor: responses of the same shape, with all binary columns coded as -1/+1.
    """
    return torch.where(zero_one, 2 * y - 1, y)


def decode_response(y, zero_one):
    """Transform the binary columns back to the coding of the training data; the inverse of `encode_response`.

    Args:
        y (torch.Tensor): responses or predictions of shape (data_size, response_dim) or (data_size, response_dim, sample_size).
        zero_one (torch.Tensor): boolean tensor of shape (response_dim,) that indicates the binary columns coded as 0/1.

    Returns:
        torch.Tensor: responses of the same shape.
    """
    if len(y.shape) == 3:
        zero_one = zero_one.unsqueeze(0).unsqueeze(2)
    return torch.where(zero_one, (y + 1) / 2, y)
