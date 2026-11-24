import torch

from torch_geometric.nn import DCQConv


def test_dcq_conv_forward():
    x = torch.randn(10, 16)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7],
    ])
    value = torch.rand(edge_index.size(1))

    conv = DCQConv(16, 8)
    assert str(conv) == "DCQConv(16, 8, opt=['cvx', 'cvx'])"

    out = conv(x, edge_index)
    assert out.size() == (10, 8)
    assert torch.isfinite(out).all()

    out = conv(x, edge_index, value)
    assert out.size() == (10, 8)
    assert torch.isfinite(out).all()


def test_dcq_conv_concave():
    x = torch.randn(10, 16)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7],
    ])

    conv = DCQConv(16, 8, opt=['ccv', 'ccv'])
    assert str(conv) == "DCQConv(16, 8, opt=['ccv', 'ccv'])"

    out = conv(x, edge_index)
    assert out.size() == (10, 8)
    assert torch.isfinite(out).all()


def test_dcq_conv_permutation_equivariance():
    x = torch.randn(10, 16)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 9],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 0, 2],
    ])

    conv = DCQConv(16, 8)
    conv.eval()  # Avoid BatchNorm running-statistics updates.

    out1 = conv(x, edge_index)
    assert out1.size() == (10, 8)

    # Permute the node ordering (relabeled edges + permuted features).
    perm = torch.randperm(10)
    x2 = torch.empty_like(x)
    x2[perm] = x
    edge_index2 = perm[edge_index]
    out2 = conv(x2, edge_index2)

    assert torch.allclose(out1[perm], out2, atol=1e-5)


def test_dcq_conv_gradients():
    x = torch.randn(10, 16)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7],
    ])

    conv = DCQConv(16, 8)
    out = conv(x, edge_index)
    out.sum().backward()

    assert conv.lambda_t.grad is not None
    assert conv.beta.grad is not None
    assert conv.gates[0].weight.grad is not None
    assert conv.gates[3].weight.grad is not None
    assert conv.lin.weight.grad is not None


def test_dcq_conv_clamp():
    x = torch.randn(10, 16)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7],
    ])

    conv = DCQConv(16, 8)
    with torch.no_grad():
        conv.lambda_t.copy_(torch.tensor([5.0, -1.0, 0.0]))
        conv.beta.copy_(torch.tensor([2.0, 0.001, 0.5, 1.0]))

    conv(x, edge_index)

    assert torch.all(conv.lambda_t >= 0.0) and torch.all(conv.lambda_t <= 2.0)
    assert torch.all(conv.beta >= 0.01) and torch.all(conv.beta <= 1.0)
    assert torch.allclose(conv.lambda_t, torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(conv.beta, torch.tensor([1.0, 0.01, 0.5, 1.0]))
