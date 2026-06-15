import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.testing import withPackage
from torch_geometric.transforms import RootedEgoNets, RootedRWSubgraph
from torch_geometric.transforms.coarsened_subgraph_selection import (
    select_coarsened_roots,
)


def test_rooted_ego_nets():
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.randn(4, 8)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    transform = RootedEgoNets(num_hops=1)
    assert str(transform) == 'RootedEgoNets(num_hops=1)'

    out = transform(data)
    assert len(out) == 8

    assert torch.equal(out.x, data.x)
    assert torch.equal(out.edge_index, data.edge_index)
    assert torch.equal(out.edge_attr, data.edge_attr)

    assert out.sub_edge_index.tolist() == [[0, 1, 2, 3, 3, 4, 5, 6],
                                           [1, 0, 3, 2, 4, 3, 6, 5]]
    assert out.n_id.tolist() == [0, 1, 0, 1, 2, 1, 2]
    assert out.n_sub_batch.tolist() == [0, 0, 1, 1, 1, 2, 2]
    assert out.e_id.tolist() == [0, 1, 0, 1, 2, 3, 2, 3]
    assert out.e_sub_batch.tolist() == [0, 0, 1, 1, 1, 1, 2, 2]

    out = out.map_data()
    assert len(out) == 4

    assert torch.allclose(out.x, x[[0, 1, 0, 1, 2, 1, 2]])
    assert out.edge_index.tolist() == [[0, 1, 2, 3, 3, 4, 5, 6],
                                       [1, 0, 3, 2, 4, 3, 6, 5]]
    assert torch.allclose(out.edge_attr, edge_attr[[0, 1, 0, 1, 2, 3, 2, 3]])
    assert out.n_sub_batch.tolist() == [0, 0, 1, 1, 1, 2, 2]


def test_select_coarsened_roots():
    # A 6-node path: coarsening should keep a spread-out subset of roots.
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])

    roots = select_coarsened_roots(edge_index, num_nodes=6, ratio=0.5)
    assert roots.dim() == 1
    assert len(roots) == 3
    # Roots are valid, unique and returned in sorted order.
    assert roots.min() >= 0 and roots.max() < 6
    assert torch.equal(roots, roots.unique())
    assert torch.equal(roots, roots.sort().values)

    # `num_roots` takes precedence over `ratio`.
    roots = select_coarsened_roots(edge_index, num_nodes=6, num_roots=2)
    assert len(roots) == 2

    # Deterministic across calls.
    again = select_coarsened_roots(edge_index, num_nodes=6, num_roots=2)
    assert torch.equal(roots, again)


def test_rooted_ego_nets_coarsened_selection():
    x = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])
    data = Data(x=x, edge_index=edge_index)

    transform = RootedEgoNets(num_hops=1, root_ratio=0.5)
    assert str(transform) == 'RootedEgoNets(num_hops=1, root_ratio=0.5)'

    out = transform(data)
    num_subgraphs = int(out.n_sub_batch.max()) + 1

    # The bag holds one subgraph per selected root, fewer than one-per-node.
    full = RootedEgoNets(num_hops=1)(data)
    full_subgraphs = int(full.n_sub_batch.max()) + 1
    assert full_subgraphs == 6
    assert num_subgraphs == 3
    assert num_subgraphs < full_subgraphs

    # The reduced bag still maps cleanly back to a batched subgraph `Data`.
    mapped = out.map_data()
    assert mapped.num_nodes == out.n_id.size(0)
    assert mapped.edge_index.size(0) == 2


@withPackage('torch_cluster')
def test_rooted_rw_subgraph():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(edge_index=edge_index, num_nodes=3)

    transform = RootedRWSubgraph(walk_length=1)
    assert str(transform) == 'RootedRWSubgraph(walk_length=1)'

    out = transform(data)
    assert len(out) == 7

    assert out.n_sub_batch.tolist() == [0, 0, 1, 1, 2, 2]
    assert out.sub_edge_index.size() == (2, 6)

    out = out.map_data()
    assert len(out) == 3

    assert out.edge_index.size() == (2, 6)
    assert out.num_nodes == 6
    assert out.n_sub_batch.tolist() == [0, 0, 1, 1, 2, 2]


def test_rooted_subgraph_minibatch():
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.randn(4, 8)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    transform = RootedEgoNets(num_hops=1)
    data = transform(data)

    loader = DataLoader([data, data], batch_size=2)
    batch = next(iter(loader))
    batch = batch.map_data()
    assert batch.num_graphs == len(batch) == 2

    assert batch.x.size() == (14, 8)
    assert batch.edge_index.size() == (2, 16)
    assert batch.edge_attr.size() == (16, 8)
    assert batch.n_sub_batch.size() == (14, )
    assert batch.batch.size() == (14, )
    assert batch.ptr.size() == (3, )

    assert batch.edge_index.min() == 0
    assert batch.edge_index.max() == 13

    assert batch.n_sub_batch.min() == 0
    assert batch.n_sub_batch.max() == 5
