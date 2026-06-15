r"""Structure-aware subgraph-root selection via graph coarsening.

Subgraph GNNs represent a graph as a *bag of subgraphs* -- typically one rooted
subgraph per node -- which is what :class:`~torch_geometric.transforms.\
RootedEgoNets` produces. Processing one subgraph per node is the main
scalability bottleneck of these methods.

This module implements a coarsening-guided strategy for choosing *which* roots
(and therefore which subgraphs) to keep, delivering the core result of
`"A Flexible, Equivariant Framework for Subgraph GNNs via Graph Products and
Graph Coarsening" <https://arxiv.org/abs/2406.09291>`_: rather than sampling
roots uniformly at random -- which the paper shows yields suboptimal,
structurally redundant selections -- the graph is coarsened into a small set
of clusters and one representative root is drawn from each. The resulting roots
spread across the graph's structure, so a reduced bag retains coverage that
random sampling of the same size would miss.

The full paper additionally builds an equivariant architecture over graph
products; that learned machinery is intentionally out of scope here. What is
reused is the selection result -- coarsening as a deterministic, parameter-free
way to pick a structurally diverse subset of subgraph roots -- which slots
directly into PyG's existing rooted-subgraph extraction.
"""
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.utils import coalesce, degree


def _cluster_adjacency(
    edge_index: Tensor,
    cluster: Tensor,
    num_clusters: int,
) -> Tuple[Tensor, Tensor]:
    # Lifts the node-level edges to weighted edges between the current
    # clusters, dropping intra-cluster edges and summing parallel edges into a
    # single weight (the number of node-pairs connecting the two clusters).
    c_row, c_col = cluster[edge_index[0]], cluster[edge_index[1]]
    mask = c_row != c_col
    c_edge = torch.stack([c_row[mask], c_col[mask]], dim=0)
    weight = torch.ones(c_edge.size(1), device=edge_index.device)
    return coalesce(c_edge, weight, num_nodes=num_clusters, reduce='sum')


def _match_clusters(
    c_edge: Tensor,
    weight: Tensor,
    num_clusters: int,
    order: List[int],
) -> Tensor:
    # Greedy heavy-edge matching: visiting clusters in ``order``, each
    # unmatched cluster is merged with its heaviest unmatched neighbor.
    # Returns a mapping from old cluster id to a contiguous new cluster id.
    neighbors: Dict[int, List[Tuple[float, int]]] = {
        i: []
        for i in range(num_clusters)
    }
    src, dst = c_edge.tolist()
    w = weight.tolist()
    for s, d, wt in zip(src, dst, w):
        neighbors[s].append((wt, d))

    matched = [-1] * num_clusters
    new_id = 0
    for node in order:
        if matched[node] != -1:
            continue
        best, best_w = -1, -1.0
        for wt, nbr in neighbors[node]:
            if matched[nbr] == -1 and nbr != node and wt > best_w:
                best, best_w = nbr, wt
        matched[node] = new_id
        if best != -1:
            matched[best] = new_id
        new_id += 1

    return torch.tensor(matched, dtype=torch.long)


def select_coarsened_roots(
    edge_index: Tensor,
    num_nodes: int,
    ratio: float = 0.5,
    num_roots: Optional[int] = None,
) -> Tensor:
    r"""Selects a structurally diverse subset of subgraph roots via graph
    coarsening.

    The graph is repeatedly coarsened by heavy-edge matching until the number
    of clusters drops to the requested budget; one representative root (the
    highest-degree original node, ties broken by lowest index) is then returned
    per cluster. The selection is deterministic and dependency-free.

    Args:
        edge_index (torch.Tensor): The edge indices.
        num_nodes (int): The number of nodes in the graph.
        ratio (float, optional): The fraction of nodes to keep as roots, used
            when :obj:`num_roots` is not given. (default: :obj:`0.5`)
        num_roots (int, optional): The exact number of roots to select. Takes
            precedence over :obj:`ratio` when provided. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor` -- the sorted indices of the selected roots.
    """
    if num_roots is not None:
        target = num_roots
    else:
        target = int(round(ratio * num_nodes))
    target = min(max(target, 1), num_nodes)

    deg = degree(edge_index[0], num_nodes=num_nodes)

    cluster = torch.arange(num_nodes, device=edge_index.device)
    num_clusters = num_nodes
    while num_clusters > target:
        c_edge, weight = _cluster_adjacency(edge_index, cluster, num_clusters)
        if c_edge.numel() == 0:
            break  # Fully disconnected clusters: no further contraction.

        # Match low-degree clusters first so that hubs do not absorb the graph
        # in a single pass, keeping the coarsening balanced.
        c_deg = degree(c_edge[0], num_nodes=num_clusters)
        order = torch.argsort(c_deg, stable=True).tolist()

        relabel = _match_clusters(c_edge, weight, num_clusters, order)
        relabel = relabel.to(cluster.device)
        new_cluster = relabel[cluster]
        new_num = int(new_cluster.max()) + 1
        if new_num == num_clusters:
            break  # No pair could be matched; avoid an infinite loop.
        cluster, num_clusters = new_cluster, new_num

    return _representatives(cluster, num_clusters, deg)


def _representatives(cluster: Tensor, num_clusters: int,
                     deg: Tensor) -> Tensor:
    # Picks one root per cluster: the node with the highest degree, breaking
    # ties towards the lowest node index for determinism.
    num_nodes = cluster.size(0)
    index = torch.arange(num_nodes, device=cluster.device)
    # Score so that a strict argmax recovers (max degree, then min index).
    score = deg * num_nodes - index.to(deg.dtype)
    best = torch.full((num_clusters, ), -1, dtype=torch.long,
                      device=cluster.device)
    best_score = torch.full((num_clusters, ), float('-inf'), device=deg.device)
    for node in range(num_nodes):
        c = int(cluster[node])
        if score[node] > best_score[c]:
            best_score[c] = score[node]
            best[c] = node
    return torch.sort(best).values
