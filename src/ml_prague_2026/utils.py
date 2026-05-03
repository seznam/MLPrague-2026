from __future__ import annotations

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from torch_geometric.utils import degree as pyg_degree


def prepare_yelp_chi_tabular_data(yelp_chi, train_split, test_split, add_degree_feature: bool = True, edge_paths=None):
    """Prepare train/test features and labels for Yelp-Chi in tabular form.

    Args:
        yelp_chi: DataFrame with node features (f_*) and 'spam' label.
        add_degree_feature: If True, append degree features from RUR/RSR/RTR edges.

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Prepare features and labels
    feature_cols = [c for c in yelp_chi.columns if c.startswith('f_')]
    X = yelp_chi[feature_cols].values
    y_true = yelp_chi['spam'].values

    if add_degree_feature and edge_paths:
        def undirected_degree(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
            """Sum degrees from both directions to handle undirected edges."""
            return (
                pyg_degree(edge_index[0], num_nodes=num_nodes)
                + pyg_degree(edge_index[1], num_nodes=num_nodes)
            ).numpy()

        n_nodes = len(yelp_chi)

        degree_features = []
        for path in edge_paths.values():
            edgs = torch.tensor(np.load(path, allow_pickle=True).T, dtype=torch.long)
            degree_features.append(undirected_degree(edgs, n_nodes)[:, None])

        X = np.concatenate([X, *degree_features], axis=1)
    else:
        print("No edge paths - skipping degree features...")

    X_train, X_test = X[train_split], X[test_split]
    y_train, y_test = y_true[train_split], y_true[test_split]

    return X_train, X_test, y_train, y_test


def create_undirected_edge_index(edges_array):
    """Create undirected edge_index from a numpy array of edges of shape (num_edges, 2)."""
    src = torch.tensor(edges_array[:, 0], dtype=torch.long)
    dst = torch.tensor(edges_array[:, 1], dtype=torch.long)
    # Add reverse edges to make it undirected
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return edge_index.contiguous()


def stratified_split_indices(y, train_ratio=0.7, val_ratio=0.1, seed=1):
    """Stratified train/val/test split over node indices.

    Returns (idx_train, idx_val, idx_test) as numpy arrays.
    """
    y = np.asarray(y)
    all_idx = np.arange(len(y))

    idx_train, idx_remaining = train_test_split(
        all_idx, train_size=train_ratio, stratify=y, random_state=seed,
    )

    val_relative_ratio = val_ratio / (1 - train_ratio)

    idx_val, idx_test = train_test_split(
        idx_remaining,
        train_size=val_relative_ratio,
        stratify=y[idx_remaining],
        random_state=seed,
    )

    return idx_train, idx_val, idx_test


def indices_to_mask(indices, num_nodes):
    """Boolean mask tensor of length `num_nodes` with True at `indices`."""
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[indices] = True
    return mask


def sample_pinsage_neighbors(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_neighbors: int = 5,
    num_walks: int = 500,
    walk_length: int = 3,
    restart_prob: float = 0.5,
    min_visits: int = 1,
    return_weights: bool = True,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    PinSAGE-style importance-based neighbor sampling.
    """
    device = edge_index.device

    src_cpu = edge_index[0].cpu().numpy()
    dst_cpu = edge_index[1].cpu().numpy()

    import numpy as np
    order = np.argsort(src_cpu, kind="stable")
    src_sorted = src_cpu[order]
    dst_sorted = dst_cpu[order]

    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.add.at(indptr, src_sorted + 1, 1)
    np.cumsum(indptr, out=indptr)
    neighbors = dst_sorted

    rng = np.random.default_rng(seed)

    out_src: list[int] = []
    out_dst: list[int] = []
    out_w: list[float] = []

    report_every = max(1, num_nodes // 20)

    for target in tqdm(range(num_nodes)):
        deg = indptr[target + 1] - indptr[target]
        if deg == 0:
            continue

        visits: dict[int, int] = {}

        for _ in range(num_walks):
            current = target
            for _ in range(walk_length):
                if rng.random() < restart_prob:
                    current = target
                    continue
                start = indptr[current]
                end = indptr[current + 1]
                if end == start:
                    current = target
                    continue
                current = int(neighbors[rng.integers(start, end)])
                if current != target:
                    visits[current] = visits.get(current, 0) + 1

        if not visits:
            continue

        items = [(v, n) for n, v in visits.items() if v >= min_visits]
        if not items:
            continue
        items.sort(reverse=True)
        items = items[:num_neighbors]

        total = sum(v for v, _ in items)
        for v, n in items:
            out_src.append(target)
            out_dst.append(n)
            out_w.append(v / total)

        if verbose and (target + 1) % report_every == 0:
            print(f"  sampled {target + 1}/{num_nodes} nodes")

    if not out_src:
        empty = torch.empty((2, 0), dtype=torch.long, device=device)
        return empty, (torch.empty(0, device=device) if return_weights else None)

    pruned_edge_index = torch.tensor([out_src, out_dst], dtype=torch.long, device=device)
    weights = torch.tensor(out_w, dtype=torch.float32, device=device) if return_weights else None
    return pruned_edge_index, weights


def sample_pinsage_neighbors_hetero(
    hetero_data,
    node_type: str = "review",
    num_neighbors: int = 15,
    num_walks: int = 300,
    walk_length: int = 3,
    restart_prob: float = 0.5,
    seed: int = 0,
    verbose: bool = False,
):
    """
    Run PinSAGE-style importance sampling independently on each relation
    of a single-node-type heterogeneous graph.

    Mutates hetero_data in place: replaces each edge_index with a pruned
    version and attaches edge_weight per relation.
    """
    num_nodes = hetero_data[node_type].x.size(0)

    for edge_type in hetero_data.edge_types:
        src_type, rel_name, dst_type = edge_type
        assert src_type == node_type == dst_type, (
            f"This sampler assumes a single node type; got {edge_type}"
        )

        ei = hetero_data[edge_type].edge_index
        if verbose:
            print(f"[{rel_name}] sampling from {ei.size(1)} edges...")

        pruned_ei, weights = sample_pinsage_neighbors(
            ei,
            num_nodes=num_nodes,
            num_neighbors=num_neighbors,
            num_walks=num_walks,
            walk_length=walk_length,
            restart_prob=restart_prob,
            return_weights=True,
            seed=seed,
            verbose=False,
        )

        pruned_ei, weights = to_undirected(
            pruned_ei, edge_attr=weights, reduce="mean"
        )

        hetero_data[edge_type].edge_index = pruned_ei
        hetero_data[edge_type].edge_weight = weights

        if verbose:
            print(f"[{rel_name}]   -> {pruned_ei.size(1)} edges after pruning")

    return hetero_data