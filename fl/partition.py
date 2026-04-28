"""Partition a dataset across federated clients.

Two schemes:
- iid: uniform random shards
- dirichlet: label-skewed shards via Dirichlet(alpha). alpha=100 ~ IID, alpha=0.1 ~ heavily skewed.
"""
from __future__ import annotations

import numpy as np


def iid_partition(labels: np.ndarray, n_clients: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    return [a for a in np.array_split(idx, n_clients)]


def dirichlet_partition(
    labels: np.ndarray, n_clients: int, alpha: float, seed: int = 0
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1
    client_idx: list[list[int]] = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet([alpha] * n_clients)
        # cumulative split points
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        chunks = np.split(idx_c, splits)
        for i, ch in enumerate(chunks):
            client_idx[i].extend(ch.tolist())
    return [np.array(sorted(c)) for c in client_idx]


if __name__ == "__main__":
    labels = np.random.randint(0, 2, size=1000)
    iid = iid_partition(labels, 5)
    diri = dirichlet_partition(labels, 5, alpha=0.1)
    print("IID sizes:", [len(p) for p in iid])
    print("Dirichlet(0.1) sizes:", [len(p) for p in diri])
    for i, p in enumerate(diri):
        print(f"  client {i} class balance:", np.bincount(labels[p], minlength=2))
