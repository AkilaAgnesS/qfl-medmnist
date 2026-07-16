
"""Federated training driver for the three SUSQA case studies.

In-process FedAvg with no external orchestrator (no Flower simulation, no Ray).
Each round we sequentially train every client locally on its data partition,
then aggregate parameters with sample-count-weighted averaging, then evaluate
the global model on the centralized test set. Emits a SUSQA report per seed.

Why no Flower simulation:
- Flower 1.20+ split simulation into a Ray-based backend.
- Ray has no wheels for several Python versions (3.13+) at this time.
- Our FL is sequential anyway (single-process simulation), so an explicit
  loop is simpler, more reproducible, and dependency-free.

Usage:
    python experiments/run_federated.py --config experiments/configs/fl_smoke_C1_breast.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loaders import get_dataloaders, get_dataset, get_labels  # noqa: E402
from experiments._common import (  # noqa: E402
    build_model,
    case_study_tag,
    class_weights_from_labels,
    evaluate,
    maybe_start_tracker,
    set_seed,
    stop_tracker,
)
from fl.partition import dirichlet_partition, iid_partition  # noqa: E402
from susqa import SUSQALogger, hybrid_qnn_gate_count  # noqa: E402


# ---------------------------------------------------------------------------
# Parameter <-> tensor helpers
# ---------------------------------------------------------------------------

def get_state_arrays(model: nn.Module) -> list[np.ndarray]:
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def set_state_arrays(model: nn.Module, arrays: list[np.ndarray]) -> None:
    sd = OrderedDict()
    for (k, v_old), v_new in zip(model.state_dict().items(), arrays):
        t = torch.as_tensor(v_new).reshape(v_old.shape).to(v_old.dtype)
        sd[k] = t
    model.load_state_dict(sd, strict=True)


def fedavg_aggregate(updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """FedAvg = sample-count-weighted average of per-client parameter lists."""
    total = float(sum(n for _, n in updates))
    aggregated: list[np.ndarray] = []
    n_layers = len(updates[0][0])
    for layer_idx in range(n_layers):
        weighted = sum((n / total) * params[layer_idx] for params, n in updates)
        aggregated.append(weighted)
    return aggregated


# ---------------------------------------------------------------------------
# Local client training
# ---------------------------------------------------------------------------

def train_local(
    model_cfg: dict,
    init_params: list[np.ndarray],
    train_loader: DataLoader,
    local_epochs: int,
    lr: float,
    class_weights: torch.Tensor | None,
    device: str,
) -> tuple[list[np.ndarray], int]:
    """Train one client locally; return (updated_params, num_samples)."""
    model = build_model(model_cfg).to(device)
    set_state_arrays(model, init_params)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    weights = class_weights.to(device) if class_weights is not None else None
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    model.train()
    for _ in range(local_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    return get_state_arrays(model), len(train_loader.dataset)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_one_seed(cfg: dict, seed: int, device: str) -> dict:
    set_seed(seed)
    dataset = cfg["data"]["dataset"]
    batch_size = cfg["data"].get("batch_size", 32)
    data_root = cfg["data"].get("root", "data/raw")

    n_clients = cfg["fl"]["n_clients"]
    rounds = cfg["fl"]["rounds"]
    local_epochs = cfg["fl"].get("local_epochs", 1)
    partition_kind = cfg["fl"].get("partition", "iid")
    alpha = cfg["fl"].get("dirichlet_alpha", 0.5)

    lr = cfg["training"].get("lr", 1e-3)
    use_class_weighting = cfg["training"].get("class_weighting", True)

    # Partition the training labels across clients (deterministic per seed).
    train_labels = get_labels(dataset, split="train", root=data_root)
    if partition_kind == "iid":
        partitions = iid_partition(train_labels, n_clients, seed=seed)
    elif partition_kind == "dirichlet":
        partitions = dirichlet_partition(train_labels, n_clients, alpha=alpha, seed=seed)
    else:
        raise ValueError(f"Unknown partition kind: {partition_kind!r}")
    print(f"  [seed {seed}] partition sizes: {[len(p) for p in partitions]}")

    num_classes = cfg["model"].get("num_classes", 2)
    class_weights = (
        class_weights_from_labels(train_labels, num_classes=num_classes)
        if use_class_weighting else None
    )
    if class_weights is not None:
        print(f"  [seed {seed}] class weights: {class_weights.numpy().round(3).tolist()}")

    # Per-client training loaders + a centralized test loader for global evaluation.
    # Empty partitions (possible under low-alpha Dirichlet) become None so we
    # can skip them in the training loop without crashing the DataLoader.
    train_full = get_dataset(dataset, split="train", root=data_root)
    client_loaders: list[DataLoader | None] = []
    for idx in partitions:
        if len(idx) == 0:
            client_loaders.append(None)
        else:
            client_loaders.append(
                DataLoader(Subset(train_full, idx.tolist()), batch_size=batch_size, shuffle=True)
            )
    n_active = sum(1 for cl in client_loaders if cl is not None)
    if n_active < n_clients:
        print(
            f"  [seed {seed}] WARNING: {n_clients - n_active} client(s) have 0 samples and "
            f"will be skipped this seed (extreme non-IID partition). Active clients: {n_active}."
        )
    if n_active == 0:
        raise RuntimeError("All client partitions are empty — partition is degenerate.")

    _, _, test_loader = get_dataloaders(dataset, batch_size=batch_size, root=data_root)

    # Initialize the global model and capture its initial parameters.
    global_model = build_model(cfg["model"]).to(device)
    global_params = get_state_arrays(global_model)

    # Logger is created before training so CodeCarbon can write into results_dir.
    experiment_id = f"{cfg['experiment_id']}__seed{seed}"
    logger = SUSQALogger(
        experiment_id=experiment_id,
        case_study=case_study_tag(cfg["model"]["name"]),
        dataset=dataset,
    )
    tracker = maybe_start_tracker(cfg["training"].get("codecarbon", False), logger.results_dir)

    history: list[dict] = []
    print(f"  [seed {seed}] starting {rounds} rounds with {n_active}/{n_clients} clients ({partition_kind})")
    t0 = time.time()
    for r in range(1, rounds + 1):
        updates = []
        for cid in range(n_clients):
            if client_loaders[cid] is None:
                continue
            params, n = train_local(
                model_cfg=cfg["model"],
                init_params=global_params,
                train_loader=client_loaders[cid],
                local_epochs=local_epochs,
                lr=lr,
                class_weights=class_weights,
                device=device,
            )
            updates.append((params, n))
        global_params = fedavg_aggregate(updates)

        # Evaluate global model on test set.
        set_state_arrays(global_model, global_params)
        m = evaluate(global_model, test_loader, device)
        m["round"] = r
        history.append(m)
        print(
            f"  [server] round {r}/{rounds}: "
            f"acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  auc={m['auc']:.3f}"
        )
    train_time = time.time() - t0
    stop_tracker(tracker, logger)

    # Save final global weights so trained circuits can be re-evaluated later
    # (e.g. under a hardware-calibrated noise model) without retraining.
    torch.save(global_model.state_dict(), logger.results_dir / "checkpoint.pt")

    final = history[-1]

    logger.log_parameters(global_model)
    logger.log_communication(n_clients=n_clients, n_rounds=rounds)
    logger.log_metrics(final["accuracy"], final["f1"], final["auc"])
    logger.add_note(f"train_seconds={train_time:.1f} partition={partition_kind}")

    if cfg["model"]["name"] == "hybrid_qnn":
        n_qubits = cfg["model"].get("n_qubits", 8)
        n_layers = cfg["model"].get("n_layers", 2)
        gate_count = hybrid_qnn_gate_count(n_qubits, n_layers)
        shots = cfg["model"].get("shots_per_inference", 1000)
        # Total samples seen across all clients per round, times rounds * local_epochs.
        samples = sum(len(p) for p in partitions)
        epoch_equivalents = rounds * local_epochs
        logger.log_quantum_cost(
            gate_count=gate_count, shots=shots, samples=samples, epochs=epoch_equivalents
        )
        logger.add_note(f"gate_count={gate_count} n_qubits={n_qubits} n_layers={n_layers}")

    p_levels = cfg.get("noise_sweep", [0.0])
    for p in p_levels:
        logger.log_noise_robustness(p=p, accuracy_at_p=final["accuracy"], baseline=final["accuracy"])

    row = logger.finalize()
    print(f"  [seed {seed}] FINAL: {row}")
    (logger.results_dir / "history.json").write_text(json.dumps(history, indent=2))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Override the config's seed list (e.g. --seeds 3 4 5 6 7 8 9 to extend to 10 seeds).",
    )
    parser.add_argument(
        "--codecarbon", action="store_true",
        help="Track energy/CO2 of this run with CodeCarbon (overrides config).",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.codecarbon:
        cfg.setdefault("training", {})["codecarbon"] = True
    seeds = args.seeds if args.seeds is not None else cfg["training"].get("seed", [0])
    if isinstance(seeds, int):
        seeds = [seeds]

    print(f"Running {cfg['experiment_id']} on {args.device} with seeds {seeds}")
    rows = []
    for s in seeds:
        rows.append(run_one_seed(cfg, seed=s, device=args.device))

    print("\n=== SUSQA rows for this experiment ===")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
