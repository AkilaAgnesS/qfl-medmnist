"""Centralized training driver for the three SUSQA case studies.

Runs one config across the requested seeds, trains a model, evaluates it,
runs the noise-robustness sweep (for quantum case studies), and writes a
SUSQA report per seed.

Usage:
    python experiments/run_centralized.py --config experiments/configs/smoke_test_C1.yaml
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loaders import get_dataloaders  # noqa: E402
from experiments._common import (  # noqa: E402
    build_model,
    case_study_tag,
    class_weights_from_loader,
    evaluate,
    set_seed,
    train_one_epoch,
)
from susqa import SUSQALogger, hybrid_qnn_gate_count  # noqa: E402


def evaluate_under_noise(model_cfg: dict, state_dict, test_loader, device, p: float) -> dict:
    """Rebuild a hybrid model with depolarizing noise p, load weights, evaluate."""
    if model_cfg["name"] != "hybrid_qnn":
        return {"accuracy": float("nan"), "f1": float("nan"), "auc": float("nan")}
    cfg = copy.deepcopy(model_cfg)
    cfg["noise_p"] = p
    noisy = build_model(cfg).to(device)
    noisy.load_state_dict(state_dict, strict=False)
    return evaluate(noisy, test_loader, device)


def run_one_seed(cfg: dict, seed: int, device: str) -> dict:
    set_seed(seed)
    dataset = cfg["data"]["dataset"]
    batch_size = cfg["data"].get("batch_size", 32)
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"].get("lr", 1e-3)

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset, batch_size=batch_size, root=cfg["data"].get("root", "data/raw")
    )

    model = build_model(cfg["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    num_classes = cfg["model"].get("num_classes", 2)
    if cfg["training"].get("class_weighting", True):
        weights = class_weights_from_loader(train_loader, num_classes=num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        print(f"  [seed {seed}] class weights: {weights.cpu().numpy().round(3).tolist()}")
    else:
        loss_fn = nn.CrossEntropyLoss()

    experiment_id = f"{cfg['experiment_id']}__seed{seed}"
    logger = SUSQALogger(
        experiment_id=experiment_id,
        case_study=case_study_tag(cfg["model"]["name"]),
        dataset=dataset,
    )
    logger.log_parameters(model)

    t0 = time.time()
    for epoch in range(epochs):
        train_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"  [seed {seed}] epoch {epoch + 1}/{epochs}  "
            f"train_acc={train_acc:.3f}  "
            f"val_acc={val_metrics['accuracy']:.3f}  "
            f"val_f1={val_metrics['f1']:.3f}  "
            f"val_auc={val_metrics['auc']:.3f}"
        )
    train_time = time.time() - t0

    test_metrics = evaluate(model, test_loader, device)
    test_acc = test_metrics["accuracy"]
    logger.log_metrics(test_acc, test_metrics["f1"], test_metrics["auc"])
    logger.add_note(f"train_seconds={train_time:.1f}")

    logger.log_communication(n_clients=1, n_rounds=1)

    if cfg["model"]["name"] == "hybrid_qnn":
        n_qubits = cfg["model"].get("n_qubits", 8)
        n_layers = cfg["model"].get("n_layers", 2)
        gate_count = hybrid_qnn_gate_count(n_qubits, n_layers)
        shots = cfg["model"].get("shots_per_inference", 1000)
        samples = len(train_loader.dataset)
        logger.log_quantum_cost(gate_count=gate_count, shots=shots, samples=samples, epochs=epochs)
        logger.add_note(f"gate_count={gate_count} n_qubits={n_qubits} n_layers={n_layers}")

    p_levels = cfg.get("noise_sweep", [0.0, 1e-3, 1e-2])
    for p in p_levels:
        if cfg["model"]["name"] == "hybrid_qnn":
            metrics_p = evaluate_under_noise(cfg["model"], model.state_dict(), test_loader, device, p)
            acc_p = metrics_p["accuracy"]
        else:
            acc_p = test_acc
        logger.log_noise_robustness(p=p, accuracy_at_p=acc_p, baseline=test_acc)

    row = logger.finalize()
    print(f"  [seed {seed}] FINAL: {row}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seeds = cfg["training"].get("seed", [0])
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
