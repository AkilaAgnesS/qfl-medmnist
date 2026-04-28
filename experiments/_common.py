"""Shared utilities for centralized and federated training drivers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def build_model(model_cfg: dict) -> nn.Module:
    name = model_cfg["name"]
    if name == "classical_cnn":
        from models.classical_cnn import ClassicalCNN
        return ClassicalCNN(
            in_channels=model_cfg.get("in_channels", 1),
            num_classes=model_cfg.get("num_classes", 2),
        )
    if name == "classical_compressed":
        from models.classical_compressed import ClassicalCompressed
        return ClassicalCompressed(
            in_channels=model_cfg.get("in_channels", 1),
            num_classes=model_cfg.get("num_classes", 2),
            hidden=tuple(model_cfg.get("hidden", (16, 8))),
        )
    if name == "hybrid_qnn":
        from models.hybrid_qnn import HybridQNN
        return HybridQNN(
            in_channels=model_cfg.get("in_channels", 1),
            num_classes=model_cfg.get("num_classes", 2),
            n_qubits=model_cfg.get("n_qubits", 8),
            n_layers=model_cfg.get("n_layers", 2),
            noise_p=model_cfg.get("noise_p", 0.0),
        )
    raise ValueError(f"Unknown model name: {name!r}")


def case_study_tag(model_name: str) -> str:
    return {
        "classical_cnn": "C1_classical",
        "classical_compressed": "C2_compressed",
        "hybrid_qnn": "C3_hybrid_quantum",
    }.get(model_name, model_name)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, opt, loss_fn, device) -> float:
    model.train()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total += y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_y, all_pred, all_probs = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        all_y.append(y.cpu().numpy())
        all_pred.append(logits.argmax(1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    probs = np.concatenate(all_probs, axis=0)
    num_classes = probs.shape[1]

    acc = float((y_true == y_pred).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        auc = 0.5
    elif num_classes == 2:
        auc = float(roc_auc_score(y_true, probs[:, 1]))
    else:
        try:
            auc = float(roc_auc_score(
                y_true, probs, multi_class="ovr",
                average="macro", labels=np.arange(num_classes),
            ))
        except ValueError:
            present = unique_classes
            mask = np.isin(np.arange(num_classes), present)
            sub = probs[:, mask]
            sub = sub / sub.sum(axis=1, keepdims=True).clip(min=1e-12)
            auc = float(roc_auc_score(
                y_true, sub, multi_class="ovr",
                average="macro", labels=present,
            ))
    return {"accuracy": acc, "f1": f1, "auc": auc}


def _weights_from_counts(counts, num_classes):
    """Linear inverse-frequency for binary; sqrt-scaled for multi-class."""
    counts = counts[:num_classes]
    total = counts.sum()
    linear = total / (num_classes * np.maximum(counts, 1))
    if num_classes <= 2:
        weights = linear
    else:
        weights = np.sqrt(linear)
        weights = weights * (num_classes / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


def class_weights_from_loader(loader, num_classes=2):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in loader:
        c = np.bincount(y.numpy(), minlength=num_classes)
        counts += c[:num_classes]
    return _weights_from_counts(counts, num_classes)


def class_weights_from_labels(labels, num_classes=2):
    counts = np.bincount(labels, minlength=num_classes)[:num_classes]
    return _weights_from_counts(counts, num_classes)
