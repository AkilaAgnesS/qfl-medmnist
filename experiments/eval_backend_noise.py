"""Evaluate trained hybrid checkpoints under a hardware-calibrated noise model.

Addresses Reviewer 1 (major revision, SUSCOM): depolarising noise alone is a
theoretical stress test; real devices add readout error, T1/T2 relaxation,
gate-dependent and coherent errors, and connectivity constraints. This script
re-evaluates already-trained circuits (no retraining) under three settings and
reports eta for each:

    1. ideal        — noise-free default.qubit simulator (analytic)
    2. depolarizing — default.mixed with each p in the config's noise_sweep
    3. backend      — qiskit-aer simulator carrying NoiseModel.from_backend(...)
                      calibrated against a real IBM Quantum backend (or a fake
                      snapshot backend if no IBM account/backend is available)

Requires: pennylane-qiskit, qiskit-aer, qiskit-ibm-runtime  (see requirements.txt)

Usage (fake calibrated backend, all seeds of one experiment):
    python experiments/eval_backend_noise.py --config experiments/configs/baseline_C3_breast.yaml

Real backend (needs a saved IBM Quantum account/token):
    python experiments/eval_backend_noise.py --config ... --backend ibm_brisbane

Output: results/<experiment_id>__seed<k>/backend_noise_report.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pennylane as qml
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loaders import get_dataloaders  # noqa: E402
from experiments._common import build_model, evaluate, set_seed  # noqa: E402
from models.hybrid_qnn import HybridQNN  # noqa: E402

DEFAULT_FAKE_BACKEND = "FakeGuadalupeV2"  # 16-qubit calibration snapshot, no account needed


def get_backend(name: str):
    """Return an IBM backend (real if `name` is an IBM backend name, else fake)."""
    if name.lower().startswith("fake"):
        from qiskit_ibm_runtime import fake_provider

        cls = getattr(fake_provider, name, None)
        if cls is None:
            available = [n for n in dir(fake_provider) if n.startswith("Fake")]
            raise ValueError(f"Unknown fake backend {name!r}. Available: {available}")
        return cls()
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    return service.backend(name)


def build_hybrid_with_device(model_cfg: dict, dev) -> HybridQNN:
    return HybridQNN(
        in_channels=model_cfg.get("in_channels", 1),
        num_classes=model_cfg.get("num_classes", 2),
        n_qubits=model_cfg.get("n_qubits", 8),
        n_layers=model_cfg.get("n_layers", 2),
        noise_p=0.0,
        dev=dev,
    )


def eval_checkpoint(model, state_dict, loader, device) -> dict:
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return evaluate(model, loader, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override the config's seed list.")
    parser.add_argument("--backend", default=DEFAULT_FAKE_BACKEND,
                        help="IBM backend name (e.g. ibm_brisbane) or a fake_provider "
                             f"class name (default: {DEFAULT_FAKE_BACKEND}).")
    parser.add_argument("--shots", type=int, default=1000,
                        help="Shots for the backend-calibrated evaluation (default 1000, "
                             "matching shots_per_inference in the configs).")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Optionally cap test batches (noisy Aer evaluation is slow).")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if cfg["model"]["name"] != "hybrid_qnn":
        raise SystemExit("Backend-noise evaluation only applies to the hybrid_qnn case study.")

    seeds = args.seeds if args.seeds is not None else cfg["training"].get("seed", [0])
    if isinstance(seeds, int):
        seeds = [seeds]

    n_qubits = cfg["model"].get("n_qubits", 8)
    dataset = cfg["data"]["dataset"]
    batch_size = cfg["data"].get("batch_size", 32)
    _, _, test_loader = get_dataloaders(dataset, batch_size=batch_size,
                                        root=cfg["data"].get("root", "data/raw"))
    if args.max_batches is not None:
        from itertools import islice

        class _Capped:
            def __init__(self, loader, n):
                self.loader, self.n = loader, n
            def __iter__(self):
                return islice(iter(self.loader), self.n)
        test_loader = _Capped(test_loader, args.max_batches)

    # Hardware-calibrated noise model (includes readout error, T1/T2, gate errors).
    print(f"Building noise model from backend: {args.backend}")
    backend = get_backend(args.backend)
    from qiskit_aer.noise import NoiseModel

    noise_model = NoiseModel.from_backend(backend)
    backend_dev = qml.device("qiskit.aer", wires=n_qubits,
                             noise_model=noise_model, shots=args.shots)

    p_levels = [p for p in cfg.get("noise_sweep", [1e-3, 1e-2]) if p > 0]

    for seed in seeds:
        set_seed(seed)
        exp_id = f"{cfg['experiment_id']}__seed{seed}"
        results_dir = ROOT / "results" / exp_id
        ckpt_path = results_dir / "checkpoint.pt"
        if not ckpt_path.exists():
            print(f"  [seed {seed}] SKIP — no checkpoint at {ckpt_path} "
                  f"(re-run training with the updated driver to produce one)")
            continue
        state_dict = torch.load(ckpt_path, map_location="cpu")

        # 1. ideal
        ideal_model = build_model(cfg["model"])
        m_ideal = eval_checkpoint(ideal_model, state_dict, test_loader, args.device)
        baseline = m_ideal["accuracy"]
        report = {
            "experiment_id": exp_id,
            "backend": args.backend,
            "shots_backend_eval": args.shots,
            "ideal": m_ideal,
            "depolarizing": {},
            "backend_calibrated": {},
        }
        print(f"  [seed {seed}] ideal acc={baseline:.4f}")

        # 2. depolarizing sweep (same as paper, for side-by-side comparison)
        for p in p_levels:
            mc = copy.deepcopy(cfg["model"])
            mc["noise_p"] = p
            noisy = build_model(mc)
            m_p = eval_checkpoint(noisy, state_dict, test_loader, args.device)
            eta = m_p["accuracy"] / baseline if baseline > 0 else 0.0
            report["depolarizing"][f"{p:.0e}"] = {**m_p, "eta": eta}
            print(f"  [seed {seed}] depolarizing p={p:.0e}  acc={m_p['accuracy']:.4f}  eta={eta:.4f}")

        # 3. backend-calibrated
        bk_model = build_hybrid_with_device(cfg["model"], backend_dev)
        m_bk = eval_checkpoint(bk_model, state_dict, test_loader, args.device)
        eta_bk = m_bk["accuracy"] / baseline if baseline > 0 else 0.0
        report["backend_calibrated"] = {**m_bk, "eta": eta_bk}
        print(f"  [seed {seed}] backend({args.backend})  acc={m_bk['accuracy']:.4f}  eta={eta_bk:.4f}")

        out = results_dir / "backend_noise_report.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"  [seed {seed}] wrote {out}")


if __name__ == "__main__":
    main()
