"""SUSQA — SUstainable Quantum federated learning Accounting.

Reference implementation of the four-axis sustainability protocol introduced
in the paper. Drop a SUSQALogger into any centralized or federated training
loop and call its hooks to emit a four-axis report per case study:

    P        — trainable parameter count
    C_total  — total federated communication (bytes)
    Q        — quantum resource cost (gate-shots-samples-epochs)
    eta      — noise robustness (accuracy retention vs depolarizing p)

The logger is intentionally thin: it does not assume a specific FL framework,
model class, or quantum backend. Train however you want, then call the hooks.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


__version__ = "0.2.0"


@dataclass
class SUSQAReport:
    """One row of the SUSQA reporting template."""

    experiment_id: str
    case_study: str = ""              # "C1_classical" | "C2_compressed" | "C3_hybrid_quantum"
    dataset: str = ""
    accuracy: float = 0.0
    f1: float = 0.0                   # macro F1 — primary metric for imbalanced medical binary
    auc: float = 0.0                  # ROC-AUC — primary metric for imbalanced medical binary
    P: int = 0                        # trainable parameter count
    P_bytes_fp32: int = 0
    C_round_bytes: int = 0            # one direction per client
    C_total_bytes: int = 0
    Q: int = 0                        # gate-shots-samples-epochs (0 for classical)
    eta: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def as_row(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "case_study": self.case_study,
            "dataset": self.dataset,
            "accuracy": round(self.accuracy, 4),
            "f1": round(self.f1, 4),
            "auc": round(self.auc, 4),
            "P": self.P,
            "C_total_MB": round(self.C_total_bytes / 1e6, 3),
            "Q": self.Q,
            **{f"eta_p={k}": round(v, 4) for k, v in self.eta.items()},
        }


class SUSQALogger:
    """Per-experiment logger that fills a SUSQAReport via explicit hooks.

    Usage:
        logger = SUSQALogger("classical_breast", case_study="C1_classical", dataset="breastmnist")
        logger.log_parameters(model)
        # ... train ...
        logger.log_accuracy(test_acc)
        logger.log_communication(n_clients=1, n_rounds=1)   # centralized => degenerate case
        # for quantum case studies:
        logger.log_quantum_cost(gate_count=42, shots=1000, samples=4708, epochs=5)
        # for noise robustness:
        logger.log_noise_robustness(p=1e-3, accuracy_at_p=0.85, baseline=0.88)
        logger.finalize()
    """

    def __init__(
        self,
        experiment_id: str,
        case_study: str = "",
        dataset: str = "",
        results_dir: str | Path = "results",
    ) -> None:
        self.results_dir = Path(results_dir) / experiment_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.report = SUSQAReport(
            experiment_id=experiment_id,
            case_study=case_study,
            dataset=dataset,
        )

    # --- Axis 1: parameters ---------------------------------------------------
    def log_parameters(self, model: torch.nn.Module) -> int:
        P = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.report.P = P
        self.report.P_bytes_fp32 = P * 4
        return P

    # --- Axis 2: communication ------------------------------------------------
    def log_communication(self, n_clients: int, n_rounds: int) -> int:
        """Compute total federated communication.

        For a centralized run pass n_clients=1, n_rounds=1: communication is
        then a single up+down of the model, which is a useful but degenerate
        baseline for the C_total column.
        """
        c_round = self.report.P_bytes_fp32  # one direction
        self.report.C_round_bytes = c_round
        self.report.C_total_bytes = 2 * n_clients * c_round * n_rounds
        return self.report.C_total_bytes

    # --- Axis 3: quantum cost -------------------------------------------------
    def log_quantum_cost(
        self, gate_count: int, shots: int, samples: int, epochs: int
    ) -> int:
        Q = int(gate_count) * int(shots) * int(samples) * int(epochs)
        self.report.Q = Q
        return Q

    # --- Axis 4: noise robustness --------------------------------------------
    def log_noise_robustness(self, p: float, accuracy_at_p: float, baseline: float) -> float:
        eta = accuracy_at_p / baseline if baseline > 0 else 0.0
        self.report.eta[f"{p:.0e}"] = eta
        return eta

    # --- Accuracy / housekeeping ---------------------------------------------
    def log_accuracy(self, accuracy: float) -> None:
        self.report.accuracy = float(accuracy)

    def log_metrics(self, accuracy: float, f1: float, auc: float) -> None:
        self.report.accuracy = float(accuracy)
        self.report.f1 = float(f1)
        self.report.auc = float(auc)

    def add_note(self, note: str) -> None:
        sep = " | " if self.report.notes else ""
        self.report.notes = f"{self.report.notes}{sep}{note}"

    # --- Persistence ----------------------------------------------------------
    def finalize(self) -> dict:
        path = self.results_dir / "susqa_report.json"
        path.write_text(json.dumps(asdict(self.report), indent=2))
        return self.report.as_row()


def hybrid_qnn_gate_count(n_qubits: int, n_layers: int) -> int:
    """Closed-form gate count for the ansatz used in models/hybrid_qnn.py.

    Per forward pass:
        - n_qubits RY gates (encoding)
        - n_layers * (n_qubits RY + n_qubits RZ + n_qubits CNOT) (variational)

    Noise channels (when present) are not "gates" in the cost sense; we exclude them.
    """
    return n_qubits + n_layers * (3 * n_qubits)


def aggregate_reports(results_dir: str | Path) -> list[dict]:
    """Walk results_dir and return a list of SUSQA report rows for paper tables."""
    rows: list[dict] = []
    for p in Path(results_dir).glob("*/susqa_report.json"):
        data = json.loads(p.read_text())
        # Reconstruct a SUSQAReport so .as_row() is consistent.
        report = SUSQAReport(**data)
        rows.append(report.as_row())
    return rows


if __name__ == "__main__":
    # Tiny self-test
    logger = SUSQALogger("susqa_selftest", case_study="C1_classical", dataset="breastmnist")
    fake_model = torch.nn.Linear(10, 2)
    logger.log_parameters(fake_model)
    logger.log_communication(n_clients=5, n_rounds=10)
    logger.log_accuracy(0.85)
    logger.log_noise_robustness(p=0, accuracy_at_p=0.85, baseline=0.85)
    print(json.dumps(logger.finalize(), indent=2))
