"""Per-round convergence plots from FL history.json files.

Reads results/<experiment_id>__seed{N}/history.json for every FL run, groups
by (dataset, setting, case_study), and plots metric-vs-round curves with
seed-band shading. One PNG per (dataset, setting, metric).

This is Figure 3 of the paper.

Usage:
    python notebooks/04_convergence_plots.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


CASE_COLORS = {
    "C1_classical": "#2c7fb8",
    "C2_compressed": "#7fcdbb",
    "C3_hybrid_quantum": "#d95f0e",
}


def parse_setting(experiment_id: str) -> str:
    eid = experiment_id.lower()
    if eid.startswith("baseline_"):
        return "centralized"
    if "_iid" in eid:
        return "fl_iid"
    if "dirichlet01" in eid:
        return "fl_dirichlet_alpha=0.1"
    if "dirichlet05" in eid:
        return "fl_dirichlet_alpha=0.5"
    return "unknown"


def parse_case_study(experiment_id: str) -> str:
    eid = experiment_id.lower()
    if "classical" in eid and "compressed" not in eid:
        return "C1_classical"
    if "compressed" in eid:
        return "C2_compressed"
    if "hybrid" in eid:
        return "C3_hybrid_quantum"
    return "unknown"


def parse_dataset(experiment_id: str) -> str:
    eid = experiment_id.lower()
    if "breast" in eid:
        return "breastmnist"
    if "pneumonia" in eid:
        return "pneumoniamnist"
    if "derma" in eid:
        return "dermamnist"
    return "unknown"


def load_histories():
    """Return {(dataset, setting, case_study): list of per-seed histories (each a list of dicts)}."""
    grouped: dict[tuple, list[list[dict]]] = defaultdict(list)
    for hist_path in sorted(RESULTS_DIR.glob("*/history.json")):
        exp_dir = hist_path.parent.name  # e.g. "fl_C2_compressed_breast_dirichlet05__seed3"
        m = re.match(r"^(?P<eid>.+)__seed(?P<seed>\d+)$", exp_dir)
        if not m:
            continue
        eid = m.group("eid")
        history = json.loads(hist_path.read_text())
        if not history:
            continue
        key = (parse_dataset(eid), parse_setting(eid), parse_case_study(eid))
        grouped[key].append(history)
    return grouped


def plot_metric_vs_round(grouped, dataset, setting, metric, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cases_in_panel = []
    for (ds, st, cs), seed_histories in grouped.items():
        if ds != dataset or st != setting:
            continue
        # Stack seed histories; pad shorter ones with NaN if needed.
        max_rounds = max(len(h) for h in seed_histories)
        arr = np.full((len(seed_histories), max_rounds), np.nan)
        for i, h in enumerate(seed_histories):
            arr[i, :len(h)] = [step.get(metric, np.nan) for step in h]
        rounds = np.arange(1, max_rounds + 1)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        color = CASE_COLORS.get(cs, "gray")
        ax.plot(rounds, mean, color=color, linewidth=2, label=cs)
        ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.20)
        cases_in_panel.append(cs)
    if not cases_in_panel:
        plt.close(fig)
        return False
    ax.set_xlabel("FL round")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{dataset} — {setting} — {metric.upper()} vs round (mean ± std over seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def main():
    print(f"Scanning {RESULTS_DIR} for history.json files...")
    grouped = load_histories()
    if not grouped:
        raise SystemExit("No history.json files found. Run at least one FL experiment first.")
    settings = sorted({k[1] for k in grouped})
    datasets = sorted({k[0] for k in grouped})
    metrics = ["accuracy", "f1", "auc"]
    print(f"  found {sum(len(v) for v in grouped.values())} histories across "
          f"{len(grouped)} (dataset, setting, case_study) groups")

    written = 0
    for dataset in datasets:
        for setting in settings:
            for metric in metrics:
                out = FIG_DIR / f"convergence_{dataset}_{setting}_{metric}.png".replace("=", "")
                if plot_metric_vs_round(grouped, dataset, setting, metric, out):
                    print(f"  wrote {out}")
                    written += 1
    print(f"\nDone. {written} convergence plots in {FIG_DIR}")


if __name__ == "__main__":
    main()
