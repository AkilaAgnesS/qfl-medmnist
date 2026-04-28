"""SUSQA analysis: aggregate reports, generate paper figures and tables.

Reads every results/<exp>/susqa_report.json, groups by (case_study, setting,
dataset), computes 3-seed mean ± std for each metric, runs paired Wilcoxon
significance tests, and emits:

    results/figures/pareto_auc_vs_params.png
    results/figures/pareto_auc_vs_comm.png
    results/figures/auc_by_setting.png
    results/figures/comparison_table.csv  (the headline SUSQA table for the paper)
    results/figures/stats_tests.csv

Usage:
    python notebooks/03_susqa_analysis.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def parse_setting(experiment_id: str) -> str:
    """Infer the experimental setting from the experiment_id."""
    eid = experiment_id.lower()
    if eid.startswith("baseline_"):
        return "centralized"
    if "_iid" in eid:
        return "fl_iid"
    if "_dirichlet" in eid:
        m = re.search(r"dirichlet0?(\d+)", eid)
        return f"fl_dirichlet_alpha={'0.' + m.group(1) if m else 'X'}"
    if eid.startswith("fl_smoke") or eid.startswith("smoke_"):
        return "smoke"
    return "unknown"


def load_reports(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*/susqa_report.json")):
        data = json.loads(path.read_text())
        # Drop quirky fields we don't want to flatten.
        eta = data.get("eta", {}) or {}
        flat = {
            "experiment_id": data["experiment_id"],
            "case_study": data["case_study"],
            "dataset": data["dataset"],
            "setting": parse_setting(data["experiment_id"]),
            "accuracy": data.get("accuracy", float("nan")),
            "f1": data.get("f1", float("nan")),
            "auc": data.get("auc", float("nan")),
            "P": data.get("P", 0),
            "C_total_MB": data.get("C_total_bytes", 0) / 1e6,
            "Q": data.get("Q", 0),
            **{f"eta_p={k}": v for k, v in eta.items()},
            "notes": data.get("notes", ""),
        }
        rows.append(flat)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(
            f"No SUSQA reports found under {results_dir}. "
            "Run at least one experiment before invoking this script."
        )
    return df


# ---------------------------------------------------------------------------
# Aggregation (mean ± std across seeds)
# ---------------------------------------------------------------------------

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "setting", "case_study"]
    metric_cols = ["accuracy", "f1", "auc", "P", "C_total_MB", "Q"]
    g = df.groupby(group_cols)[metric_cols]
    mean = g.mean().add_suffix("_mean")
    std = g.std().add_suffix("_std")
    n = g.size().rename("n_seeds")
    out = pd.concat([mean, std, n], axis=1).reset_index()
    return out


def comparison_table(agg: pd.DataFrame) -> pd.DataFrame:
    """The headline SUSQA table (Table 6 of the paper).

    Columns: case_study | setting | dataset | AUC | F1 | P | C_total_MB | Q
    Values:  formatted "mean ± std"
    """
    rows = []
    for _, r in agg.iterrows():
        rows.append({
            "dataset": r["dataset"],
            "setting": r["setting"],
            "case_study": r["case_study"],
            "AUC": f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}",
            "F1": f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f}",
            "Acc": f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}",
            "P": int(r["P_mean"]),
            "C_total_MB": f"{r['C_total_MB_mean']:.3f}",
            "Q": int(r["Q_mean"]),
            "n_seeds": int(r["n_seeds"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests (paired Wilcoxon across seeds)
# ---------------------------------------------------------------------------

def pair_for_wilcoxon(df: pd.DataFrame, dataset: str, setting: str, a: str, b: str, metric: str = "auc"):
    """Return (a_values, b_values) aligned by seed for a paired test."""
    sub = df[(df["dataset"] == dataset) & (df["setting"] == setting)].copy()
    sub["seed"] = sub["experiment_id"].str.extract(r"seed(\d+)$").astype(int)
    a_vals = sub[sub["case_study"] == a].sort_values("seed")[metric].values
    b_vals = sub[sub["case_study"] == b].sort_values("seed")[metric].values
    return a_vals, b_vals


def run_stats(df: pd.DataFrame) -> pd.DataFrame:
    case_studies = sorted(df["case_study"].unique())
    rows = []
    for dataset in df["dataset"].unique():
        for setting in df[df["dataset"] == dataset]["setting"].unique():
            for i, a in enumerate(case_studies):
                for b in case_studies[i + 1:]:
                    av, bv = pair_for_wilcoxon(df, dataset, setting, a, b, metric="auc")
                    if len(av) < 2 or len(bv) < 2 or len(av) != len(bv):
                        continue
                    try:
                        stat, p = wilcoxon(av, bv, zero_method="wilcox")
                    except ValueError:
                        stat, p = float("nan"), float("nan")
                    rows.append({
                        "dataset": dataset,
                        "setting": setting,
                        "comparison": f"{a} vs {b} (AUC)",
                        "stat": stat,
                        "p_value": p,
                        "n_pairs": len(av),
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

CASE_COLORS = {
    "C1_classical": "#2c7fb8",
    "C2_compressed": "#7fcdbb",
    "C3_hybrid_quantum": "#d95f0e",
}
CASE_MARKERS = {
    "C1_classical": "o",
    "C2_compressed": "s",
    "C3_hybrid_quantum": "^",
}
SETTING_LINESTYLE = {
    "centralized": "-",
    "fl_iid": "--",
    "fl_dirichlet_alpha=0.5": "-.",
    "fl_dirichlet_alpha=0.1": ":",
}


def pareto_plot(agg: pd.DataFrame, x_col: str, x_label: str, out_path: Path):
    """Pareto: AUC (y) vs cost (x). One marker per (case study, setting)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for _, r in agg.iterrows():
        cs = r["case_study"]
        ax.errorbar(
            r[f"{x_col}_mean"],
            r["auc_mean"],
            yerr=r["auc_std"],
            fmt=CASE_MARKERS.get(cs, "x"),
            color=CASE_COLORS.get(cs, "gray"),
            markersize=10,
            capsize=4,
            label=f"{cs} ({r['setting']})",
        )
        ax.annotate(
            r["setting"].replace("_", " "),
            (r[f"{x_col}_mean"], r["auc_mean"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
            color="gray",
        )
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("AUC (mean ± std over seeds)")
    ax.set_title(f"SUSQA Pareto frontier — AUC vs {x_label}")
    ax.grid(True, which="both", alpha=0.3)
    # Custom legend: one entry per case study
    handles = []
    seen = set()
    for _, r in agg.iterrows():
        cs = r["case_study"]
        if cs in seen:
            continue
        seen.add(cs)
        handles.append(
            plt.Line2D(
                [0], [0],
                marker=CASE_MARKERS.get(cs, "x"),
                color=CASE_COLORS.get(cs, "gray"),
                linestyle="",
                markersize=10,
                label=cs,
            )
        )
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path}")


def auc_by_setting_plot(agg: pd.DataFrame, out_path: Path):
    """Bar chart: AUC for each (case_study, setting) cell."""
    pivot = agg.pivot_table(
        index="setting", columns="case_study", values="auc_mean"
    )
    err = agg.pivot_table(
        index="setting", columns="case_study", values="auc_std"
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    pivot.plot(
        kind="bar",
        yerr=err,
        ax=ax,
        capsize=3,
        color=[CASE_COLORS.get(c, "gray") for c in pivot.columns],
        edgecolor="black",
        rot=20,
    )
    ax.set_ylabel("AUC (mean ± std)")
    ax.set_title("AUC by case study and setting")
    ax.set_ylim(0.5, 0.9)
    ax.legend(title="case study", loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading SUSQA reports from {RESULTS_DIR}")
    df = load_reports(RESULTS_DIR)
    # Drop smoke runs (they're not part of paper-quality results).
    df = df[df["setting"] != "smoke"].copy()
    print(f"  loaded {len(df)} reports across {df['case_study'].nunique()} case studies")

    agg = aggregate(df)
    table = comparison_table(agg)
    stats = run_stats(df)

    table_path = FIG_DIR / "comparison_table.csv"
    stats_path = FIG_DIR / "stats_tests.csv"
    table.to_csv(table_path, index=False)
    stats.to_csv(stats_path, index=False)
    print(f"  wrote {table_path}")
    print(f"  wrote {stats_path}")

    print("\n=== Headline SUSQA comparison table ===")
    print(table.to_string(index=False))

    print("\n=== Paired Wilcoxon (AUC) ===")
    if not stats.empty:
        print(stats.to_string(index=False))
    else:
        print("  not enough seeds in any cell yet for a paired test")

    pareto_plot(agg, "P", "Trainable parameters (P)", FIG_DIR / "pareto_auc_vs_params.png")
    pareto_plot(agg, "C_total_MB", "Communication (MB)", FIG_DIR / "pareto_auc_vs_comm.png")
    auc_by_setting_plot(agg, FIG_DIR / "auc_by_setting.png")

    print(f"\nDone. Figures and tables in {FIG_DIR}")


if __name__ == "__main__":
    main()
