"""Effect sizes, bootstrap confidence intervals, and TOST equivalence tests.

Reads SUSQA reports under results/ and emits paper-grade statistical depth
that the paired Wilcoxon table alone cannot produce at small n.

Outputs (results/figures/):
    effect_sizes.csv        - Cohen's d for every pairwise AUC comparison
    bootstrap_cis.csv       - 95% bootstrap CIs of AUC mean for every cell
    tost_equivalence.csv    - TOST results for the headline comparisons

Usage:
    python notebooks/06_effect_sizes.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)


def _ncdf(x: float) -> float:
    """Standard normal CDF without scipy dependency."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def parse_setting(eid: str) -> str:
    eid = eid.lower()
    if eid.startswith("baseline_"):
        return "centralized"
    if "_iid" in eid:
        return "fl_iid"
    if "dirichlet01" in eid:
        return "fl_dirichlet_alpha=0.1"
    if "dirichlet05" in eid:
        return "fl_dirichlet_alpha=0.5"
    return "unknown"


def load_per_seed():
    out = defaultdict(list)
    for path in sorted(RESULTS_DIR.glob("*/susqa_report.json")):
        d = json.loads(path.read_text())
        eid = d["experiment_id"]
        m = re.match(r"^(?P<base>.+)__seed(?P<s>\d+)$", eid)
        if not m:
            continue
        setting = parse_setting(m.group("base"))
        if setting in ("smoke", "unknown"):
            continue
        key = (d.get("dataset", ""), setting, d.get("case_study", ""))
        out[key].append(float(d.get("auc", float("nan"))))
    return {k: np.asarray(v) for k, v in out.items() if len(v) > 0}


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s_pool = float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0))
    if s_pool == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / s_pool)


def bootstrap_ci(values, n_boot=10_000, alpha=0.05):
    if len(values) < 2:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    samples = values[idx].mean(axis=1)
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def tost(a, b, eps=0.01, alpha=0.05):
    """Two One-Sided Tests for equivalence within +/- eps.

    Returns p-values for H01: mean(a) - mean(b) <= -eps and
                       H02: mean(a) - mean(b) >=  eps.
    Equivalence (within eps) is concluded if both are rejected at alpha.
    """
    if len(a) < 2 or len(b) < 2:
        return {"diff": float("nan"), "p_lower": float("nan"),
                "p_upper": float("nan"), "equivalent_at_5pct": False, "eps": eps}
    diff = float(a.mean() - b.mean())
    pooled_se = float(np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
    if pooled_se == 0:
        return {"diff": diff, "p_lower": 0.0 if diff > -eps else 1.0,
                "p_upper": 0.0 if diff < eps else 1.0,
                "equivalent_at_5pct": -eps < diff < eps, "eps": eps}
    z_lower = (diff - (-eps)) / pooled_se
    z_upper = (diff - (eps)) / pooled_se
    p_lower = 1.0 - _ncdf(z_lower)
    p_upper = _ncdf(z_upper)
    return {"diff": diff, "p_lower": float(p_lower), "p_upper": float(p_upper),
            "equivalent_at_5pct": (p_lower < alpha) and (p_upper < alpha),
            "eps": eps}


def main():
    per_seed = load_per_seed()
    if not per_seed:
        raise SystemExit("No SUSQA reports found.")

    # Bootstrap CIs
    rows = []
    for (ds, st, cs), vals in sorted(per_seed.items()):
        mean, lo, hi = bootstrap_ci(vals, n_boot=10_000)
        rows.append({
            "dataset": ds, "setting": st, "case_study": cs,
            "n": len(vals), "auc_mean": round(mean, 4),
            "boot_ci_low": round(lo, 4), "boot_ci_high": round(hi, 4),
            "ci_width": round(hi - lo, 4),
        })
    bcis = pd.DataFrame(rows)
    bcis_path = FIG_DIR / "bootstrap_cis.csv"
    bcis.to_csv(bcis_path, index=False)
    print(f"  wrote {bcis_path} ({len(bcis)} rows)")

    # Cohen's d for every pairwise AUC comparison
    rows = []
    keys_by_setting = defaultdict(list)
    for k in per_seed:
        keys_by_setting[(k[0], k[1])].append(k)
    for (ds, st), keys in sorted(keys_by_setting.items()):
        case_keys = sorted(keys, key=lambda x: x[2])
        for i, k_a in enumerate(case_keys):
            for k_b in case_keys[i + 1:]:
                a = per_seed[k_a]
                b = per_seed[k_b]
                d = cohens_d(a, b)
                rows.append({
                    "dataset": ds, "setting": st,
                    "comparison": f"{k_a[2]} vs {k_b[2]}",
                    "mean_a": round(float(a.mean()), 4),
                    "mean_b": round(float(b.mean()), 4),
                    "diff": round(float(a.mean() - b.mean()), 4),
                    "cohens_d": round(d, 3),
                    "n": min(len(a), len(b)),
                })
    eff = pd.DataFrame(rows)
    eff_path = FIG_DIR / "effect_sizes.csv"
    eff.to_csv(eff_path, index=False)
    print(f"  wrote {eff_path} ({len(eff)} rows)")

    # TOST equivalence on headline comparisons
    headline_keys = [
        ("pneumoniamnist", "fl_iid", "C1_classical", "C3_hybrid_quantum"),
        ("pneumoniamnist", "fl_iid", "C2_compressed", "C3_hybrid_quantum"),
        ("breastmnist",    "fl_dirichlet_alpha=0.5", "C1_classical", "C3_hybrid_quantum"),
        ("breastmnist",    "fl_dirichlet_alpha=0.5", "C2_compressed", "C3_hybrid_quantum"),
    ]
    rows = []
    for ds, st, ca, cb in headline_keys:
        a = per_seed.get((ds, st, ca))
        b = per_seed.get((ds, st, cb))
        if a is None or b is None:
            continue
        r = tost(b, a, eps=0.01)
        rows.append({
            "dataset": ds, "setting": st,
            "comparison": f"{cb} - {ca} (AUC)",
            "diff": round(r["diff"], 4),
            "eps": r["eps"],
            "p_lower": round(r["p_lower"], 3),
            "p_upper": round(r["p_upper"], 3),
            "equivalent_at_5pct": r["equivalent_at_5pct"],
        })
    tost_df = pd.DataFrame(rows)
    tost_path = FIG_DIR / "tost_equivalence.csv"
    tost_df.to_csv(tost_path, index=False)
    print(f"  wrote {tost_path} ({len(tost_df)} rows)")

    print("\n=== Cohen's d (sorted by |d| descending) ===")
    print(eff.assign(abs_d=eff["cohens_d"].abs())
            .sort_values("abs_d", ascending=False)
            .head(15)
            .drop(columns="abs_d")
            .to_string(index=False))

    print("\n=== Bootstrap 95% CIs (headline cells) ===")
    print(bcis[bcis["setting"].isin(["fl_iid", "fl_dirichlet_alpha=0.5"])]
            .sort_values(["dataset", "setting", "case_study"])
            .to_string(index=False))

    print("\n=== TOST equivalence (within 0.01 AUC) ===")
    print(tost_df.to_string(index=False))


if __name__ == "__main__":
    main()
