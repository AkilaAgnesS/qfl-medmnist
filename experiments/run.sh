#!/usr/bin/env bash
# Top-level driver — reproduces every table in the paper.
# Run from repo root.
set -euo pipefail

CONFIGS_DIR="experiments/configs"

echo "== Centralized baselines =="
python experiments/run_centralized.py --config "$CONFIGS_DIR/classical_breast.yaml"
# TODO add: classical_pneumonia, hybrid_breast, hybrid_pneumonia

echo "== Federated experiments =="
python experiments/run_federated.py --config "$CONFIGS_DIR/hybrid_pneumonia_fl.yaml"
# TODO add: classical_*_fl, hybrid_*_fl_noniid_*, ablations

echo "Done. Results in results/."
