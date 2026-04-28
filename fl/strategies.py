"""Aggregation strategies. FedAvg is provided by Flower; FedProx + quantum-aware variants live here."""
from __future__ import annotations

import flwr as fl

# Placeholder — implement FedProx (proximal term) and a quantum-aware aggregator
# (e.g., averaging in parameter space with circuit-symmetry awareness) once
# the baseline FedAvg pipeline is producing results.


def get_strategy(name: str, **kwargs):
    if name == "fedavg":
        return fl.server.strategy.FedAvg(**kwargs)
    raise NotImplementedError(f"Strategy {name!r} not yet implemented.")
