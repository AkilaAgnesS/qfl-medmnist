"""Flower server entrypoint — minimal FedAvg setup, override strategy via config."""
from __future__ import annotations

import flwr as fl


def start_server(num_rounds: int = 10, min_clients: int = 5, address: str = "0.0.0.0:8080"):
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )
    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server()
