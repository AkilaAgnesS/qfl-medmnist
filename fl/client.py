"""Flower client wrapping a PyTorch model."""
from __future__ import annotations

from collections import OrderedDict

import flwr as fl
import torch
from torch.utils.data import DataLoader


class TorchClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        local_epochs: int = 1,
        lr: float = 1e-3,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device).long().flatten()
                opt.zero_grad()
                loss = loss_fn(self.model(x), y)
                loss.backward()
                opt.step()
        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device).long().flatten()
                logits = self.model(x)
                loss += loss_fn(logits, y).item()
                correct += (logits.argmax(1) == y).sum().item()
                n += y.size(0)
        return loss / n, n, {"accuracy": correct / n}
