from __future__ import annotations

import flwr as fl
import torch

from fulbase.data import FederatedData, make_client_loaders
from fulbase.models import build_model
from fulbase.train import evaluate, train_one_client
from fulbase.utils import get_model_parameters, set_model_parameters


class FULClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        fd: FederatedData,
        dataset: str,
        batch_size: int,
        local_epochs: int,
        lr: float,
        device: str,
    ) -> None:
        self.cid = int(cid)
        self.dataset = dataset
        self.local_epochs = int(local_epochs)
        self.lr = float(lr)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.trainloader, self.testloader = make_client_loaders(fd, self.cid, batch_size=batch_size)
        self.model = build_model(dataset=dataset, num_classes=fd.num_classes)

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        lr = float(config.get("lr", self.lr))
        epochs = int(config.get("local_epochs", self.local_epochs))
        loss = train_one_client(self.model, self.trainloader, epochs=epochs, lr=lr, device=self.device)
        return get_model_parameters(self.model), len(self.trainloader.dataset), {
            "train_loss": float(loss),
            "cid": str(self.cid),
        }

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        metrics = evaluate(self.model, self.testloader, self.device)
        return float(metrics["loss"]), int(metrics["num_examples"]), {
            "eval_acc": float(metrics["acc"]),
            "cid": str(self.cid),
        }


def make_client_fn(
    fd: FederatedData,
    dataset: str,
    batch_size: int,
    local_epochs: int,
    lr: float,
    device: str,
):
    def client_fn(cid: str):
        client = FULClient(
            cid=int(cid),
            fd=fd,
            dataset=dataset,
            batch_size=batch_size,
            local_epochs=local_epochs,
            lr=lr,
            device=device,
        )
        return client.to_client()

    return client_fn
