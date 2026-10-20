from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class FederatedData:
    trainset: Dataset
    testset: Dataset
    train_indices: list[list[int]]
    test_indices: list[list[int]]
    num_classes: int
    in_channels: int
    image_size: int


def _targets(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    raise ValueError("Dataset has no targets attribute")


def dirichlet_partition(
    targets: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float,
    seed: int,
    min_size: int = 8,
) -> list[list[int]]:
    """Dirichlet label-skew partition.

    alpha 越小，非 IID 越强。为了避免某些客户端没有样本，循环到每个客户端至少 min_size。
    """
    rng = np.random.default_rng(seed)
    n = len(targets)
    while True:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            rng.shuffle(idx_k)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            # 避免已经很多样本的客户端继续拿太多
            proportions = np.array(
                [p * (len(idx_j) < n / num_clients * 1.5) for p, idx_j in zip(proportions, idx_batch)]
            )
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions)[:-1] * len(idx_k)).astype(int)
            for cid, idx in enumerate(np.split(idx_k, split_points)):
                idx_batch[cid].extend(idx.tolist())
        sizes = [len(x) for x in idx_batch]
        if min(sizes) >= min_size:
            for idx in idx_batch:
                rng.shuffle(idx)
            return idx_batch


def iid_partition(targets: np.ndarray, num_clients: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(targets))
    rng.shuffle(indices)
    return [x.tolist() for x in np.array_split(indices, num_clients)]


def load_federated_data(
    dataset: Literal["fashionmnist", "mnist", "cifar10"] = "fashionmnist",
    data_dir: str | Path = "./data",
    num_clients: int = 10,
    partition: Literal["dirichlet", "iid"] = "dirichlet",
    alpha: float = 0.5,
    seed: int = 42,
) -> FederatedData:
    dataset = dataset.lower()
    data_dir = Path(data_dir)
    if dataset == "fashionmnist":
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tfm)
        testset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tfm)
        num_classes, in_channels, image_size = 10, 1, 28
    elif dataset == "mnist":
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainset = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
        testset = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
        num_classes, in_channels, image_size = 10, 1, 28
    elif dataset == "cifar10":
        tfm_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        trainset = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm_train)
        testset = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm_test)
        num_classes, in_channels, image_size = 10, 3, 32
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_targets = _targets(trainset)
    test_targets = _targets(testset)
    if partition == "iid":
        train_indices = iid_partition(train_targets, num_clients, seed)
        test_indices = iid_partition(test_targets, num_clients, seed + 999)
    else:
        train_indices = dirichlet_partition(train_targets, num_clients, num_classes, alpha, seed)
        test_indices = dirichlet_partition(test_targets, num_clients, num_classes, alpha, seed + 999)

    return FederatedData(
        trainset=trainset,
        testset=testset,
        train_indices=train_indices,
        test_indices=test_indices,
        num_classes=num_classes,
        in_channels=in_channels,
        image_size=image_size,
    )


def make_client_loaders(fd: FederatedData, cid: int, batch_size: int, num_workers: int = 0):
    trainloader = DataLoader(
        Subset(fd.trainset, fd.train_indices[cid]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    testloader = DataLoader(
        Subset(fd.testset, fd.test_indices[cid]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return trainloader, testloader


def make_merged_loader(
    dataset: Dataset,
    indices_list: list[list[int]],
    client_ids: list[int],
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    indices: list[int] = []
    for cid in client_ids:
        indices.extend(indices_list[cid])
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
