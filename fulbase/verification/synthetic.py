"""Versioned in-memory functional inputs, never benchmark measurements."""
from __future__ import annotations

GENERATOR_KEY = "verification.synthetic_data.v1"


def synthetic_data(num_clients=3, seed=42):
    import torch
    from torch.utils.data import TensorDataset
    from fulbase.data import FederatedData
    gen = torch.Generator().manual_seed(seed)

    def make(per_client):
        images, labels, indices = [], [], []
        for cid in range(num_clients):
            start = len(images)
            for k in range(per_client):
                label = k % 10
                image = torch.randn(1, 28, 28, generator=gen) * 0.1
                image[:, 2*label:2*label+3, :] += 0.5
                images.append(image)
                labels.append(label)
            indices.append(list(range(start, len(images))))
        return TensorDataset(torch.stack(images), torch.tensor(labels)), indices

    train, train_indices = make(20)
    test, test_indices = make(10)
    return FederatedData(train, test, train_indices, test_indices, 10, 1, 28)
