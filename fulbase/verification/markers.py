from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MarkerConfig:
    enabled: bool = False
    seed: int = 1729  # public structure/anchor seed, not an audit-sample secret
    families_per_class: int = 4
    variants_per_family: int = 2
    batch_size: int = 32
    ce_weight: float = 0.2
    margin_weight: float = 0.1
    geometry_weight: float = 0.05
    target_margin: float = 1.0
    noise_std: float = 0.03
    version: str = "synthetic-texture-v1"

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.ce_weight, self.margin_weight,
                   self.geometry_weight, self.target_margin, self.noise_std)):
            raise ValueError("Marker configuration must be finite")
        if min(self.families_per_class, self.variants_per_family, self.batch_size) < 1:
            raise ValueError("Marker sizes must be positive")
        if min(self.ce_weight, self.margin_weight, self.geometry_weight,
               self.target_margin, self.noise_std) < 0:
            raise ValueError("Marker weights, margin and noise must be nonnegative")
        if self.version != "synthetic-texture-v1":
            raise ValueError("Unsupported marker generator version")

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def derived_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32)


class MarkerDataset(Dataset):
    """Balanced synthetic markers. Family is the resampling unit, not each variant.

    Training/selection/audit sample seeds are separate from the shared structure
    seed. A caller in training only receives the training dataset/seed.
    """

    def __init__(self, cfg: MarkerConfig, cid: int, dataset: str, role: str,
                 sample_seed: int, num_classes: int = 10):
        if role not in {"train", "calibration", "audit"}:
            raise ValueError("Unknown marker role")
        if cid < 0 or num_classes < 2:
            raise ValueError("Invalid client or class count")
        self.cfg, self.cid, self.role = cfg, cid, role
        self.dataset, self.sample_seed, self.num_classes = dataset, sample_seed, num_classes
        self.channels, self.size = (3, 32) if dataset == "cifar10" else (1, 28)
        if dataset not in {"cifar10", "fashionmnist", "mnist"}:
            raise ValueError("Unsupported marker input dataset")
        self.ids = []
        images, labels, self.families = [], [], []
        sample_token = derived_seed(cfg.fingerprint(), role, sample_seed)
        for label in range(num_classes):
            for family in range(cfg.families_per_class):
                # Each role has independent base families, with common learnable
                # client/class motifs. Variants share the base and stay clustered.
                family_id = f"{cid}:{role}:{sample_token}:{label}:{family}"
                base = self._base(label, family)
                for variant in range(cfg.variants_per_family):
                    rng = np.random.default_rng(derived_seed(sample_seed, role, cid, label, family, variant))
                    img = np.clip(base + rng.normal(0, cfg.noise_std, base.shape), 0, 1)
                    images.append(self._normalize(torch.tensor(img, dtype=torch.float32)))
                    labels.append(label)
                    self.families.append(family_id)
                    self.ids.append(f"{family_id}:{variant}")
        self.images, self.labels = torch.stack(images), torch.tensor(labels, dtype=torch.long)

    def _base(self, label, family):
        structure = np.random.default_rng(derived_seed(self.cfg.seed, self.cid, label))
        coarse = torch.tensor(structure.uniform(0.1, 0.9, (1, self.channels, 6, 6)), dtype=torch.float32)
        img = F.interpolate(coarse, size=(self.size, self.size), mode="bilinear", align_corners=False)[0].numpy()
        rng = np.random.default_rng(derived_seed(self.sample_seed, self.role, self.cid, label, family, "base"))
        return np.clip(img + rng.normal(0, 0.025, img.shape), 0, 1)

    def _normalize(self, img):
        if self.dataset == "cifar10":
            mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        elif self.dataset == "mnist":
            mean, std = (0.1307,), (0.3081,)
        else:
            mean, std = (0.5,), (0.5,)
        return (img - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]

    def manifest(self):
        return [{"cid": self.cid, "role": self.role, "marker_id": mid,
                 "family_id": fid, "label": int(label)}
                for mid, fid, label in zip(self.ids, self.families, self.labels)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def client_anchor(cfg: MarkerConfig, cid: int, dim: int, device):
    # Positive orthant also supports SmallCNN's ReLU features. Public references
    # approximate inter-client separation, not the original cross-client loss.
    rng = np.random.default_rng(derived_seed(cfg.seed, cid, "anchor"))
    anchor = torch.tensor(rng.uniform(0, 1, dim), dtype=torch.float32, device=device)
    return F.normalize(anchor, dim=0)


def marker_loss(model, x, y, cfg: MarkerConfig, cid: int, components: dict | None = None):
    z = model.features(x)
    logits = model.classifier(z)
    correct = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.masked_fill(F.one_hot(y, logits.shape[1]).bool(), -torch.inf).max(1).values
    ce = F.cross_entropy(logits, y)
    margin = F.relu(cfg.target_margin - (correct - other)).mean()
    center = F.normalize(F.normalize(z, dim=1).mean(0), dim=0)
    geometry = (center - client_anchor(cfg, cid, z.shape[1], z.device)).square().sum()
    if components is not None:
        components.update(marker_ce=float(ce.detach()), marker_margin=float(margin.detach()),
                          marker_geometry=float(geometry.detach()))
    return cfg.ce_weight * ce + cfg.margin_weight * margin + cfg.geometry_weight * geometry
