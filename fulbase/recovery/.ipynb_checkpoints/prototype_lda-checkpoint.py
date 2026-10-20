from __future__ import annotations

import numpy as np
import torch

from fulbase.dp.prototype import PrototypeBank
from fulbase.models import build_model
from fulbase.utils import get_model_parameters, set_model_parameters


def recover_with_prototype_lda(
    unlearned_parameters: list[np.ndarray],
    proto: PrototypeBank,
    dataset: str,
    keep_classes: list[int] | None = None,
    ridge: float = 1e-2,
    closed_class_bias: float = -1e4,
) -> list[np.ndarray]:
    """Rebuild only the classifier head with diagonal LDA.

    LDA discriminant:
        g_c(z)= z^T Sigma^{-1} mu_c - 1/2 mu_c^T Sigma^{-1} mu_c + log pi_c
    """
    model = build_model(dataset=dataset, num_classes=proto.num_classes)
    set_model_parameters(model, unlearned_parameters)

    keep = list(range(proto.num_classes)) if keep_classes is None else keep_classes
    mean = proto.mean.astype(np.float64)
    var = proto.var.astype(np.float64)
    count = proto.count.astype(np.float64)
    shared_var = np.average(var[keep], axis=0, weights=np.maximum(count[keep], 1.0))
    shared_var = np.maximum(shared_var, ridge)

    weight = np.zeros_like(mean, dtype=np.float64)
    bias = np.ones((proto.num_classes,), dtype=np.float64) * closed_class_bias
    total_count = np.maximum(count[keep].sum(), 1.0)
    for c in keep:
        inv = 1.0 / shared_var
        mu = mean[c]
        prior = max(count[c] / total_count, 1e-12)
        weight[c] = inv * mu
        bias[c] = -0.5 * float((mu * inv * mu).sum()) + np.log(prior)

    with torch.no_grad():
        model.classifier.weight.copy_(torch.tensor(weight, dtype=model.classifier.weight.dtype))
        model.classifier.bias.copy_(torch.tensor(bias, dtype=model.classifier.bias.dtype))
    return get_model_parameters(model)
