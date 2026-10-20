from __future__ import annotations

import torch

from fulbase.data import make_merged_loader
from fulbase.models import build_model
from fulbase.train import train_one_client
from fulbase.unlearning.base import BaseUnlearner, UnlearningContext, UnlearningResult
from fulbase.utils import get_model_parameters, set_model_parameters


class RetrainOracleUnlearner(BaseUnlearner):
    name = "retrain_oracle"

    def unlearn(self, ctx: UnlearningContext) -> UnlearningResult:
        """保留数据重训参考上界。

        为了先把实验闭环跑通，这里使用“中心化 retained data 重训”的轻量 oracle，
        从初始模型参数开始，只在保留客户端训练集上训练 retrain_epochs 轮。
        后续如果要更严格，可以改成 retained clients 的 Flower FedAvg 重训。
        """
        retrain_epochs = int(getattr(ctx.cfg, "retrain_epochs", 5))
        lr_value = getattr(ctx.cfg, "retrain_lr", None)
        if lr_value is None:
            lr_value = getattr(ctx.cfg, "lr", 0.01)
        lr = float(lr_value)

        model = build_model(dataset=ctx.dataset, num_classes=ctx.num_classes)
        set_model_parameters(model, ctx.initial_parameters)
        loader = make_merged_loader(
            ctx.fd.trainset,
            ctx.fd.train_indices,
            ctx.retain_client_ids,
            batch_size=ctx.batch_size,
            shuffle=True,
        )
        loss = train_one_client(
            model=model,
            trainloader=loader,
            epochs=retrain_epochs,
            lr=lr,
            device=ctx.device,
        )
        return UnlearningResult(
            name=self.name,
            parameters=get_model_parameters(model),
            logs={
                "note": "centralized retained-data retrain oracle; replace with FL retrain for stricter comparison",
                "retrain_epochs": retrain_epochs,
                "retrain_lr": lr,
                "last_train_loss": float(loss),
            },
        )
