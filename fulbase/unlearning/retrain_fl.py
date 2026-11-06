from __future__ import annotations

from fulbase.unlearning.base import BaseUnlearner, UnlearningResult


class RetrainFLUnlearner(BaseUnlearner):
    name = "retrain_fl"

    def unlearn(self, ctx):
        from fulbase.flower.serial import run_fedavg
        cfg = ctx.cfg
        # Partial-participation counterfactual schedules are not calibrated yet.
        if cfg.fraction_fit != 1.0:
            raise ValueError("Strict retrain currently requires full participation")
        result = run_fedavg(
            backend=getattr(cfg, "backend", "flower"), fd=ctx.fd, dataset=ctx.dataset,
            num_clients=cfg.num_clients, num_rounds=cfg.num_rounds, fraction_fit=1.0,
            fraction_evaluate=0.0, min_fit_clients=len(ctx.retain_client_ids),
            min_evaluate_clients=1, batch_size=ctx.batch_size, local_epochs=cfg.local_epochs,
            lr=cfg.lr, client_device=cfg.client_device, client_cpus=cfg.client_cpus,
            client_gpus=cfg.client_gpus, initial_model_parameters=ctx.initial_parameters,
            active_client_ids=ctx.retain_client_ids, forget_client_ids=ctx.forget_client_ids,
            marker_config=ctx.marker_config, training_seed=cfg.seed, record_history=False)
        return UnlearningResult(self.name, result.final_parameters, {
            "backend": getattr(cfg, "backend", "flower"), "num_rounds": cfg.num_rounds,
            "active_client_ids": ctx.retain_client_ids,
            "initialization": "identical_full_initial_checkpoint",
            "budget": "same_rounds_epochs_lr_full_retained_participation",
            "input_scope": "retained_real_data_and_retained_training_markers"})
