from __future__ import annotations

import torch
import torch.nn as nn

from train.ppo_recurrent import metrics as metrics_lib
from train.ppo_recurrent.losses import action_logprob_and_entropy, compute_ppo_loss
from train.recurrent_common import next_update_sampling_state


def update_ppo(trainer, rollout) -> dict[str, float]:
    cfg = trainer.config
    last_value = getattr(rollout, "last_value", torch.zeros(cfg.num_envs, device=rollout.device))
    last_done = getattr(rollout, "last_done", torch.zeros(cfg.num_envs, device=rollout.device))
    rollout.compute_gae(last_values=last_value, last_dones=last_done)

    if cfg.value_normalization:
        with torch.no_grad():
            ret_mean = float(rollout.returns.mean().item())
            ret_std = float(rollout.returns.std(unbiased=False).clamp(min=1e-6).item())
    else:
        ret_mean, ret_std = 0.0, 1.0

    sampling_state = next_update_sampling_state(trainer.seed, trainer._update_counter)
    mb_seed = sampling_state.minibatch_seed
    metrics_sum = metrics_lib.init_metrics_sum()
    total_valid = 0.0
    num_minibatches = 0

    for _epoch in range(cfg.num_epochs):
        gen = torch.Generator()
        gen.manual_seed(mb_seed)
        for batch in rollout.iter_episode_minibatches(
            minibatch_size=cfg.minibatch_size, generator=gen
        ):
            mb_stats, n_valid = ppo_minibatch_step(
                trainer, batch, return_mean=ret_mean, return_std=ret_std
            )
            metrics_lib.accumulate(metrics_sum, mb_stats, n_valid)
            if n_valid > 0:
                total_valid += n_valid
            num_minibatches += 1

    metrics = metrics_lib.reduce_metrics(
        metrics_sum,
        total_valid=total_valid,
        num_minibatches=num_minibatches,
        lr=trainer.current_learning_rate,
    )
    metrics_lib.add_post_update_diagnostics(metrics, rollout=rollout, model=trainer.model)
    trainer._update_counter = sampling_state.update_counter
    return metrics


def ppo_minibatch_step(
    trainer, batch: dict[str, torch.Tensor], *, return_mean: float, return_std: float
):
    cfg = trainer.config
    if hasattr(trainer, "_training_h_init_log"):
        trainer._training_h_init_log.append(batch["h_init"].detach().clone())

    obs = batch["obs"]
    action = batch["action"]
    old_logprob = batch["old_logprob"]
    advantage = batch["advantage"]
    return_ = batch["return_"]
    old_value = batch["old_value"]
    h_init = batch["h_init"]
    valid_mask = batch["valid_mask"]

    n_valid = float(valid_mask.sum().item())
    if n_valid <= 0.0:
        return ({k: 0.0 for k in metrics_lib.init_metrics_sum()}, 0.0)

    h = h_init
    new_logprobs, entropies, values = [], [], []
    for t in range(valid_mask.shape[1]):
        outputs = trainer.model.policy_outputs(obs[:, t], h)
        h = outputs.h_next
        logp_t, ent_t = action_logprob_and_entropy(
            outputs.continuous_mean, outputs.continuous_log_std, outputs.binary_logits, action[:, t]
        )
        new_logprobs.append(logp_t)
        entropies.append(ent_t)
        values.append(outputs.value)

    new_logprob = torch.stack(new_logprobs, dim=1)
    entropy = torch.stack(entropies, dim=1)
    value = torch.stack(values, dim=1)

    loss = compute_ppo_loss(
        new_logprob=new_logprob,
        old_logprob=old_logprob,
        advantage=advantage,
        value=value,
        old_value=old_value,
        return_=return_,
        valid_mask=valid_mask,
        clip_ratio=cfg.clip_ratio,
        value_clip_ratio=cfg.value_clip_ratio,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        entropy=entropy,
        return_mean=return_mean,
        return_std=return_std,
    )

    trainer.optimizer.zero_grad()
    loss.total_loss.backward()
    actor_grad_norm = trainer._group_grad_norm(trainer._actor_params)
    critic_grad_norm = trainer._group_grad_norm(trainer._critic_params)
    trunk_grad_norm = trainer._group_grad_norm(trainer._trunk_params)
    nn.utils.clip_grad_norm_(trainer.model.parameters(), cfg.max_grad_norm)
    trainer.optimizer.step()

    scalar_keys = (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "total_loss",
    )
    scalars = torch.stack(
        [
            loss.policy_loss,
            loss.value_loss,
            loss.entropy,
            loss.approx_kl,
            loss.clip_fraction,
            loss.total_loss,
        ]
    ).detach().cpu().tolist()
    out = {k: float(v) for k, v in zip(scalar_keys, scalars, strict=False)}
    out["actor_grad_norm"] = actor_grad_norm
    out["critic_grad_norm"] = critic_grad_norm
    out["trunk_grad_norm"] = trunk_grad_norm
    return (out, n_valid)
