from __future__ import annotations

import torch
import torch.nn as nn

from train.ppo_recurrent.losses import _masked_mean, action_logprob_and_entropy
from train.ppo_recurrent import metrics as metrics_lib


def update_ppo(trainer, rollout) -> dict[str, float]:
    cfg = trainer.config
    last_value = getattr(rollout, "last_value", torch.zeros(cfg.num_envs))
    last_done = getattr(rollout, "last_done", torch.zeros(cfg.num_envs))
    rollout.compute_gae(last_values=last_value, last_dones=last_done)

    if cfg.value_normalization:
        with torch.no_grad():
            ret_mean = float(rollout.returns.mean().item())
            ret_std = float(rollout.returns.std(unbiased=False).clamp(min=1e-6).item())
    else:
        ret_mean, ret_std = 0.0, 1.0

    mb_seed = trainer.seed * 1_000_003 + (trainer._update_counter + 1)
    metrics_sum = metrics_lib.init_metrics_sum()
    total_valid = 0.0
    num_minibatches = 0

    for _epoch in range(cfg.num_epochs):
        gen = torch.Generator()
        gen.manual_seed(mb_seed)
        for batch in rollout.iter_episode_minibatches(minibatch_size=cfg.minibatch_size, generator=gen):
            mb_stats, n_valid = ppo_minibatch_step(trainer, batch, return_mean=ret_mean, return_std=ret_std)
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
    trainer._update_counter += 1
    return metrics


def ppo_minibatch_step(trainer, batch: dict[str, torch.Tensor], *, return_mean: float, return_std: float):
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
        return ({k: 0.0 for k in metrics_lib.init_metrics_sum().keys()}, 0.0)

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

    adv_mean = _masked_mean(advantage, valid_mask)
    adv_var = _masked_mean((advantage - adv_mean) ** 2, valid_mask)
    norm_adv = (advantage - adv_mean) / adv_var.clamp(min=1e-8).sqrt()

    ratio = (new_logprob - old_logprob).exp()
    pg1 = ratio * norm_adv
    pg2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * norm_adv
    policy_loss = _masked_mean(-torch.min(pg1, pg2), valid_mask)

    value_n = (value - return_mean) / return_std
    old_value_n = (old_value - return_mean) / return_std
    return_n = (return_ - return_mean) / return_std
    value_clipped_n = old_value_n + torch.clamp(value_n - old_value_n, -cfg.value_clip_ratio, cfg.value_clip_ratio)
    vl_unclipped = (value_n - return_n) ** 2
    vl_clipped = (value_clipped_n - return_n) ** 2
    value_loss = _masked_mean(0.5 * torch.max(vl_unclipped, vl_clipped), valid_mask)

    entropy_mean = _masked_mean(entropy, valid_mask)
    total_loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean

    trainer.optimizer.zero_grad()
    total_loss.backward()
    actor_grad_norm = trainer._group_grad_norm(trainer._actor_params)
    critic_grad_norm = trainer._group_grad_norm(trainer._critic_params)
    trunk_grad_norm = trainer._group_grad_norm(trainer._trunk_params)
    nn.utils.clip_grad_norm_(trainer.model.parameters(), cfg.max_grad_norm)
    trainer.optimizer.step()

    with torch.no_grad():
        approx_kl = _masked_mean(old_logprob - new_logprob, valid_mask)
        clip_fraction = _masked_mean(((ratio - 1.0).abs() > cfg.clip_ratio).float(), valid_mask)

    return ({
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy_mean.item()),
        "approx_kl": float(approx_kl.item()),
        "clip_fraction": float(clip_fraction.item()),
        "total_loss": float(total_loss.item()),
        "actor_grad_norm": actor_grad_norm,
        "critic_grad_norm": critic_grad_norm,
        "trunk_grad_norm": trunk_grad_norm,
    }, n_valid)
