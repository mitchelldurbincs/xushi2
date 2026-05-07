from __future__ import annotations

import numpy as np
import torch

from train.rollout_buffer import RolloutBuffer


def collect_rollout(trainer) -> RolloutBuffer:
    """Collect a rollout using trainer-owned model/env/state.

    Keeps behavior identical to legacy ``PPOTrainer.collect_rollout``.
    """
    cfg = trainer.config
    buf = trainer._make_buffer()

    last_obs = trainer._last_obs
    h = trainer.h

    for tick in range(cfg.rollout_len):
        with torch.no_grad():
            prev_rng = torch.get_rng_state()
            torch.set_rng_state(trainer._sampling_rng_state)
            try:
                action, logprob, h_next = trainer.model.sample_action(last_obs, h)
                trainer._sampling_rng_state = torch.get_rng_state()
            finally:
                torch.set_rng_state(prev_rng)
            _, _, value, _ = trainer.model.forward(last_obs, h)

        h_init_to_write = h
        action_np = action.detach().cpu().numpy()
        next_obs, reward, terminated, truncated, _info = trainer.envs.step(action_np)
        done_np = np.logical_or(terminated, truncated)

        reward_t = torch.as_tensor(reward, dtype=torch.float32)
        done_t = torch.as_tensor(done_np, dtype=torch.float32)

        buf.add(
            tick=tick,
            obs=last_obs,
            action=action,
            logprob=logprob,
            reward=reward_t,
            value=value,
            done=done_t,
            h_init=h_init_to_write,
        )

        h = h_next
        if bool(done_np.any()):
            for e in range(cfg.num_envs):
                if bool(done_np[e]):
                    h[e] = 0.0
                    buf.mark_reset(e)

        last_obs = torch.as_tensor(next_obs, dtype=torch.float32)

    with torch.no_grad():
        _, _, last_value, _ = trainer.model.forward(last_obs, h)
    buf.last_value = last_value
    buf.last_done = buf.done[:, -1].clone()

    trainer._last_obs = last_obs
    trainer.h = h
    return buf
