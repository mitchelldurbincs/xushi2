from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from train.mappo import MappoActorCritic, MappoConfig
from train.ppo_recurrent.orchestration import make_env_fn
from train.runtime_specs import resolve_runtime_spec

from .formatting import (
    action_to_fields,
    format_decision,
    format_decision_six,
    policy_action_to_world_fields,
)
from .header import header_fields


def load_phase4_checkpoint(path: str | Path) -> tuple[MappoActorCritic, dict]:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint at {path} must be a dict, got {type(ckpt)!r}")
    ckpt_config = ckpt.get("config", {})
    cfg = MappoConfig(**ckpt_config["mappo"])
    model = MappoActorCritic(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt_config


def dump_phase3(
    model,
    ckpt_config: dict,
    *,
    seed: int,
    episodes: int,
    max_decisions: int | None,
    output_path: Path,
) -> int:
    train_config = {
        "phase": int(ckpt_config.get("phase", 3)),
        "env": ckpt_config["env"],
    }
    env_fn, _env_meta, _seed_base = make_env_fn(train_config)
    header = header_fields(ckpt_config, seed=seed)

    n_decisions = 0
    with output_path.open("w", encoding="ascii") as f:
        f.write(" ".join(f"{k}={v}" for k, v in header.items()) + "\n")

        for ep in range(int(episodes)):
            env = env_fn()
            try:
                obs, info = env.reset(seed=int(seed) + ep)
                h = model.init_hidden(batch_size=1)
                done = False
                tick = int(info.get("tick", 0))
                while not done:
                    if max_decisions is not None and n_decisions >= max_decisions:
                        return n_decisions
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).view(1, -1)
                    with torch.no_grad():
                        action_t, h = model.greedy_action(obs_t, h)
                    action = action_t.squeeze(0).cpu().numpy()
                    learner_fields = action_to_fields(action)
                    obs, _r, term, trunc, info = env.step(action)
                    opp = info["opponent_action"]
                    opponent_fields = [
                        float(opp["move_x"]),
                        float(opp["move_y"]),
                        float(opp["aim_delta"]),
                        float(opp["primary_fire"]),
                        float(opp["ability_1"]),
                        float(opp["ability_2"]),
                    ]
                    f.write(format_decision(tick, learner_fields, opponent_fields))
                    f.write("\n")
                    n_decisions += 1
                    tick = int(info.get("tick", tick + int(header["action_repeat"])))
                    done = bool(term or trunc)
            finally:
                env.close()
    return n_decisions


def dump_mappo(
    model: MappoActorCritic,
    ckpt_config: dict,
    *,
    seed: int,
    episodes: int,
    max_decisions: int | None,
    output_path: Path,
    stochastic: bool = False,
) -> int:
    raw_phase = ckpt_config.get("phase", 4)
    try:
        wanted_phase = int(str(raw_phase).removeprefix("phase"))
    except ValueError:
        wanted_phase = 4
    if wanted_phase != 11 and ckpt_config["env"].get("learner_team", "A") != "A":
        raise ValueError("MAPPO replay dumping currently supports learner_team='A'")

    runtime = resolve_runtime_spec({"phase": raw_phase, "env": ckpt_config["env"]})
    if runtime.learner.kind != "mappo" or runtime.env_fn is None:
        raise ValueError(
            "MAPPO replay dumping requires a MAPPO runtime, "
            f"got learner={runtime.learner.kind!r} env={runtime.env.kind!r}"
        )
    env_fn = runtime.env_fn
    header = header_fields(ckpt_config, seed=seed)
    include_target = int(model.cfg.target_action_dim) > 0
    zero_slot = [0.0] * (7 if include_target else 6)

    n_decisions = 0
    with output_path.open("w", encoding="ascii") as f:
        f.write(" ".join(f"{k}={v}" for k, v in header.items()) + "\n")

        for ep in range(int(episodes)):
            env = env_fn()
            try:
                obs, info = env.reset(seed=int(seed) + ep)
                h = model.init_hidden(model.cfg.n_agents)
                done = False
                tick = int(info.get("tick", 0))
                while not done:
                    if max_decisions is not None and n_decisions >= max_decisions:
                        return n_decisions
                    obs_t = torch.as_tensor(obs, dtype=torch.float32)
                    with torch.no_grad():
                        if stochastic:
                            action_t, _logprob, h = model.sample_action(obs_t, h)
                        else:
                            action_t, h = model.greedy_action(obs_t, h)
                    action = action_t.cpu().numpy()
                    policy_slots = [
                        policy_action_to_world_fields(
                            action[i],
                            slot=i,
                            include_target=include_target,
                        )
                        for i in range(model.cfg.n_agents)
                    ]
                    if wanted_phase == 11:
                        if len(policy_slots) != 6:
                            raise ValueError("phase11 replay dump requires six policy action slots")
                        obs, _reward, term, trunc, info = env.step(action)
                        if str(info.get("match_type", "current")) == "current":
                            slots = policy_slots
                        else:
                            opponent_actions = np.asarray(
                                info.get("opponent_actions"), dtype=np.float32
                            )
                            if opponent_actions.shape != (3, 6):
                                raise ValueError(
                                    "phase11 league replay dump requires opponent_actions info"
                                )
                            opponent_slots = [
                                action_to_fields(opponent_actions[i]) for i in range(3)
                            ]
                            slots = policy_slots[:3] + opponent_slots
                    else:
                        obs, _reward, term, trunc, info = env.step(action)
                        self_play_enabled = bool(
                            dict(ckpt_config["env"].get("self_play", {})).get(
                                "enabled", False
                            )
                        )
                        if self_play_enabled:
                            if len(policy_slots) != 6:
                                raise ValueError(
                                    "phase4 self-play replay dump requires six policy slots"
                                )
                            slots = policy_slots
                            opponent_actions_raw = None
                        else:
                            opponent_actions_raw = info.get("opponent_actions")
                        if opponent_actions_raw is None and not self_play_enabled:
                            slots = [*policy_slots[:3], zero_slot, zero_slot, zero_slot]
                        elif opponent_actions_raw is not None:
                            opponent_actions = np.asarray(opponent_actions_raw, dtype=np.float32)
                            if opponent_actions.shape != (3, 6):
                                raise ValueError(
                                    "phase>=4 replay dump expects opponent_actions of shape (3, 6)"
                                )
                            opponent_slots = [
                                action_to_fields(opponent_actions[i], include_target=include_target)
                                for i in range(3)
                            ]
                            slots = policy_slots[:3] + opponent_slots
                    f.write(format_decision_six(tick, slots))
                    f.write("\n")
                    n_decisions += 1
                    tick = int(info.get("tick", tick + int(header["action_repeat"])))
                    done = bool(term or trunc)
            finally:
                env.close()
    return n_decisions
