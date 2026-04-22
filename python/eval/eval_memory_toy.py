from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from envs.memory_toy import MemoryToyEnv
from train.models import build_model


@dataclass
class AblationResult:
    mode: str
    mean: float
    ci95: float
    n: int


def load_checkpoint(path: Path) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(path, map_location="cpu")
    cfg = payload["config"]
    model_cfg = cfg["model"]
    model = build_model(
        obs_dim=3,
        action_dim=2,
        use_recurrence=bool(model_cfg["use_recurrence"]),
        embed_dim=int(model_cfg["embed_dim"]),
        gru_hidden=int(model_cfg["gru_hidden"]),
        head_hidden=int(model_cfg["head_hidden"]),
        action_log_std_init=float(model_cfg["action_log_std_init"]),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, cfg


def run_ablation(
    model: torch.nn.Module,
    config: dict,
    mode: Literal["normal", "zero_every_tick", "random_every_tick"],
    num_episodes: int,
    seed: int,
) -> AblationResult:
    env_cfg = config.get("env", {})
    rewards = []
    for ep in range(num_episodes):
        env = MemoryToyEnv(
            episode_length=int(env_cfg.get("episode_length", 64)),
            cue_visible_ticks=int(env_cfg.get("cue_visible_ticks", 4)),
        )
        obs, _ = env.reset(seed=seed + ep)
        h = model.init_hidden(batch_size=1)
        done = False
        ep_reward = 0.0
        while not done:
            if mode == "zero_every_tick":
                h.zero_()
            elif mode == "random_every_tick":
                h = torch.randn_like(h)

            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, h = model.sample_action(obs_t, h)
            obs, r, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
            ep_reward += float(r)
            done = bool(term or trunc)
        rewards.append(ep_reward)
        env.close()

    arr = np.asarray(rewards, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(max(1, arr.size))
    return AblationResult(mode=mode, mean=mean, ci95=float(ci95), n=int(arr.size))


def ablation_modes_differ(num_episodes: int = 20, seed: int = 0) -> bool:
    model = build_model(
        obs_dim=3,
        action_dim=2,
        use_recurrence=True,
        embed_dim=16,
        gru_hidden=16,
        head_hidden=16,
        action_log_std_init=-1.0,
    )
    cfg = {"env": {"episode_length": 32, "cue_visible_ticks": 4}}
    n = run_ablation(model, cfg, "normal", num_episodes, seed).mean
    z = run_ablation(model, cfg, "zero_every_tick", num_episodes, seed).mean
    r = run_ablation(model, cfg, "random_every_tick", num_episodes, seed).mean
    return len({round(n, 6), round(z, 6), round(r, 6)}) == 3


def _check_gate(normal: AblationResult, zero: AblationResult, random_: AblationResult) -> tuple[bool, list[str]]:
    failures = []
    if not (normal.mean > -0.15):
        failures.append(f"normal_mean={normal.mean:.3f} is not > -0.15")
    if not (-1.2 <= zero.mean <= -0.8):
        failures.append(f"zero_every_tick_mean={zero.mean:.3f} outside [-1.2, -0.8]")
    if not (-1.5 <= random_.mean <= -0.8):
        failures.append(f"random_every_tick_mean={random_.mean:.3f} outside [-1.5, -0.8]")
    gap = normal.mean - zero.mean
    if not (gap > 0.5):
        failures.append(f"gap normal-zero = {gap:.3f} is not > 0.5")
    return len(failures) == 0, failures


def _print_table(normal: AblationResult, zero: AblationResult, random_: AblationResult) -> None:
    print("mode                  mean     ci95      n")
    print("--------------------  -------  --------  ---")
    for res in (normal, zero, random_):
        print(f"{res.mode:<20}  {res.mean:>7.3f}  ±{res.ci95:<7.3f}  {res.n}")
    print(f"\ngap (normal - zero):  {normal.mean - zero.mean:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-2 memory-toy ablation gate")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, cfg = load_checkpoint(args.checkpoint)
    normal = run_ablation(model, cfg, "normal", args.episodes, args.seed)
    zero = run_ablation(model, cfg, "zero_every_tick", args.episodes, args.seed)
    random_ = run_ablation(model, cfg, "random_every_tick", args.episodes, args.seed)

    _print_table(normal, zero, random_)
    ok, failures = _check_gate(normal, zero, random_)
    if ok:
        print("PHASE 2 GATE: PASS")
        return 0

    print("PHASE 2 GATE: FAIL")
    for msg in failures:
        print(f" - {msg}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
