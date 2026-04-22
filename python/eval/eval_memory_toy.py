"""Phase-2 memory-toy ablation gate.

Exit-code-gated harness that decides whether Phase 2 passes. Given a
recurrent-policy checkpoint, runs three ablations:

* ``normal``            — hidden state carried normally
* ``zero_every_tick``   — ``h`` clobbered to zero before every forward
* ``random_every_tick`` — ``h`` clobbered to ``N(0, 1)`` before every forward

and asserts the four gate conditions from ``docs/memory_toy.md``:

1. ``normal_mean > -0.15``
2. ``-1.2 <= zero_mean <= -0.8``
3. ``-1.5 <= random_mean <= -0.8``
4. ``normal_mean - zero_mean > 0.5``  (memory effect dominates noise)

A passing gate proves the recurrent trainer is actually using the hidden
state; a failing gate indicates the silent recurrent-PPO failures
enumerated in ``rl_design.md`` §10.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from envs.memory_toy import MemoryToyEnv
from train.models import ActorCritic, build_model


@dataclass(frozen=True)
class AblationResult:
    mean: float
    ci95: float
    n: int
    mode: str = ""


_MODES = ("normal", "zero_every_tick", "random_every_tick")


def load_checkpoint(path: str | Path) -> tuple[ActorCritic, dict]:
    """Load a checkpoint saved by ``train.ppo_recurrent._save_checkpoint``.

    Returns ``(model, config)`` where ``config`` has ``env`` + ``model``
    sub-dicts.
    """
    ckpt = torch.load(Path(path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint at {path} must be a dict, got {type(ckpt)!r}")
    state_dict = ckpt["model_state_dict"]
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})

    model = build_model(
        obs_dim=int(model_cfg.get("obs_dim", 3)),
        action_dim=int(model_cfg.get("action_dim", 2)),
        use_recurrence=bool(model_cfg.get("use_recurrence", True)),
        embed_dim=int(model_cfg.get("embed_dim", 64)),
        gru_hidden=int(model_cfg.get("gru_hidden", 64)),
        head_hidden=int(model_cfg.get("head_hidden", 64)),
        action_log_std_init=float(model_cfg.get("action_log_std_init", -1.0)),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def _apply_hidden_mutation(
    h: torch.Tensor, mode: str, rng: torch.Generator
) -> torch.Tensor:
    if mode == "normal":
        return h
    if mode == "zero_every_tick":
        return torch.zeros_like(h)
    if mode == "random_every_tick":
        return torch.randn(h.shape, dtype=h.dtype, generator=rng)
    raise ValueError(f"unsupported ablation mode: {mode!r}")


def _ci95(samples: np.ndarray) -> float:
    if samples.size <= 1:
        return 0.0
    return 1.96 * float(samples.std(ddof=1)) / float(np.sqrt(samples.size))


def run_ablation(
    model: ActorCritic,
    config: dict,
    num_episodes: int,
    seed: int,
    mode: str,
) -> AblationResult:
    """Run ``num_episodes`` greedy rollouts under an ablation mode.

    Returns the mean terminal reward plus 95% CI. The per-tick hidden
    state is mutated by ``_apply_hidden_mutation`` *before* every forward
    pass (including the first one of the episode, which has no effect in
    ``normal`` or ``zero_every_tick`` modes since ``h`` is already zero).
    """
    env_cfg = config.get("env", {}) if isinstance(config, dict) else {}
    episode_length = int(env_cfg.get("episode_length", 64))
    cue_visible_ticks = int(env_cfg.get("cue_visible_ticks", 4))

    mode_rng = torch.Generator()
    mode_rng.manual_seed(int(seed) + 0xABCDEF)

    terminal_rewards: list[float] = []
    for ep_idx in range(int(num_episodes)):
        env = MemoryToyEnv(
            episode_length=episode_length,
            cue_visible_ticks=cue_visible_ticks,
        )
        obs, _ = env.reset(seed=int(seed) + ep_idx)
        h = model.init_hidden(batch_size=1)

        last_reward = 0.0
        done = False
        while not done:
            h_in = _apply_hidden_mutation(h, mode=mode, rng=mode_rng)
            obs_t = torch.as_tensor(obs, dtype=torch.float32).view(1, -1)
            with torch.no_grad():
                mean, _log_std, _value, h_next = model.forward(obs_t, h_in)
            action = torch.tanh(mean).squeeze(0).cpu().numpy()
            obs, r, term, trunc, _info = env.step(action)
            last_reward = float(r)
            h = h_next
            done = bool(term or trunc)

        terminal_rewards.append(last_reward)
        env.close()

    samples = np.asarray(terminal_rewards, dtype=np.float64)
    return AblationResult(
        mean=float(samples.mean()),
        ci95=_ci95(samples),
        n=int(samples.size),
        mode=mode,
    )


def ablation_modes_differ(
    model: ActorCritic, config: dict, num_episodes: int, seed: int
) -> bool:
    """Sanity check: the three ablation modes must route through distinct
    code paths. Returns True iff all three mean rewards are pairwise distinct.
    """
    results = {
        mode: run_ablation(
            model=model,
            config=config,
            num_episodes=num_episodes,
            seed=seed,
            mode=mode,
        )
        for mode in _MODES
    }
    means = [results[m].mean for m in _MODES]
    return len(set(means)) == len(means)


def _check_gate(
    normal: AblationResult,
    zero: AblationResult,
    random_: AblationResult,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not (normal.mean > -0.15):
        failures.append(f"normal_mean={normal.mean:.3f} is not > -0.15")
    if not (-1.2 <= zero.mean <= -0.8):
        failures.append(
            f"zero_every_tick_mean={zero.mean:.3f} outside [-1.2, -0.8]"
        )
    if not (-1.5 <= random_.mean <= -0.8):
        failures.append(
            f"random_every_tick_mean={random_.mean:.3f} outside [-1.5, -0.8]"
        )
    if not (normal.mean - zero.mean > 0.5):
        failures.append(
            f"gap normal-zero = {normal.mean - zero.mean:.3f} is not > 0.5"
        )
    return len(failures) == 0, failures


def _format_table(results: dict[str, AblationResult]) -> str:
    lines = [
        "mode                  mean      ci95      n",
        "--------------------  --------  --------  ----",
    ]
    for mode in _MODES:
        r = results[mode]
        lines.append(f"{mode:20}  {r.mean:+8.4f}  {r.ci95:8.4f}  {r.n:4d}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="MemoryToy ablation gate")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument(
        "--seed", type=lambda s: int(s, 0), default=0x4D454D54
    )
    args = parser.parse_args()

    model, config = load_checkpoint(args.checkpoint)

    results = {
        mode: run_ablation(
            model=model,
            config=config,
            num_episodes=args.episodes,
            seed=args.seed,
            mode=mode,
        )
        for mode in _MODES
    }

    print(_format_table(results))
    gap = results["normal"].mean - results["zero_every_tick"].mean
    print(f"\ngap (normal - zero): {gap:+.3f}")

    ok, failures = _check_gate(
        results["normal"],
        results["zero_every_tick"],
        results["random_every_tick"],
    )
    if ok:
        print("PHASE 2 GATE: PASS")
        return 0

    print("PHASE 2 GATE: FAIL")
    for msg in failures:
        print(f" - {msg}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
