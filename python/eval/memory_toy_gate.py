from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from envs.memory_toy import MemoryToyEnv
from train.models import ActorCritic, build_model

MODES: tuple[str, str, str] = ("normal", "zero_every_tick", "random_every_tick")


@dataclass(frozen=True)
class MemoryToyGateThresholds:
    normal_mean_min: float
    zero_mean_range: tuple[float, float]
    random_mean_range: tuple[float, float]
    normal_zero_gap_min: float
    version: str
    source_path: str


@dataclass(frozen=True)
class AblationResult:
    mean: float
    ci95: float
    n: int
    mode: str = ""


@dataclass(frozen=True)
class GateAggregateResult:
    per_mode: dict[str, AblationResult]
    gap_normal_minus_zero: float
    passed: bool
    failure_reasons: list[str]
    thresholds: MemoryToyGateThresholds


def load_checkpoint(path: str | Path) -> tuple[ActorCritic, dict]:
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
        continuous_action_dim=int(
            model_cfg.get("continuous_action_dim", model_cfg.get("action_dim", 2))
        ),
        binary_action_dim=int(model_cfg.get("binary_action_dim", 0)),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def _parse_range(name: str, obj: object) -> tuple[float, float]:
    if not isinstance(obj, (list, tuple)) or len(obj) != 2:
        raise ValueError(f"{name} must be a 2-item list/tuple")
    lo, hi = float(obj[0]), float(obj[1])
    if lo > hi:
        raise ValueError(f"{name}[0] must be <= {name}[1]")
    return lo, hi


def load_memory_toy_gate_thresholds(path: str | Path) -> MemoryToyGateThresholds:
    threshold_path = Path(path)
    payload = yaml.safe_load(threshold_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("threshold config must be a mapping")

    required = {
        "version",
        "normal_mean_min",
        "zero_mean_range",
        "random_mean_range",
        "normal_zero_gap_min",
    }
    extras = set(payload.keys()) - required
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(f"threshold config missing required keys: {sorted(missing)}")
    if extras:
        raise ValueError(f"threshold config has unexpected keys: {sorted(extras)}")

    return MemoryToyGateThresholds(
        normal_mean_min=float(payload["normal_mean_min"]),
        zero_mean_range=_parse_range("zero_mean_range", payload["zero_mean_range"]),
        random_mean_range=_parse_range("random_mean_range", payload["random_mean_range"]),
        normal_zero_gap_min=float(payload["normal_zero_gap_min"]),
        version=str(payload["version"]),
        source_path=str(threshold_path),
    )


def _apply_hidden_mutation(h: torch.Tensor, mode: str, rng: torch.Generator) -> torch.Tensor:
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


def evaluate_gate_thresholds(
    results: dict[str, AblationResult], thresholds: MemoryToyGateThresholds
) -> tuple[bool, list[str]]:
    normal = results["normal"]
    zero = results["zero_every_tick"]
    random_ = results["random_every_tick"]

    failures: list[str] = []
    if not (normal.mean > thresholds.normal_mean_min):
        failures.append(f"normal_mean={normal.mean:.3f} is not > {thresholds.normal_mean_min}")
    if not (thresholds.zero_mean_range[0] <= zero.mean <= thresholds.zero_mean_range[1]):
        failures.append(
            f"zero_every_tick_mean={zero.mean:.3f} outside [{thresholds.zero_mean_range[0]}, {thresholds.zero_mean_range[1]}]"
        )
    if not (thresholds.random_mean_range[0] <= random_.mean <= thresholds.random_mean_range[1]):
        failures.append(
            f"random_every_tick_mean={random_.mean:.3f} outside [{thresholds.random_mean_range[0]}, {thresholds.random_mean_range[1]}]"
        )

    gap = normal.mean - zero.mean
    if not (gap > thresholds.normal_zero_gap_min):
        failures.append(f"gap normal-zero = {gap:.3f} is not > {thresholds.normal_zero_gap_min}")

    return len(failures) == 0, failures


def evaluate_memory_toy_gate(
    model: ActorCritic,
    config: dict,
    num_episodes: int,
    seed: int,
    thresholds: MemoryToyGateThresholds,
) -> GateAggregateResult:
    per_mode = {
        mode: run_ablation(
            model=model,
            config=config,
            num_episodes=num_episodes,
            seed=seed,
            mode=mode,
        )
        for mode in MODES
    }
    passed, failure_reasons = evaluate_gate_thresholds(per_mode, thresholds=thresholds)
    gap = per_mode["normal"].mean - per_mode["zero_every_tick"].mean
    return GateAggregateResult(
        per_mode=per_mode,
        gap_normal_minus_zero=gap,
        passed=passed,
        failure_reasons=failure_reasons,
        thresholds=thresholds,
    )


def format_result_table(results: dict[str, AblationResult]) -> str:
    lines = [
        "mode                  mean      ci95      n",
        "--------------------  --------  --------  ----",
    ]
    for mode in MODES:
        r = results[mode]
        lines.append(f"{mode:20}  {r.mean:+8.4f}  {r.ci95:8.4f}  {r.n:4d}")
    return "\n".join(lines)
