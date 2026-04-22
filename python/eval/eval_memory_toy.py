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


@dataclass(frozen=True)
class EvalConfig:
    episode_length: int
    cue_visible_ticks: int


def _extract_state_dict(ckpt: dict) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict", "model"):
        state = ckpt.get(key)
        if isinstance(state, dict):
            return state
    raise KeyError(
        "checkpoint missing model state dict; expected one of "
        "{'model_state_dict', 'state_dict', 'model'}"
    )


def _extract_config(ckpt: dict) -> dict:
    for key in ("config", "cfg", "train_config"):
        cfg = ckpt.get(key)
        if isinstance(cfg, dict):
            return cfg
    return {}


def _load_state_dict_with_prefix_fallback(
    model: ActorCritic,
    state_dict: dict[str, torch.Tensor],
) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass

    if all(k.startswith("model.") for k in state_dict):
        stripped = {k[len("model.") :]: v for k, v in state_dict.items()}
        model.load_state_dict(stripped, strict=True)
        return

    raise RuntimeError("could not load checkpoint state dict into ActorCritic")


def load_checkpoint_for_eval(checkpoint: str | Path) -> tuple[ActorCritic, EvalConfig]:
    ckpt = torch.load(Path(checkpoint), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError("checkpoint must deserialize to a dict")

    state_dict = _extract_state_dict(ckpt)
    cfg = _extract_config(ckpt)
    model_cfg = cfg.get("model", cfg)
    env_cfg = cfg.get("env", {})

    obs_dim = int(model_cfg.get("obs_dim", 3))
    action_dim = int(model_cfg.get("action_dim", 2))
    use_recurrence = bool(model_cfg.get("use_recurrence", True))
    embed_dim = int(model_cfg.get("embed_dim", 64))
    gru_hidden = int(model_cfg.get("gru_hidden", 64))
    head_hidden = int(model_cfg.get("head_hidden", 64))
    action_log_std_init = float(model_cfg.get("action_log_std_init", -1.0))

    model = build_model(
        obs_dim=obs_dim,
        action_dim=action_dim,
        use_recurrence=use_recurrence,
        embed_dim=embed_dim,
        gru_hidden=gru_hidden,
        head_hidden=head_hidden,
        action_log_std_init=action_log_std_init,
    )
    _load_state_dict_with_prefix_fallback(model, state_dict)
    model.eval()

    eval_cfg = EvalConfig(
        episode_length=int(env_cfg.get("episode_length", 64)),
        cue_visible_ticks=int(env_cfg.get("cue_visible_ticks", 4)),
    )
    return model, eval_cfg


def _apply_hidden_mutation(
    h: torch.Tensor,
    mode: str,
    rng: torch.Generator,
) -> torch.Tensor:
    if mode == "normal":
        return h
    if mode == "zero_every_tick":
        return torch.zeros_like(h)
    if mode == "random_every_tick":
        return torch.randn(h.shape, dtype=h.dtype, device=h.device, generator=rng)
    raise ValueError(f"unsupported ablation mode: {mode}")


def _ci95(samples: np.ndarray) -> float:
    if samples.size <= 1:
        return 0.0
    return 1.96 * float(samples.std(ddof=1)) / float(np.sqrt(samples.size))


def run_ablation(
    model: ActorCritic,
    eval_cfg: EvalConfig,
    episodes: int,
    seed: int,
    mode: str,
) -> AblationResult:
    terminal_rewards: list[float] = []
    mode_rng = torch.Generator(device="cpu")
    mode_rng.manual_seed(int(seed) + 0xABCDEF)

    for ep_idx in range(episodes):
        env = MemoryToyEnv(
            episode_length=eval_cfg.episode_length,
            cue_visible_ticks=eval_cfg.cue_visible_ticks,
        )
        obs, _ = env.reset(seed=int(seed) + ep_idx)
        h = model.init_hidden(batch_size=1)

        terminated = False
        truncated = False
        while not (terminated or truncated):
            h_in = _apply_hidden_mutation(h, mode=mode, rng=mode_rng)
            obs_t = torch.as_tensor(obs, dtype=torch.float32).view(1, -1)
            with torch.no_grad():
                action_mean, _log_std, _value, h_next = model.forward(obs_t, h_in)
            action = torch.tanh(action_mean).squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _info = env.step(action)
            h = h_next

        terminal_rewards.append(float(reward))
        env.close()

    samples = np.asarray(terminal_rewards, dtype=np.float64)
    return AblationResult(mean=float(samples.mean()), ci95=_ci95(samples), n=samples.size)


def gate_ablation_results(results: dict[str, AblationResult]) -> tuple[bool, list[str]]:
    failures: list[str] = []

    normal = results["normal"].mean
    zero = results["zero_every_tick"].mean
    random = results["random_every_tick"].mean

    if not (normal > -0.15):
        failures.append(f"normal mean must be > -0.15, got {normal:.4f}")
    if not (-1.2 <= zero <= -0.8):
        failures.append(f"zero_every_tick mean must be in [-1.2, -0.8], got {zero:.4f}")
    if not (-1.5 <= random <= -0.8):
        failures.append(f"random_every_tick mean must be in [-1.5, -0.8], got {random:.4f}")
    if not (normal - zero > 0.5):
        failures.append(
            "normal - zero_every_tick mean must be > 0.5, "
            f"got {(normal - zero):.4f}"
        )

    return len(failures) == 0, failures


def _format_results_table(results: dict[str, AblationResult]) -> str:
    lines = [
        "mode              mean      ci95      n",
        "----------------  --------  --------  ---",
    ]
    for mode in ("normal", "zero_every_tick", "random_every_tick"):
        r = results[mode]
        lines.append(f"{mode:16}  {r.mean:8.4f}  {r.ci95:8.4f}  {r.n:3d}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="MemoryToy ablation evaluator")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0x4D454D54)
    args = parser.parse_args()

    model, eval_cfg = load_checkpoint_for_eval(args.checkpoint)

    results = {
        mode: run_ablation(
            model=model,
            eval_cfg=eval_cfg,
            episodes=args.episodes,
            seed=args.seed,
            mode=mode,
        )
        for mode in ("normal", "zero_every_tick", "random_every_tick")
    }

    print(_format_results_table(results))
    ok, failures = gate_ablation_results(results)

    if ok:
        print("\nPASS: all ablation gates satisfied")
        return 0

    print("\nFAIL: ablation gates not satisfied")
    for failure in failures:
        print(f" - {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
