"""Phase-3 checkpoint evaluation against the scripted Ranger env."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train.models import build_model
from train.ppo_recurrent.evaluate import evaluate_policy
from train.ppo_recurrent.orchestration import make_env_fn


def load_checkpoint(path: str | Path):
    ckpt = torch.load(Path(path), map_location="cpu")
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})
    model = build_model(
        obs_dim=int(model_cfg["obs_dim"]),
        action_dim=int(model_cfg["action_dim"]),
        use_recurrence=bool(model_cfg.get("use_recurrence", True)),
        embed_dim=int(model_cfg["embed_dim"]),
        gru_hidden=int(model_cfg["gru_hidden"]),
        head_hidden=int(model_cfg["head_hidden"]),
        action_log_std_init=float(model_cfg["action_log_std_init"]),
        continuous_action_dim=int(model_cfg.get("continuous_action_dim", model_cfg["action_dim"])),
        binary_action_dim=int(model_cfg.get("binary_action_dim", 0)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Phase-3 recurrent checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0xD1CEDA7A)
    args = parser.parse_args()

    model, ckpt_config = load_checkpoint(args.checkpoint)
    train_config = {
        "phase": int(ckpt_config.get("phase", 3)),
        "env": ckpt_config["env"],
    }
    env_fn, _env_meta, _seed_base = make_env_fn(train_config)
    mean_reward = evaluate_policy(
        model,
        env_fn,
        num_episodes=args.episodes,
        seed=args.seed,
    )
    print(f"[phase3-eval] mean_reward={mean_reward:+.3f} episodes={args.episodes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
