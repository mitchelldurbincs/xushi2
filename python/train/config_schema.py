"""Config key validation.

Training configs are plain dicts read with ``.get(key, default)`` all the way
down. Nothing rejected a key nobody reads, so a typo silently fell back to a
default and the run completed looking entirely normal. ``entropy_coeff`` for
``entropy_coef``, ``gae_lamda`` for ``gae_lambda``, ``fog_moad`` for
``fog_mode`` are all indistinguishable from "that feature is off", and the
result gets recorded in the journal as evidence about a setting that was never
applied.

Two committed keys proved the cost before this module existed:

- ``model.use_recurrence`` appeared in every Phase-4 config and in eight test
  files, and was read by nothing.
- ``ppo.minibatch_size`` was a *required* key stored on MappoConfig and read by
  nothing; the update path does one full-batch step per epoch.

Scope, deliberately: this validates key *names*, not value types or ranges.
Value validation already exists where it matters and is closer to the code that
uses it -- ``_validate_mappo_hyperparameters`` for the PPO block,
``RewardCalculator.__init__`` for rewards, ``runner._build_config`` for the sim
block, which already rejects unknown keys. Duplicating those here would create
a second source of truth. What was missing is the check that a key is read at
all, and that is what this provides.

Adding a config key means adding it here. That is the point: the registry is
the list of keys the code actually consumes, and keeping it current is the
mechanism by which a typo stays an error.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ConfigKeyError", "validate_config_keys"]


class ConfigKeyError(ValueError):
    """A config contains a key nothing reads."""


# Top-level sections. `sim` and `env.sim` are validated by
# xushi2.runner._build_config, which already rejects unknown keys.
_TOP_LEVEL = frozenset(
    {
        "phase",
        "name",
        "experiment",
        "metadata",
        "learner",
        "env",
        "sim",
        "run",
        "ppo",
        "model",
        "wandb",
        "phase_gate",
        "phase_gate_defaults",
    }
)

_ENV_KEYS = frozenset(
    {
        "kind",
        "sim",
        "reward",
        "seed_base",
        "opponent",
        "opponent_bot",
        "learner_team",
        "actor_obs",
        "critic_obs",
        "features",
        "fog_mode",
        "visible_radius",
        "map_randomization",
        # Native entity-obs pipeline flag (ObservationEngine), read by
        # runtime_factory; required by the sim_pool vector backend.
        "native_entity_obs",
        "mini_game",
        "mini_game_config",
        "objective_timing_curriculum",
        "respawn_curriculum",
        "self_play",
        "self_play_schedule",
        "snapshot_league",
        "team_size",
        "n_agents",
        "match_type",
        "current_selfplay",
        # Ladder/self-play campaign keys (2026-07-29 onward), read by
        # mappo_training_hooks / opponent_mix / runtime_factory.
        "opponent_bot_mix",
        "opponent_handicap_curriculum",
        "opponent_recent_self",
        "opponent_snapshot_stochastic",
        "snapshot_paths",
    }
)

_PPO_KEYS = frozenset(
    {
        "num_envs",
        "rollout_len",
        "num_epochs",
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_ratio",
        "value_clip_ratio",
        "value_coef",
        "value_normalization",
        "value_per_agent",
        "entropy_coef",
        "entropy_coef_move",
        "entropy_coef_aim",
        "entropy_coef_binary",
        "max_grad_norm",
        "lr_schedule",
        "lr_final_ratio",
        "warmup_updates",
        "vector_env",
        "torch_num_threads",
        "device",
        "agent_loss_mask",
        "mask_fire_when_no_visible_enemy",
        "team_spirit_initial",
        "team_spirit_final",
        "team_spirit_ramp_fraction",
        "aim_aux_coef",
        "mode_gated_combat",
        "mode_aux_coef",
        "critic_warmup_updates",
        "anchor_kl_coef",
        "anchor_kl_anneal_updates",
        "target_selection_dim",
        "target_selection_label",
        "target_conditioned_combat",
        "target_selection_aux_coef",
        "target_selection_aux_mode",
        "target_selection_objective_proximity_coef",
        # Sharpening anneals (2026-07-29 onward), read by mappo_model /
        # mappo_rollout_trainer via mappo_training_hooks.
        "entropy_anneal_updates",
        "entropy_final_scale",
        "log_std_anneal_updates",
        "log_std_final_offset",
    }
)

_MODEL_KEYS = frozenset(
    {
        "embed_dim",
        "gru_hidden",
        "head_hidden",
        "action_log_std_init",
        "obs_encoder",
        "entity_token_count",
        "entity_token_dim",
        "entity_num_heads",
        "grid_channels",
        "grid_size",
    }
)

_RUN_KEYS = frozenset(
    {
        "total_updates",
        "eval_every",
        "eval_episodes",
        "checkpoint_every",
        "log_every",
        "output_dir",
        "device",
        "episodes",
        "team_a_bot",
        "team_b_bot",
        "assert_determinism",
        "init_from_checkpoint",
        "resume_from",
        "warm_start_migration",
        "eval_gate",
        "eval_opponent",
        "matrix_eval",
        "direct_diagnostic",
        "full_env_rehearsal",
        "multi_enemy_supervised_bridge",
        "cap_duel_distill",
        "cap_duel_distill_early_stop",
        "snapshot_retention",
        "bc_pretrain_steps",
        "bc_pretrain_variant",
        "bc_batch_size",
        "bc_learning_rate",
        "bc_combat_gate",
        "bc_freeze_actor_aim",
        "bc_aim_target_checkpoint",
        "bc_aim_rehearsal_batch_size",
        "composition_pretrain",
        "composition_pretrain_steps",
        "composition_gate",
        "composition_eval_episodes",
        "composition_objective_env",
        "composition_combat_env",
        "composition_objective_batch_size",
        "composition_combat_batch_size",
        "composition_objective_teacher_checkpoint",
        "composition_combat_teacher_checkpoint",
    }
)

_WANDB_KEYS = frozenset(
    {"enabled", "required", "project", "entity", "group", "name", "tags", "init_timeout_seconds"}
)

# Keys nothing reads, kept out of the allow-lists on purpose so an existing
# config carrying one gets a pointer rather than a bare "unknown key".
_REMOVED_KEYS: dict[tuple[str, str], str] = {
    ("model", "use_recurrence"): (
        "the actor is always recurrent; this key was never read"
    ),
    ("ppo", "minibatch_size"): (
        "the update path does one full-batch step per epoch; there is no "
        "minibatching to configure"
    ),
    ("env", "target_slot"): "use ppo.target_selection_dim",
}

_SECTIONS: dict[str, frozenset[str]] = {
    "env": _ENV_KEYS,
    "ppo": _PPO_KEYS,
    "model": _MODEL_KEYS,
    "run": _RUN_KEYS,
    "wandb": _WANDB_KEYS,
}

# `metadata` is free-form provenance for humans and is never read by the
# training code, so it is not key-checked. `phase_gate` / `phase_gate_defaults`
# have their own pydantic models in train.phase_gate.
_UNCHECKED_SECTIONS = frozenset({"metadata", "phase_gate", "phase_gate_defaults", "experiment"})


def _suggest(unknown: str, known: frozenset[str]) -> str:
    """Cheap did-you-mean, since most of these are typos."""
    import difflib

    matches = difflib.get_close_matches(unknown, sorted(known), n=1, cutoff=0.7)
    return f"; did you mean {matches[0]!r}?" if matches else ""


def validate_config_keys(config: dict[str, Any]) -> None:
    """Raise ConfigKeyError if the config carries a key nothing reads."""
    if not isinstance(config, dict):
        raise ConfigKeyError(f"config must be a mapping, got {type(config).__name__}")

    problems: list[str] = []

    unknown_top = sorted(set(config) - _TOP_LEVEL)
    for key in unknown_top:
        problems.append(f"unknown top-level key {key!r}{_suggest(key, _TOP_LEVEL)}")

    for section, allowed in _SECTIONS.items():
        block = config.get(section)
        if not isinstance(block, dict):
            continue
        for key in sorted(set(block) - allowed):
            removed = _REMOVED_KEYS.get((section, key))
            if removed is not None:
                problems.append(f"{section}.{key} has been removed: {removed}")
            else:
                problems.append(f"unknown key {section}.{key}{_suggest(key, allowed)}")

    if problems:
        raise ConfigKeyError(
            "config contains keys that nothing reads, which would be silently "
            "ignored and produce a run that does not match its config:\n  - "
            + "\n  - ".join(problems)
            + f"\n(unchecked sections: {sorted(_UNCHECKED_SECTIONS)}; "
            "env.sim is validated separately by xushi2.runner)"
        )
