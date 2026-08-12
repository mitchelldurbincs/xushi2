"""SimPoolVectorEnv vs the legacy per-env XushiVectorEnv stack.

Same seeds, same action stream, full episodes across auto-resets: entity and
critic observations must be byte-identical (both sides build them with the
native ObservationEngine), terminated/truncated must match exactly, rewards
to 1e-6 (identical RewardCalculator code — the pool feeds it through the
feature block, where the only wiggle room is hypot rounding), and the
training-consumed info metrics must agree.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from envs.runtime_factory import make_mappo_match_env
from xushi2.vector_env import XushiVectorEnv, make_xushi_vector_env

_SIM_CFG = {
    "seed": 20260812,
    "round_length_seconds": 10,
    "fog_of_war_enabled": False,
    "randomize_map": False,
    "action_repeat": 3,
    "mechanics": {
        "revolver_damage_centi_hp": 7500,
        "revolver_fire_cooldown_ticks": 15,
        "revolver_hitbox_radius": 0.75,
        "respawn_ticks": 240,
    },
}

# Probe-v5-shaped reward config: exercises kills/deaths/damage, distance and
# on-point shaping, uncontested-on-point, and the conversion potential — i.e.
# every fake-sim hook the feature view feeds.
_REWARD_CFG = {
    "shaping_clip": 30.0,
    "score_per_second": 1.0,
    "time_penalty_per_second": 0.05,
    "death_penalty": 0.5,
    "kill_bonus": 0.1,
    "damage_dealt_coef": 0.005,
    "distance_shaping_coef": 0.005,
    "on_point_shaping_coef": 0.02,
    "uncontested_on_point_coef": 0.15,
    "cap_progress_potential_coef": 1.0,
    "capture_completed_bonus": 2.0,
}

_TRAINING_OBJECTIVE_KEYS = (
    "uncontested_on_point_seconds_a",
    "uncontested_on_point_seconds_b",
    "majority_on_point_seconds_a",
    "majority_on_point_seconds_b",
    "alive_edge_no_score_seconds_a",
    "alive_edge_no_score_seconds_b",
    "cap_progress_gain_ticks",
    "cap_progress_loss_ticks",
    "team_a_score_delta_ticks",
    "team_b_score_delta_ticks",
)


def _env_fns(num_envs: int):
    fn = functools.partial(
        make_mappo_match_env,
        sim_cfg=dict(_SIM_CFG),
        opponent_bot="weak_basic_v2",
        learner_team="A",
        reward_cfg=dict(_REWARD_CFG),
        actor_obs="multi_enemy_entity_grid",
        native_entity_obs=True,
    )
    return [fn for _ in range(num_envs)]


def _action_stream(rng: np.ndarray, step: int, num_envs: int) -> np.ndarray:
    act = np.zeros((num_envs, 3, 6), dtype=np.float32)
    act[:, :, 1] = 1.0  # advance toward the objective
    act[:, :, 0] = np.sin(0.13 * step + rng[:, None])
    act[:, :, 2] = 0.1
    act[:, :, 3] = 1.0  # fire
    return act


def test_sim_pool_env_matches_legacy_vector_env() -> None:
    num_envs = 4
    seed_base = 991
    fns = _env_fns(num_envs)
    legacy = XushiVectorEnv(fns, critic_obs_dim=135, seed_base=seed_base)
    pool = make_xushi_vector_env(
        fns, critic_obs_dim=135, seed_base=seed_base, backend="sim_pool"
    )
    phase = np.linspace(0.0, 2.0, num_envs)
    saw_done = False
    try:
        obs_l, critic_l, _ = legacy.reset(seed=seed_base)
        obs_p, critic_p, _ = pool.reset(seed=seed_base)
        np.testing.assert_array_equal(obs_p, obs_l)
        np.testing.assert_array_equal(critic_p, critic_l)

        # Curriculum pushes must land identically on both stacks.
        legacy.set_opponent_handicap("weak_basic_v2", 1.5, 60)
        pool.set_opponent_handicap("weak_basic_v2", 1.5, 60)
        legacy.set_objective_timing_seconds(15.0, 8.0)
        pool.set_objective_timing_seconds(15.0, 8.0)
        legacy.set_uncontested_on_point_alpha(0.15)
        pool.set_uncontested_on_point_alpha(0.15)

        # 10 s round = 100 decisions; 130 steps crosses the auto-reset.
        for step in range(130):
            act = _action_stream(phase, step, num_envs)
            obs_l, rew_l, term_l, trunc_l, critic_l, infos_l = legacy.step(act)
            obs_p, rew_p, term_p, trunc_p, critic_p, infos_p = pool.step(act)

            np.testing.assert_array_equal(
                term_p, term_l, err_msg=f"terminated diverged at step {step}"
            )
            np.testing.assert_array_equal(
                trunc_p, trunc_l, err_msg=f"truncated diverged at step {step}"
            )
            np.testing.assert_array_equal(
                obs_p, obs_l, err_msg=f"entity obs diverged at step {step}"
            )
            np.testing.assert_array_equal(
                critic_p, critic_l, err_msg=f"critic obs diverged at step {step}"
            )
            np.testing.assert_allclose(
                rew_p, rew_l, atol=1e-6, rtol=0,
                err_msg=f"rewards diverged at step {step}",
            )
            for i in range(num_envs):
                om_l = infos_l[i].get("objective_metrics", {})
                om_p = infos_p[i].get("objective_metrics", {})
                for key in _TRAINING_OBJECTIVE_KEYS:
                    assert om_p.get(key) == pytest.approx(
                        om_l.get(key), abs=1e-9
                    ), f"objective metric {key} diverged (env {i}, step {step})"
                for key in ("reward_team_a", "reward_team_b",
                            "uncontested_on_point_reward_a",
                            "uncontested_on_point_reward_b",
                            "capture_completed_reward_a"):
                    assert infos_p[i].get(key) == pytest.approx(
                        infos_l[i].get(key), abs=1e-6
                    ), f"info key {key} diverged (env {i}, step {step})"
                if bool(term_l[i]) or bool(trunc_l[i]):
                    np.testing.assert_allclose(
                        infos_p[i]["final_critic_observation"],
                        infos_l[i]["final_critic_observation"],
                        atol=0, rtol=0,
                    )
            saw_done |= bool(term_l.any() or trunc_l.any())
    finally:
        legacy.close()
        pool.close()
    assert saw_done, "scenario never crossed an episode boundary — auto-reset untested"


def test_sim_pool_backend_rejects_unsupported_configs() -> None:
    fn = functools.partial(
        make_mappo_match_env,
        sim_cfg=dict(_SIM_CFG),
        opponent_bot="weak_basic_v2",
        actor_obs="multi_enemy_entity_grid",
        native_entity_obs=False,
    )
    with pytest.raises(ValueError, match="legacy Python obs path"):
        make_xushi_vector_env([fn], critic_obs_dim=135, backend="sim_pool")

    fn_flat = functools.partial(
        make_mappo_match_env,
        sim_cfg=dict(_SIM_CFG),
        opponent_bot="weak_basic_v2",
        actor_obs="flat",
    )
    with pytest.raises(ValueError, match="actor_obs"):
        make_xushi_vector_env([fn_flat], critic_obs_dim=135, backend="sim_pool")
