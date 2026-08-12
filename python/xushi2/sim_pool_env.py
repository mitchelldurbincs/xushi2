"""SimPoolVectorEnv — batched vector env over the native C++ SimPool.

Drop-in replacement for ``XushiVectorEnv`` wrapping N
``Phase4MultiEnemyMappoEnv`` instances (the live Phase-4 training shape:
3 learner agents vs 3 scripted-bot opponents, native entity-grid actor
observations). One GIL-released FFI call advances all envs per vector step;
entity obs, critic obs, and the reward feature block come back in
preallocated buffers.

What stays in Python, by design:
- Reward math: one ``RewardCalculator`` per env consuming a
  ``RewardFeatureView`` — line-for-line the legacy reward path, fed from
  the feature block instead of ~30 FFI reads (see reward_features.py).
- Episode control: terminal envs are reset from Python (map/seed
  bookkeeping is per-episode, not per-step), matching XushiVectorEnv's
  auto-reset seed schedule exactly.
- Infos carry only what training consumes: reward metrics, the
  training-read objective metrics, loss-mask defaults, and final_* /
  reset bookkeeping. The legacy per-step diagnostics (combat_metrics,
  opponent_actions, hex state hashes) are deliberately absent — use the
  per-env backends for eval/replay tooling.

Parity with the legacy stack is pinned by
python/tests/test_sim_pool_env_parity.py.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from . import xushi2_cpp as _cpp
from .entity_obs_native import phase4_multi_enemy_obs_config
from .multi_enemy_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM
from .obs_manifest import CRITIC_DIM, REWARD_FEATURE_DIM, reward_feature_slice
from .reward import RewardCalculator
from .reward_features import RewardFeatureView
from .runner import _build_config

__all__ = ["SimPoolVectorEnv"]

_AGENTS = int(_cpp.AGENTS_PER_MATCH)
_ACTION_DIM = int(_cpp.POOL_ACTION_DIM)
_TICK_HZ = float(_cpp.TICK_HZ)

_TICK = reward_feature_slice("tick").start
_A_SCORE = reward_feature_slice("team_a_score_ticks").start
_B_SCORE = reward_feature_slice("team_b_score_ticks").start
_CAP_TICKS = reward_feature_slice("cap_progress_ticks").start
_ALIVE_SLOT = reward_feature_slice("alive_by_slot")
_ON_POINT_SLOT = reward_feature_slice("on_point_by_slot")

_SUPPORTED_SETTERS = frozenset(
    {
        "set_team_spirit",
        "set_majority_on_point_alpha",
        "set_uncontested_on_point_alpha",
        "set_objective_timing_seconds",
        "set_respawn_ticks",
    }
)


class SimPoolVectorEnv:
    """Batched Phase-4 multi-enemy vector env backed by the native SimPool."""

    def __init__(
        self,
        *,
        num_envs: int,
        sim_cfg: dict[str, Any],
        opponent_bot: str,
        learner_team: str = "A",
        reward_cfg: dict[str, Any] | None = None,
        critic_obs_dim: int = CRITIC_DIM,
        seed_base: int = 0,
        auto_reset: bool = True,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be > 0")
        if learner_team not in ("A", "B"):
            raise ValueError(f"learner_team must be 'A' or 'B', got {learner_team!r}")
        if str(opponent_bot) == "snapshot" or str(opponent_bot).startswith("snapshot:"):
            raise ValueError(
                "SimPoolVectorEnv does not drive snapshot opponents yet; use "
                "the per-env vector backends for snapshot/league runs"
            )
        if int(critic_obs_dim) != CRITIC_DIM:
            raise ValueError(
                f"SimPoolVectorEnv serves the {CRITIC_DIM}-float Phase-4 "
                f"critic; got critic_obs_dim={critic_obs_dim}"
            )
        if "team_size" in sim_cfg:
            raise ValueError("sim_cfg must not carry 'team_size'; the env owns this knob")

        self.num_envs = int(num_envs)
        self.critic_obs_dim = CRITIC_DIM
        self.seed_base = int(seed_base)
        self.auto_reset = bool(auto_reset)
        self._sim_cfg = dict(sim_cfg)
        self._learner_team = learner_team
        self._learner_slots = (0, 1, 2) if learner_team == "A" else (3, 4, 5)
        self._opponent_slots = (3, 4, 5) if learner_team == "A" else (0, 1, 2)
        self._critic_row = 0 if learner_team == "A" else 1

        base_cfg = _build_config(self._sim_cfg)
        base_cfg.team_size = 3
        self._pool = _cpp.SimPool(self.num_envs, base_cfg, phase4_multi_enemy_obs_config())
        self._applied_unlock_ticks = int(base_cfg.objective_unlock_ticks)
        self._applied_capture_ticks = int(base_cfg.objective_capture_ticks)
        self._applied_respawn_ticks = int(base_cfg.mechanics.respawn_ticks)

        self._opponent_bots = [str(opponent_bot)] * self.num_envs
        self._pending_opponent_bots: list[str | None] = [None] * self.num_envs
        for i in range(self.num_envs):
            for slot in self._opponent_slots:
                self._pool.set_slot_scripted(i, slot, self._opponent_bots[i])
                self._pool.set_obs_slot(i, slot, False)

        reward_kwargs = dict(reward_cfg or {})
        reward_kwargs.pop("per_agent_rewards", None)
        self._reward_calcs = [
            RewardCalculator(per_agent_rewards=True, **reward_kwargs)
            for _ in range(self.num_envs)
        ]

        # Preallocated step buffers — reused every vector step.
        self._entity = np.zeros(
            (self.num_envs, _AGENTS, MULTI_ENEMY_ENTITY_GRID_OBS_DIM), dtype=np.float32
        )
        self._critic = np.zeros((self.num_envs, 2, CRITIC_DIM), dtype=np.float32)
        self._features = np.zeros((self.num_envs, REWARD_FEATURE_DIM), dtype=np.float32)
        self._prev_features = np.zeros_like(self._features)
        self._terminated = np.zeros(self.num_envs, dtype=np.uint8)
        self._truncated = np.zeros(self.num_envs, dtype=np.uint8)
        self._actions6 = np.zeros((self.num_envs, _AGENTS, _ACTION_DIM), dtype=np.float32)
        self._views = [RewardFeatureView(self._features[i]) for i in range(self.num_envs)]
        self._episode_counts = np.zeros(self.num_envs, dtype=np.int64)

        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM),
            dtype=np.float32,
        )
        low = np.tile(np.array([-1, -1, -1, 0, 0, 0], dtype=np.float32), (3, 1))
        high = np.ones((3, 6), dtype=np.float32)
        self.single_action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_envs, 3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.broadcast_to(low, (self.num_envs, 3, 6)),
            high=np.broadcast_to(high, (self.num_envs, 3, 6)),
            dtype=np.float32,
        )

    # --- episode control ----------------------------------------------------

    def _reset_env_i(self, i: int, seed: int) -> None:
        pending = self._pending_opponent_bots[i]
        if pending is not None:
            self._pending_opponent_bots[i] = None
            if pending.startswith("snapshot:") or pending == "snapshot":
                raise ValueError(
                    "SimPoolVectorEnv does not drive snapshot opponents yet"
                )
            self._opponent_bots[i] = pending
            for slot in self._opponent_slots:
                self._pool.set_slot_scripted(i, slot, pending)
        self._pool.reset_env(i, int(seed))
        self._pool.env_outputs(
            i,
            self._entity[i].reshape(-1),
            self._critic[i].reshape(-1),
            self._features[i],
        )
        self._reward_calcs[i].reset(self._views[i])

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        base_seed = self.seed_base if seed is None else int(seed)
        self._episode_counts[:] = 0
        for i in range(self.num_envs):
            self._reset_env_i(i, base_seed + i)
        infos = [self._make_info(i) for i in range(self.num_envs)]
        return self._learner_obs(), self.critic_obs(), infos

    # --- observation views ---------------------------------------------------

    def _learner_obs(self) -> np.ndarray:
        lo = self._learner_slots[0]
        return self._entity[:, lo : lo + 3].copy()

    def critic_obs(self) -> np.ndarray:
        return self._critic[:, self._critic_row].copy()

    # --- the hot path ---------------------------------------------------------

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self.num_envs, 3, _ACTION_DIM)
        if actions.shape != expected:
            raise ValueError(f"actions shape must be {expected}, got {actions.shape}")

        lo = self._learner_slots[0]
        self._actions6[:, lo : lo + 3] = actions
        self._prev_features[:] = self._features
        self._pool.step(
            self._actions6.reshape(-1),
            self._entity.reshape(-1),
            self._critic.reshape(-1),
            self._features.reshape(-1),
            self._terminated,
            self._truncated,
        )

        terminated = self._terminated.astype(bool)
        truncated = self._truncated.astype(bool)
        rewards = np.zeros((self.num_envs, 3), dtype=np.float32)
        obs_out = self._learner_obs()
        infos: list[dict[str, Any]] = []

        for i in range(self.num_envs):
            calc = self._reward_calcs[i]
            view = self._views[i]
            r_a, r_b = calc.step(view)
            own = r_a if self._learner_team == "A" else r_b
            term = bool(terminated[i])
            trunc = bool(truncated[i])
            if term or trunc:
                ta, tb = calc.add_terminal(view)
                own = own + (ta if self._learner_team == "A" else tb)
            rewards[i] = own

            info = self._make_info(i)
            reward_metrics = calc.majority_on_point_metrics()
            reward_metrics.update(calc.uncontested_on_point_metrics())
            reward_metrics.update(calc.objective_conversion_metrics())
            info["reward_team_a"] = float(np.asarray(r_a).sum())
            info["reward_team_b"] = float(np.asarray(r_b).sum())
            info.update({k: float(v) for k, v in reward_metrics.items()})
            info["reward_metrics"] = reward_metrics
            info["objective_metrics"] = self._objective_metrics(i)

            if (term or trunc) and self.auto_reset:
                info["final_info"] = dict(info)
                info["final_observation"] = obs_out[i].copy()
                info["final_critic_observation"] = self._critic[i, self._critic_row].copy()
                self._episode_counts[i] += 1
                reset_seed = self.seed_base + 10_000 * int(self._episode_counts[i]) + i
                self._reset_env_i(i, reset_seed)
                obs_out[i] = self._entity[i, lo : lo + 3]
                info["reset_info"] = self._make_info(i)
            infos.append(info)

        return (
            obs_out,
            rewards,
            terminated,
            truncated,
            self.critic_obs(),
            infos,
        )

    # --- info assembly ---------------------------------------------------------

    def _make_info(self, i: int) -> dict[str, Any]:
        return {
            "learner_team": self._learner_team,
            "objective_unlock_seconds": self._applied_unlock_ticks / _TICK_HZ,
            "objective_capture_seconds": self._applied_capture_ticks / _TICK_HZ,
            "respawn_ticks": self._applied_respawn_ticks,
            "opponent_bot": self._opponent_bots[i],
        }

    def _objective_metrics(self, i: int) -> dict[str, float]:
        """The training-consumed subset of the legacy objective metrics.

        Same formulas as Phase4MappoEnv._objective_metrics_after_step,
        computed from consecutive reward feature blocks. The eval-only
        engagement metrics (nearest-enemy distances, LoS counts) are
        deliberately not produced on this backend.
        """
        now = self._features[i]
        before = self._prev_features[i]
        seconds = max(0.0, float(now[_TICK]) - float(before[_TICK])) / _TICK_HZ
        score_a_delta = float(now[_A_SCORE]) - float(before[_A_SCORE])
        score_b_delta = float(now[_B_SCORE]) - float(before[_B_SCORE])
        cap_delta = float(now[_CAP_TICKS]) - float(before[_CAP_TICKS])
        alive_a = float(now[_ALIVE_SLOT][:3].sum())
        alive_b = float(now[_ALIVE_SLOT][3:].sum())
        on_a = float(now[_ON_POINT_SLOT][:3].sum())
        on_b = float(now[_ON_POINT_SLOT][3:].sum())
        return {
            "uncontested_on_point_seconds_a": seconds if on_a > 0 and on_b == 0 else 0.0,
            "uncontested_on_point_seconds_b": seconds if on_b > 0 and on_a == 0 else 0.0,
            "majority_on_point_seconds_a": seconds if on_a > on_b else 0.0,
            "majority_on_point_seconds_b": seconds if on_b > on_a else 0.0,
            "alive_edge_no_score_seconds_a": (
                seconds if alive_a > alive_b and score_a_delta <= 0 else 0.0
            ),
            "alive_edge_no_score_seconds_b": (
                seconds if alive_b > alive_a and score_b_delta <= 0 else 0.0
            ),
            "cap_progress_gain_ticks": max(0.0, cap_delta),
            "cap_progress_loss_ticks": max(0.0, -cap_delta),
            "team_a_score_delta_ticks": max(0.0, score_a_delta),
            "team_b_score_delta_ticks": max(0.0, score_b_delta),
            "alive_on_point_a": on_a,
            "alive_on_point_b": on_b,
            "contested_majority_flag_a": 1.0 if on_a > on_b and on_b > 0 else 0.0,
            "contested_majority_flag_b": 1.0 if on_b > on_a and on_a > 0 else 0.0,
        }

    # --- curriculum -----------------------------------------------------------

    def supported_curriculum_setters(self) -> frozenset[str]:
        return _SUPPORTED_SETTERS

    def set_team_spirit(self, value: float) -> None:
        for calc in self._reward_calcs:
            calc.set_team_spirit(float(value))

    def set_majority_on_point_alpha(self, value: float) -> None:
        for calc in self._reward_calcs:
            calc.set_majority_on_point_alpha(float(value))

    def set_uncontested_on_point_alpha(self, value: float) -> None:
        for calc in self._reward_calcs:
            calc.set_uncontested_on_point_alpha(float(value))

    def set_objective_timing_seconds(
        self, unlock_seconds: float, capture_seconds: float
    ) -> None:
        unlock = round(float(unlock_seconds) * _TICK_HZ)
        capture = round(float(capture_seconds) * _TICK_HZ)
        if unlock <= 0 or capture <= 0:
            raise ValueError(
                f"objective timing ticks must be >0, got unlock={unlock} capture={capture}"
            )
        self._pool.set_objective_timing_ticks(int(unlock), int(capture))
        self._applied_unlock_ticks = int(unlock)
        self._applied_capture_ticks = int(capture)

    def set_respawn_ticks(self, respawn_ticks: int) -> None:
        ticks = int(respawn_ticks)
        if ticks <= 0:
            raise ValueError(f"respawn ticks must be >0, got {ticks}")
        self._pool.set_respawn_ticks(ticks)
        self._applied_respawn_ticks = ticks

    def set_opponent_bots(self, opponent_bots: list[str]) -> None:
        bots = [str(bot) for bot in opponent_bots]
        if len(bots) != self.num_envs:
            raise ValueError(
                f"opponent_bots must have one entry per env; got {len(bots)} "
                f"for {self.num_envs} envs"
            )
        # Applied at each env's next reset, like the legacy pending setter.
        for i, bot in enumerate(bots):
            self._pending_opponent_bots[i] = bot

    def set_opponent_handicap(
        self, bot: str, aim_noise_radians: float, fire_cadence_ticks: int
    ) -> None:
        self._pool.set_opponent_handicap(
            str(bot), float(aim_noise_radians), int(fire_cadence_ticks)
        )

    def close(self) -> None:
        pass
