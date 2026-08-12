"""Minimal six-agent current-vs-current MAPPO env for Phase 4.

This keeps Phase 4's flat Ranger observations and fixed-map assumptions, but
lets the shared current policy drive both teams in a 3v3 match.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.phase4_mappo import Phase4MappoEnv
from xushi2 import xushi2_cpp as _cpp
from xushi2.multi_enemy_obs import map_bounds_from_sim_cfg
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM
from xushi2.reward import RewardCalculator
from xushi2.runner import _build_config
from xushi2.self_play_schedule import SelfPlayMatch, SelfPlaySchedule
from xushi2.snapshot_policy import SnapshotPolicy

__all__ = ["Phase4CurrentSelfplayMappoEnv"]

_AGENTS_PER_MATCH = _cpp.AGENTS_PER_MATCH


class Phase4CurrentSelfplayMappoEnv(Phase4MappoEnv):
    """Phase 4 flat-observation 3v3 current self-play env.

    All six slots are controlled by the same policy. The trainer uses
    ``value_per_agent=True`` so each slot receives the centralized critic view
    for its own team.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    n_agents: int = _AGENTS_PER_MATCH
    actor_obs_dim: int = ACTOR_PHASE1_DIM
    critic_obs_dim: int = CRITIC_DIM
    action_dim: int = 6

    # See xushi2.env_capabilities. This env inherits Phase4MappoEnv's
    # set_team_spirit, but builds its RewardCalculator without
    # per_agent_rewards, and team_spirit only shapes the per-agent step path.
    # The inherited setter would therefore accept a ramp and change nothing, so
    # declare it unsupported and let a configured ramp fail at startup instead.
    UNSUPPORTED_CURRICULUM_SETTERS: ClassVar[dict[str, str]] = {
        "set_team_spirit": (
            "reward is computed on the scalar path (per_agent_rewards=False), "
            "which team_spirit does not shape"
        ),
    }

    def __init__(
        self,
        sim_cfg: dict,
        *,
        reward_cfg: dict[str, Any] | None = None,
        self_play_schedule: dict[str, Any] | None = None,
        snapshot_league: dict[str, Any] | None = None,
    ) -> None:
        gym.Env.__init__(self)
        self._sim_cfg = dict(sim_cfg)
        self._opponent_bot = "self"
        self._learner_team_str = "both"
        self._sim: _cpp.Sim | None = None
        self._opponent_policy: SnapshotPolicy | None = None
        self._reward_cfg = dict(reward_cfg or {})
        self._reward_cfg.pop("per_agent_rewards", None)
        self._reward_cfg.pop("majority_on_point_anneal_updates", None)
        self._reward_cfg.pop("uncontested_on_point_anneal_updates", None)
        self._reward_calc = RewardCalculator(**self._reward_cfg)
        self._schedule = (
            SelfPlaySchedule(weights={"current": 1.0})
            if self_play_schedule is None
            else SelfPlaySchedule.from_config(self_play_schedule, snapshot_league)
        )
        self._last_match = SelfPlayMatch(match_type="current", group="current")
        self._last_opponent_actions = np.zeros((3, 6), dtype=np.float32)

        self._actor_obs_buf = np.zeros(
            (_AGENTS_PER_MATCH, ACTOR_PHASE1_DIM), dtype=np.float32
        )
        self._objective_actor_obs_buf = np.zeros(
            (_AGENTS_PER_MATCH, ACTOR_PHASE1_DIM), dtype=np.float32
        )
        self._objective_critic_obs_buf = np.zeros(CRITIC_DIM, dtype=np.float32)
        self._first_team_a_alive_edge_tick: int | None = None
        self._first_team_a_score_after_edge_tick: int | None = None
        self._loss_mask = np.ones(_AGENTS_PER_MATCH, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(_AGENTS_PER_MATCH, ACTOR_PHASE1_DIM),
            dtype=np.float32,
        )
        low = np.tile(
            np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (_AGENTS_PER_MATCH, 1),
        )
        high = np.tile(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            (_AGENTS_PER_MATCH, 1),
        )
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(_AGENTS_PER_MATCH, 6),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)

        if "team_size" in self._sim_cfg:
            raise ValueError("sim_cfg must not carry 'team_size'; the env owns this knob")

        cfg = _build_config(self._sim_cfg, seed_override=seed)
        cfg.team_size = 3
        self._sim = _cpp.Sim(cfg)
        seed_int = 0 if seed is None else int(seed)
        self._last_match = self._schedule.sample(seed_int)
        self._loss_mask = (
            np.ones(_AGENTS_PER_MATCH, dtype=np.float32)
            if self._last_match.match_type == "current"
            else np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
        )
        self._last_opponent_actions = np.zeros((3, 6), dtype=np.float32)
        self._opponent_policy = (
            SnapshotPolicy(self._last_match.snapshot_path)
            if self._last_match.snapshot_path is not None
            and self._last_match.match_type != "current"
            else None
        )
        if self._opponent_policy is not None:
            self._opponent_policy.reset(batch_size=3)
        self._reward_calc.reset(self._sim)
        self._first_team_a_alive_edge_tick = None
        self._first_team_a_score_after_edge_tick = None
        self._build_actor_obs_all()
        return self._actor_obs_buf.copy(), self._make_info()

    def step(self, action: np.ndarray):
        if self._sim is None:
            raise RuntimeError("reset() must be called before step()")
        action = np.asarray(action, dtype=np.float32)
        if (
            action.ndim != 2
            or action.shape[0] != _AGENTS_PER_MATCH
            or action.shape[1] < 6
        ):
            raise ValueError(
                f"action shape must be ({_AGENTS_PER_MATCH}, >=6), got {action.shape}"
            )

        actions = [
            self._action_to_cpp_for_slot(action[slot], slot)
            for slot in range(_AGENTS_PER_MATCH)
        ]
        self._last_opponent_actions = np.zeros((3, 6), dtype=np.float32)
        if self._last_match.match_type != "current":
            if self._opponent_policy is not None:
                opponent = np.asarray(
                    self._opponent_policy.act(self._sim, (3, 4, 5)),
                    dtype=np.float32,
                )
                if opponent.shape[0] != 3 or opponent.shape[1] < 6:
                    raise ValueError(
                        "Phase-4 snapshot opponent must emit at least six "
                        f"controls per Team-B agent, got {opponent.shape}"
                    )
                opponent = opponent[:, :6]
                for idx, slot in enumerate((3, 4, 5)):
                    actions[slot] = self._action_to_cpp_for_slot(opponent[idx], slot)
                self._last_opponent_actions[:] = opponent
            else:
                bot = self._last_match.anchor_bot or "noop"
                for idx, slot in enumerate((3, 4, 5)):
                    scripted = _cpp.scripted_bot_action(self._sim, slot, bot)
                    actions[slot] = scripted
                    self._last_opponent_actions[idx] = np.array(
                        [
                            scripted.move_x,
                            scripted.move_y,
                            scripted.aim_delta / (np.pi / 4.0),
                            float(scripted.primary_fire),
                            float(scripted.ability_1),
                            float(scripted.ability_2),
                        ],
                        dtype=np.float32,
                    )
        previous_damage = np.asarray(self._sim.damage_dealt_by_slot, dtype=np.int64)
        objective_before = self._objective_snapshot()
        # Same ordering contract as Phase4MappoEnv.step: derive the contest
        # state once, pre-step, and hand it to both metric functions.
        contested_majority_team = self._contested_majority_team(objective_before)
        combat_metrics = self._combat_metrics_before_step(actions, contested_majority_team)

        self._sim.step_decision(actions)
        damage_delta = np.asarray(self._sim.damage_dealt_by_slot, dtype=np.int64) - previous_damage
        self._attach_damage_metrics(combat_metrics, damage_delta, contested_majority_team)
        objective_metrics = self._objective_metrics_after_step(objective_before)

        r_a, r_b = self._reward_calc.step(self._sim)
        reward_metrics = self._reward_calc.majority_on_point_metrics()
        reward_metrics.update(self._reward_calc.uncontested_on_point_metrics())
        reward = np.asarray([r_a, r_a, r_a, r_b, r_b, r_b], dtype=np.float32)

        # terminated == the MDP genuinely ended (a team reached the score
        # threshold); truncated == the round timer cut it off. Deriving
        # these from `winner` labelled a timeout-with-a-winner as
        # terminated and a draw as truncated, which inverts the common
        # case: reaching the score threshold is rare, timing out is not.
        terminated = bool(self._sim.score_threshold_reached)
        truncated = bool(self._sim.episode_over) and not terminated
        if terminated or truncated:
            ta, tb = self._reward_calc.add_terminal(self._sim)
            reward += np.asarray([ta, ta, ta, tb, tb, tb], dtype=np.float32)

        self._build_actor_obs_all()
        info = self._make_info()
        info["reward_team_a"] = float(r_a)
        info["reward_team_b"] = float(r_b)
        info.update({k: float(v) for k, v in reward_metrics.items()})
        info["reward_metrics"] = reward_metrics
        info["objective_metrics"] = objective_metrics
        info["combat_metrics"] = combat_metrics
        info["opponent_actions"] = self._last_opponent_actions.copy()
        return self._actor_obs_buf.copy(), reward, terminated, truncated, info

    def build_critic_obs(self, out: np.ndarray) -> None:
        if self._sim is None:
            raise RuntimeError("reset() must be called before build_critic_obs()")
        if not isinstance(out, np.ndarray):
            raise ValueError("out must be an np.ndarray")
        expected = (_AGENTS_PER_MATCH * CRITIC_DIM,)
        if out.shape != expected or out.dtype != np.float32:
            raise ValueError(
                f"out must be float32 ndarray of shape {expected}, got {out.shape} {out.dtype}"
            )
        views = out.reshape(_AGENTS_PER_MATCH, CRITIC_DIM)
        _cpp.build_critic_obs(self._sim, _cpp.Team.A, views[0])
        views[1:3] = views[0]
        _cpp.build_critic_obs(self._sim, _cpp.Team.B, views[3])
        views[4:6] = views[3]

    def _build_actor_obs_all(self) -> None:
        for slot in range(_AGENTS_PER_MATCH):
            _cpp.build_actor_obs(self._sim, slot, self._actor_obs_buf[slot])

    def _make_info(self) -> dict[str, Any]:
        s = self._sim
        assert s is not None
        winner = s.winner
        if winner == _cpp.Team.A:
            winner_str = "A"
        elif winner == _cpp.Team.B:
            winner_str = "B"
        else:
            winner_str = "Neutral"
        return {
            "tick": int(s.tick),
            "state_hash": f"0x{int(s.state_hash):016x}",
            "team_a_score": float(s.team_a_score),
            "team_b_score": float(s.team_b_score),
            "team_a_kills": int(s.team_a_kills),
            "team_b_kills": int(s.team_b_kills),
            "winner": winner_str,
            "learner_team": "both"
            if self._last_match.match_type == "current"
            else "A",
            "match_type": self._last_match.match_type,
            "schedule": self._schedule.summary,
            "loss_mask": self._loss_mask.copy(),
            "opponent_actions": self._last_opponent_actions.copy(),
            "snapshot_path": self._last_match.snapshot_path,
            "snapshot_group": self._last_match.group,
            "anchor_bot": self._last_match.anchor_bot,
            "objective_unlock_ticks": int(s.objective_unlock_ticks),
            "objective_capture_ticks": int(s.objective_capture_ticks),
            "objective_unlock_seconds": float(s.objective_unlock_ticks)
            / float(_cpp.TICK_HZ),
            "objective_capture_seconds": float(s.objective_capture_ticks)
            / float(_cpp.TICK_HZ),
        }
