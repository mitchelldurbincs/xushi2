"""3v3 MAPPO-shaped Gymnasium env (Phase 4).

Per-agent (3, 31) observations and (3, 6) actions, with a separate
post-step ``build_critic_obs(out)`` hook that writes 135 floats into a
caller-provided buffer. Drives the C++ sim with team_size=3.

See docs/plans/2026-05-07-phase4-mappo-env-design.md for layout
rationale.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from xushi2 import xushi2_cpp as _cpp
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM
from xushi2.reward import RewardCalculator
from xushi2.runner import _build_config

__all__ = ["Phase4MappoEnv", "VALID_OPPONENT_BOTS"]

VALID_OPPONENT_BOTS: frozenset[str] = frozenset({
    "walk_to_objective", "hold_and_shoot", "basic", "noop",
})

_AGENTS_PER_MATCH = _cpp.AGENTS_PER_MATCH

_AIM_DELTA_LIMIT = float(np.pi / 4.0)


class Phase4MappoEnv(gym.Env):
    """3v3 MAPPO env: per-agent obs/action, team-broadcast reward."""

    metadata = {"render_modes": []}

    n_agents: int = 3
    actor_obs_dim: int = ACTOR_PHASE1_DIM
    critic_obs_dim: int = CRITIC_DIM
    action_dim: int = 6

    def __init__(
        self,
        sim_cfg: dict,
        *,
        opponent_bot: str,
        learner_team: str = "A",
        reward_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        if opponent_bot not in VALID_OPPONENT_BOTS:
            raise ValueError(
                f"unknown opponent_bot {opponent_bot!r}; "
                f"valid: {sorted(VALID_OPPONENT_BOTS)}"
            )
        if learner_team not in ("A", "B"):
            raise ValueError(
                f"learner_team must be 'A' or 'B', got {learner_team!r}"
            )

        self._sim_cfg = dict(sim_cfg)
        self._opponent_bot = opponent_bot
        self._learner_team_str = learner_team
        self._learner_team = (
            _cpp.Team.A if learner_team == "A" else _cpp.Team.B
        )
        self._own_slots: tuple[int, int, int] = (
            (0, 1, 2) if learner_team == "A" else (3, 4, 5)
        )
        self._enemy_slots: tuple[int, int, int] = (
            (3, 4, 5) if learner_team == "A" else (0, 1, 2)
        )

        self._sim: _cpp.Sim | None = None
        self._reward_cfg = dict(reward_cfg or {})
        self._reward_calc = RewardCalculator(**self._reward_cfg)

        self._actor_obs_buf = np.zeros(
            (3, ACTOR_PHASE1_DIM), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(3, ACTOR_PHASE1_DIM),
            dtype=np.float32,
        )
        low = np.tile(
            np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (3, 1),
        )
        high = np.tile(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            (3, 1),
        )
        self.action_space = spaces.Box(
            low=low, high=high, shape=(3, 6), dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if "team_size" in self._sim_cfg:
            raise ValueError(
                "sim_cfg must not carry 'team_size'; the env owns this knob"
            )

        cfg = _build_config(self._sim_cfg, seed_override=seed)
        cfg.team_size = 3
        self._sim = _cpp.Sim(cfg)
        self._reward_calc.reset(self._sim)
        self._build_actor_obs_all()
        return self._actor_obs_buf.copy(), self._make_info()

    def step(self, action: np.ndarray):
        if self._sim is None:
            raise RuntimeError("reset() must be called before step()")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3, 6):
            raise ValueError(
                f"action shape must be (3, 6), got {action.shape}"
            )

        actions = [_cpp.Action() for _ in range(_AGENTS_PER_MATCH)]
        for slot, a in zip(self._own_slots, action):
            actions[slot] = self._action_to_cpp(a)
        for enemy_slot in self._enemy_slots:
            actions[enemy_slot] = _cpp.scripted_bot_action(
                self._sim, enemy_slot, self._opponent_bot
            )

        self._sim.step_decision(actions)

        r_a, r_b = self._reward_calc.step(self._sim)
        team_reward = r_a if self._learner_team_str == "A" else r_b

        terminated = bool(self._sim.episode_over) and (
            self._sim.winner != _cpp.Team.Neutral
        )
        truncated = bool(self._sim.episode_over) and (
            self._sim.winner == _cpp.Team.Neutral
        )
        if terminated or truncated:
            ta, tb = self._reward_calc.add_terminal(self._sim)
            team_reward += ta if self._learner_team_str == "A" else tb

        reward = np.full(3, team_reward, dtype=np.float32)
        self._build_actor_obs_all()
        info = self._make_info()
        info["reward_team_a"] = float(r_a)
        info["reward_team_b"] = float(r_b)
        return self._actor_obs_buf.copy(), reward, terminated, truncated, info

    @staticmethod
    def _action_to_cpp(a: np.ndarray) -> "_cpp.Action":
        a = np.asarray(a, dtype=np.float32).reshape(6)
        a[:3] = np.clip(a[:3], -1.0, 1.0)
        a[3:] = np.clip(a[3:], 0.0, 1.0)
        act = _cpp.Action()
        act.move_x = float(a[0])
        act.move_y = float(a[1])
        act.aim_delta = float(a[2] * _AIM_DELTA_LIMIT)
        act.primary_fire = bool(a[3] >= 0.5)
        act.ability_1 = bool(a[4] >= 0.5)
        act.ability_2 = bool(a[5] >= 0.5)
        return act

    def build_critic_obs(self, out: np.ndarray) -> None:
        if self._sim is None:
            raise RuntimeError(
                "reset() must be called before build_critic_obs()"
            )
        if not isinstance(out, np.ndarray):
            raise ValueError("out must be an np.ndarray")
        if out.shape != (CRITIC_DIM,) or out.dtype != np.float32:
            raise ValueError(
                f"out must be float32 ndarray of shape ({CRITIC_DIM},), "
                f"got {out.shape} {out.dtype}"
            )
        _cpp.build_critic_obs(self._sim, self._learner_team, out)

    def close(self) -> None:
        self._sim = None

    def _build_actor_obs_all(self) -> None:
        for i, slot in enumerate(self._own_slots):
            _cpp.build_actor_obs(self._sim, slot, self._actor_obs_buf[i])

    def _make_info(self) -> dict[str, Any]:
        s = self._sim
        winner = s.winner
        if winner == _cpp.Team.A:
            winner_str = "A"
        elif winner == _cpp.Team.B:
            winner_str = "B"
        else:
            winner_str = "Neutral"
        return {
            "tick": int(s.tick),
            # Gymnasium vector info collation stores Python ints in a C-long
            # array; uint64 hashes can overflow there, so expose hex text.
            "state_hash": f"0x{int(s.state_hash):016x}",
            "team_a_score": float(s.team_a_score),
            "team_b_score": float(s.team_b_score),
            "team_a_kills": int(s.team_a_kills),
            "team_b_kills": int(s.team_b_kills),
            "winner": winner_str,
            "learner_team": self._learner_team_str,
        }
