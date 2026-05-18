"""3v3 MAPPO-shaped Gymnasium env (Phase 4).

Per-agent (3, 31) observations and (3, 6) actions, with a separate
post-step ``build_critic_obs(out)`` hook that writes 135 floats into a
caller-provided buffer. Drives the C++ sim with team_size=3.

See docs/plans/2026-05-07-phase4-mappo-env-design.md for layout
rationale.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from xushi2 import xushi2_cpp as _cpp
from xushi2.multi_enemy_obs import map_bounds_from_sim_cfg
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM, critic_field_slice
from xushi2.reward import RewardCalculator
from xushi2.runner import _build_config

__all__ = ["VALID_OPPONENT_BOTS", "Phase4MappoEnv"]

VALID_OPPONENT_BOTS: frozenset[str] = frozenset(
    {
        "walk_to_objective",
        "hold_and_shoot",
        "basic",
        "weak_basic",
        "weak_basic_v2",
        "noop",
    }
)

_AGENTS_PER_MATCH = _cpp.AGENTS_PER_MATCH

_AIM_DELTA_LIMIT = float(np.pi / 4.0)


class Phase4MappoEnv(gym.Env):
    """3v3 MAPPO env: per-agent obs/action, team-broadcast reward."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

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
        opponent_policy: Any | None = None,
    ) -> None:
        super().__init__()

        if opponent_bot not in VALID_OPPONENT_BOTS and opponent_policy is None:
            raise ValueError(
                f"unknown opponent_bot {opponent_bot!r}; valid: {sorted(VALID_OPPONENT_BOTS)}"
            )
        if learner_team not in ("A", "B"):
            raise ValueError(f"learner_team must be 'A' or 'B', got {learner_team!r}")

        self._sim_cfg = dict(sim_cfg)
        self._opponent_bot = opponent_bot
        self._learner_team_str = learner_team
        self._learner_team = _cpp.Team.A if learner_team == "A" else _cpp.Team.B
        self._own_slots: tuple[int, int, int] = (0, 1, 2) if learner_team == "A" else (3, 4, 5)
        self._enemy_slots: tuple[int, int, int] = (3, 4, 5) if learner_team == "A" else (0, 1, 2)

        self._sim: _cpp.Sim | None = None
        self._opponent_policy = opponent_policy
        self._reward_cfg = dict(reward_cfg or {})
        # Phase 4 always emits per-agent rewards so MAPPO can use individual
        # credit assignment + the team_spirit lever.
        self._reward_cfg.pop("per_agent_rewards", None)
        self._reward_calc = RewardCalculator(per_agent_rewards=True, **self._reward_cfg)

        self._actor_obs_buf = np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
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
            low=low,
            high=high,
            shape=(3, 6),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if "team_size" in self._sim_cfg:
            raise ValueError("sim_cfg must not carry 'team_size'; the env owns this knob")

        cfg = _build_config(self._sim_cfg, seed_override=seed)
        cfg.team_size = 3
        self._sim = _cpp.Sim(cfg)
        if self._opponent_policy is not None:
            self._opponent_policy.reset()
        self._reward_calc.reset(self._sim)
        self._build_actor_obs_all()
        return self._actor_obs_buf.copy(), self._make_info()

    def step(self, action: np.ndarray):
        if self._sim is None:
            raise RuntimeError("reset() must be called before step()")
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 2 or action.shape[0] != 3 or action.shape[1] < 6:
            raise ValueError(f"action shape must be (3, >=6), got {action.shape}")

        actions = [_cpp.Action() for _ in range(_AGENTS_PER_MATCH)]
        for slot, a in zip(self._own_slots, action, strict=False):
            actions[slot] = self._action_to_cpp(a)
        opponent_actions = np.zeros((3, 6), dtype=np.float32)
        if self._opponent_policy is not None:
            enemy_actions = np.asarray(
                self._opponent_policy.act(
                    self._sim,
                    self._enemy_slots,
                    map_bounds=map_bounds_from_sim_cfg(self._sim_cfg),
                ),
                dtype=np.float32,
            )
            if enemy_actions.shape[0] != 3 or enemy_actions.shape[1] < 6:
                raise ValueError(
                    "snapshot opponent must emit at least six controls per agent, "
                    f"got {enemy_actions.shape}"
                )
            enemy_actions = enemy_actions[:, :6]
            for enemy_slot, enemy_action in zip(self._enemy_slots, enemy_actions, strict=False):
                actions[enemy_slot] = self._action_to_cpp(enemy_action)
            opponent_actions[:] = enemy_actions
        else:
            for i, enemy_slot in enumerate(self._enemy_slots):
                scripted = _cpp.scripted_bot_action(self._sim, enemy_slot, self._opponent_bot)
                actions[enemy_slot] = scripted
                opponent_actions[i] = np.array(
                    [
                        scripted.move_x,
                        scripted.move_y,
                        scripted.aim_delta / _AIM_DELTA_LIMIT,
                        float(scripted.primary_fire),
                        float(scripted.ability_1),
                        float(scripted.ability_2),
                    ],
                    dtype=np.float32,
                )

        previous_damage = np.asarray(self._sim.damage_dealt_by_slot, dtype=np.int64)
        combat_metrics = self._combat_metrics_before_step(actions)

        self._sim.step_decision(actions)
        damage_delta = np.asarray(self._sim.damage_dealt_by_slot, dtype=np.int64) - previous_damage
        self._attach_damage_metrics(combat_metrics, damage_delta)

        r_a, r_b = self._reward_calc.step(self._sim)  # shape (3,) each
        own_reward = r_a if self._learner_team_str == "A" else r_b

        terminated = bool(self._sim.episode_over) and (self._sim.winner != _cpp.Team.Neutral)
        truncated = bool(self._sim.episode_over) and (self._sim.winner == _cpp.Team.Neutral)
        if terminated or truncated:
            ta, tb = self._reward_calc.add_terminal(self._sim)  # (3,) each
            own_reward = own_reward + (ta if self._learner_team_str == "A" else tb)

        reward = np.asarray(own_reward, dtype=np.float32)
        self._build_actor_obs_all()
        info = self._make_info()
        info["reward_team_a"] = float(np.asarray(r_a).sum())
        info["reward_team_b"] = float(np.asarray(r_b).sum())
        info["opponent_actions"] = opponent_actions.copy()
        info["combat_metrics"] = combat_metrics
        return self._actor_obs_buf.copy(), reward, terminated, truncated, info

    @staticmethod
    def _action_to_cpp(a: np.ndarray) -> _cpp.Action:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.shape[0] < 6:
            raise ValueError(f"action must have at least 6 fields, got {a.shape}")
        a[:3] = np.clip(a[:3], -1.0, 1.0)
        a[3:6] = np.clip(a[3:6], 0.0, 1.0)
        act = _cpp.Action()
        act.move_x = float(a[0])
        act.move_y = float(a[1])
        act.aim_delta = float(a[2] * _AIM_DELTA_LIMIT)
        act.primary_fire = bool(a[3] >= 0.5)
        act.ability_1 = bool(a[4] >= 0.5)
        act.ability_2 = bool(a[5] >= 0.5)
        if a.shape[0] >= 7:
            act.target_slot = int(np.rint(a[6]).clip(0, 255))
        return act

    def build_critic_obs(self, out: np.ndarray) -> None:
        if self._sim is None:
            raise RuntimeError("reset() must be called before build_critic_obs()")
        if not isinstance(out, np.ndarray):
            raise ValueError("out must be an np.ndarray")
        if out.shape != (CRITIC_DIM,) or out.dtype != np.float32:
            raise ValueError(
                f"out must be float32 ndarray of shape ({CRITIC_DIM},), got {out.shape} {out.dtype}"
            )
        _cpp.build_critic_obs(self._sim, self._learner_team, out)

    def enemy_line_of_sight_mask(self, *, team_shared: bool = False) -> np.ndarray:
        if self._sim is None:
            raise RuntimeError("reset() must be called before enemy_line_of_sight_mask()")
        mask = np.zeros(3, dtype=bool)
        for i, (own_slot, enemy_slot) in enumerate(
            zip(self._own_slots, self._enemy_slots, strict=False)
        ):
            if team_shared:
                mask[i] = any(
                    bool(_cpp.observable_enemy_slots(self._sim, ally_slot)[enemy_slot])
                    for ally_slot in self._own_slots
                )
            else:
                mask[i] = bool(_cpp.observable_enemy_slots(self._sim, own_slot)[enemy_slot])
        return mask

    def set_team_spirit(self, value: float) -> None:
        """Update the team_spirit ramp value on the underlying reward calc.

        Trainer calls this once per update with the schedule output of
        ``compute_team_spirit``."""
        self._reward_calc.set_team_spirit(value)

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

    @staticmethod
    def _angle_wrap(x: float) -> float:
        return (x + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _team_for_slot(slot: int) -> str:
        return "A" if slot < 3 else "B"

    @staticmethod
    def _team_slots(team: str) -> range:
        return range(0, 3) if team == "A" else range(3, 6)

    @staticmethod
    def _enemy_slots_for(slot: int) -> range:
        return range(3, 6) if slot < 3 else range(0, 3)

    @staticmethod
    def _slot_position(critic: np.ndarray, slot: int) -> np.ndarray:
        if slot < 3:
            return critic[critic_field_slice(f"slot{slot}/own_position")]
        return critic[critic_field_slice(f"enemy{slot - 3}/world_position")]

    @staticmethod
    def _slot_aim_angle(critic: np.ndarray, slot: int) -> float:
        if slot < 3:
            unit = critic[critic_field_slice(f"slot{slot}/own_aim_unit")]
        else:
            unit = critic[critic_field_slice(f"enemy{slot - 3}/world_aim_unit")]
        return math.atan2(float(unit[0]), float(unit[1]))

    @staticmethod
    def _slot_alive(critic: np.ndarray, slot: int) -> bool:
        if slot < 3:
            return float(critic[critic_field_slice(f"slot{slot}/own_hp")][0]) > 0.0
        return bool(float(critic[critic_field_slice(f"enemy{slot - 3}/alive_flag")][0]) > 0.5)

    def _nearest_visible_target(
        self, critic: np.ndarray, slot: int
    ) -> tuple[int | None, float | None]:
        assert self._sim is not None
        if not self._slot_alive(critic, slot):
            return None, None
        try:
            visible = list(_cpp.observable_enemy_slots(self._sim, slot))
        except Exception:
            visible = [False] * _AGENTS_PER_MATCH
        own_pos = self._slot_position(critic, slot)
        aim_angle = self._slot_aim_angle(critic, slot)
        best_slot: int | None = None
        best_error: float | None = None
        for enemy in self._enemy_slots_for(slot):
            if not visible[enemy] or not self._slot_alive(critic, enemy):
                continue
            rel = self._slot_position(critic, enemy) - own_pos
            target_angle = math.atan2(float(rel[1]), float(rel[0]))
            error = abs(self._angle_wrap(aim_angle - target_angle))
            if best_error is None or error < best_error:
                best_error = error
                best_slot = enemy
        return best_slot, best_error

    @staticmethod
    def _empty_team_combat_metrics() -> dict[str, Any]:
        return {
            "fire_commands": 0,
            "visible_fire_commands": 0,
            "damage_hits": 0,
            "damage_centi_hp": 0,
            "aim_error_sum": 0.0,
            "aim_error_count": 0,
            "target_counts": {},
        }

    def _combat_metrics_before_step(self, actions: list[_cpp.Action]) -> dict[str, Any]:
        assert self._sim is not None
        critic = np.zeros(CRITIC_DIM, dtype=np.float32)
        _cpp.build_critic_obs(self._sim, _cpp.Team.A, critic)
        metrics = {
            "A": self._empty_team_combat_metrics(),
            "B": self._empty_team_combat_metrics(),
        }
        for slot, action in enumerate(actions):
            if not action.primary_fire:
                continue
            team = self._team_for_slot(slot)
            team_metrics = metrics[team]
            team_metrics["fire_commands"] += 1
            target_slot, aim_error = self._nearest_visible_target(critic, slot)
            if target_slot is not None:
                team_metrics["visible_fire_commands"] += 1
                counts = team_metrics["target_counts"]
                counts[target_slot] = int(counts.get(target_slot, 0)) + 1
            if aim_error is not None:
                team_metrics["aim_error_sum"] += float(aim_error)
                team_metrics["aim_error_count"] += 1
        return metrics

    def _attach_damage_metrics(
        self, combat_metrics: dict[str, Any], damage_delta: np.ndarray
    ) -> None:
        for team in ("A", "B"):
            team_metrics = combat_metrics[team]
            for slot in self._team_slots(team):
                delta = int(damage_delta[slot])
                if delta <= 0:
                    continue
                team_metrics["damage_hits"] += 1
                team_metrics["damage_centi_hp"] += delta
