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
from xushi2.obs_manifest import (
    ACTOR_PHASE1_DIM,
    CRITIC_DIM,
    actor_field_slice,
    critic_field_slice,
)
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
        opponent_snapshot_stochastic: bool = False,
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
        self._pending_opponent_bot: str | None = None
        self._opponent_handicap: tuple[str, float, int] | None = None
        self._opponent_snapshot_stochastic = bool(opponent_snapshot_stochastic)
        self._learner_team_str = learner_team
        self._learner_team = _cpp.Team.A if learner_team == "A" else _cpp.Team.B
        self._own_slots: tuple[int, int, int] = (0, 1, 2) if learner_team == "A" else (3, 4, 5)
        self._enemy_slots: tuple[int, int, int] = (3, 4, 5) if learner_team == "A" else (0, 1, 2)

        self._sim: _cpp.Sim | None = None
        self._applied_respawn_ticks: int | None = None
        self._opponent_policy = opponent_policy
        self._reward_cfg = dict(reward_cfg or {})
        # Phase 4 always emits per-agent rewards so MAPPO can use individual
        # credit assignment + the team_spirit lever.
        self._reward_cfg.pop("per_agent_rewards", None)
        self._reward_cfg.pop("majority_on_point_anneal_updates", None)
        self._reward_cfg.pop("uncontested_on_point_anneal_updates", None)
        self._reward_calc = RewardCalculator(per_agent_rewards=True, **self._reward_cfg)

        self._actor_obs_buf = np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32)
        self._objective_actor_obs_buf = np.zeros(
            (_AGENTS_PER_MATCH, ACTOR_PHASE1_DIM), dtype=np.float32
        )
        self._objective_critic_obs_buf = np.zeros(CRITIC_DIM, dtype=np.float32)
        self._first_team_a_alive_edge_tick: int | None = None
        self._first_team_a_score_after_edge_tick: int | None = None

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

        if self._pending_opponent_bot is not None:
            pending = self._pending_opponent_bot
            self._pending_opponent_bot = None
            if pending.startswith("snapshot:"):
                from xushi2.snapshot_policy import SnapshotPolicy

                self._opponent_policy = SnapshotPolicy(
                    pending[len("snapshot:"):],
                    stochastic=self._opponent_snapshot_stochastic,
                )
                self._opponent_bot = "snapshot"
            else:
                self._opponent_policy = None
                self._opponent_bot = pending

        cfg = _build_config(self._sim_cfg, seed_override=seed)
        cfg.team_size = 3
        self._sim = _cpp.Sim(cfg)
        self._applied_respawn_ticks = int(cfg.mechanics.respawn_ticks)
        if self._opponent_policy is not None:
            # Thread the env's reset seed into the frozen opponent so its
            # sampling stream varies with the campaign/eval seed and across
            # vector slots, matching the single-source seed flow in
            # docs/rl_design.md. Auto-resets (seed=None) keep the stream
            # advancing via the policy's own episode counter.
            self._opponent_policy.reset(seed=seed)
        self._reward_calc.reset(self._sim)
        self._first_team_a_alive_edge_tick = None
        self._first_team_a_score_after_edge_tick = None
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
            actions[slot] = self._action_to_cpp_for_slot(a, slot)
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
                actions[enemy_slot] = self._action_to_cpp_for_slot(enemy_action, enemy_slot)
            opponent_actions[:] = enemy_actions
        else:
            for i, enemy_slot in enumerate(self._enemy_slots):
                scripted = _cpp.scripted_bot_action(self._sim, enemy_slot, self._opponent_bot)
                self._apply_opponent_handicap(scripted, enemy_slot)
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
        objective_before = self._objective_snapshot()

        self._sim.step_decision(actions)
        damage_delta = np.asarray(self._sim.damage_dealt_by_slot, dtype=np.int64) - previous_damage
        self._attach_damage_metrics(combat_metrics, damage_delta)
        objective_metrics = self._objective_metrics_after_step(objective_before)

        r_a, r_b = self._reward_calc.step(self._sim)  # shape (3,) each
        reward_metrics = self._reward_calc.majority_on_point_metrics()
        reward_metrics.update(self._reward_calc.uncontested_on_point_metrics())
        reward_metrics.update(self._reward_calc.objective_conversion_metrics())
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
        info.update({k: float(v) for k, v in reward_metrics.items()})
        info["reward_metrics"] = reward_metrics
        info["objective_metrics"] = objective_metrics
        info["opponent_actions"] = opponent_actions.copy()
        info["combat_metrics"] = combat_metrics
        return self._actor_obs_buf.copy(), reward, terminated, truncated, info

    @staticmethod
    def _action_to_cpp(a: np.ndarray) -> _cpp.Action:
        return Phase4MappoEnv._action_to_cpp_for_team(a, "A")

    @staticmethod
    def _action_to_cpp_for_slot(a: np.ndarray, slot: int) -> _cpp.Action:
        team = "B" if int(slot) >= 3 else "A"
        return Phase4MappoEnv._action_to_cpp_for_team(a, team)

    @staticmethod
    def _action_to_cpp_for_team(a: np.ndarray, team: str) -> _cpp.Action:
        a = np.array(a, dtype=np.float32, copy=True).reshape(-1)
        if a.shape[0] < 6:
            raise ValueError(f"action must have at least 6 fields, got {a.shape}")
        a[:3] = np.clip(a[:3], -1.0, 1.0)
        a[3:6] = np.clip(a[3:6], 0.0, 1.0)
        act = _cpp.Action()
        # Actor observations are in a team-relative frame. Convert learned
        # movement back to the simulator's world frame for Team B.
        move_sign = -1.0 if team == "B" else 1.0
        act.move_x = float(move_sign * a[0])
        act.move_y = float(move_sign * a[1])
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

    def set_majority_on_point_alpha(self, value: float) -> None:
        self._reward_calc.set_majority_on_point_alpha(value)

    def set_uncontested_on_point_alpha(self, value: float) -> None:
        self._reward_calc.set_uncontested_on_point_alpha(value)

    def set_objective_timing_ticks(self, unlock_ticks: int, capture_ticks: int) -> None:
        unlock = int(unlock_ticks)
        capture = int(capture_ticks)
        if unlock <= 0 or capture <= 0:
            raise ValueError(
                f"objective timing ticks must be >0, got unlock={unlock} capture={capture}"
            )
        self._sim_cfg.pop("objective_unlock_seconds", None)
        self._sim_cfg.pop("objective_capture_seconds", None)
        nested = self._sim_cfg.get("objective_timing")
        if isinstance(nested, dict):
            nested.pop("unlock_seconds", None)
            nested.pop("capture_seconds", None)
        self._sim_cfg["objective_unlock_ticks"] = unlock
        self._sim_cfg["objective_capture_ticks"] = capture
        if self._sim is not None:
            self._sim.set_objective_timing_ticks(unlock, capture)

    def set_objective_timing_seconds(
        self, unlock_seconds: float, capture_seconds: float
    ) -> None:
        unlock_ticks = int(round(float(unlock_seconds) * float(_cpp.TICK_HZ)))
        capture_ticks = int(round(float(capture_seconds) * float(_cpp.TICK_HZ)))
        self.set_objective_timing_ticks(unlock_ticks, capture_ticks)

    def set_respawn_ticks(self, respawn_ticks: int) -> None:
        """Respawn-time curriculum knob. Unlike the objective-timing setter
        there is no live-sim setter (respawn_tick is stamped at death), so the
        new value applies from the next reset() onward; the in-flight episode
        keeps the respawn time it was built with."""
        ticks = int(respawn_ticks)
        if ticks <= 0:
            raise ValueError(f"respawn ticks must be >0, got {ticks}")
        mechanics = dict(self._sim_cfg.get("mechanics", {}))
        mechanics["respawn_ticks"] = ticks
        self._sim_cfg["mechanics"] = mechanics

    def set_opponent_bot(self, opponent_bot: str) -> None:
        """Opponent-mix curriculum knob. Accepts a scripted bot name or
        "snapshot:<checkpoint path>" for a frozen-policy opponent. Applies
        from the next reset() onward — the in-flight episode keeps the
        opponent it started with."""
        bot = str(opponent_bot)
        if bot.startswith("snapshot:"):
            path = bot[len("snapshot:"):]
            if not path:
                raise ValueError("snapshot opponent requires a checkpoint path")
        elif bot not in VALID_OPPONENT_BOTS:
            raise ValueError(
                f"unknown opponent_bot {bot!r}; valid: {sorted(VALID_OPPONENT_BOTS)} "
                "or snapshot:<path>"
            )
        self._pending_opponent_bot = bot

    def set_opponent_handicap(
        self,
        bot: str,
        aim_noise_radians: float,
        fire_cadence_ticks: int,
    ) -> None:
        """Opponent-handicap curriculum knob: soften one scripted bot by
        post-processing its emitted actions (extra aim noise, fire gating).

        This is an APPROXIMATE softening, not parametric interpolation
        between bot tiers: the noise is added AFTER the bot's own decide()
        (including any native noise and the pre-clamp aim path), so e.g.
        weak_basic at handicap (1.5, 60) is NOT bit-identical to
        weak_basic_v2 — it carries both noise sources and a post-clamp
        distribution (2026-08-02 review; the B2 config's original
        continuity claim was wrong). Exact tier equivalence would require
        injecting noise inside the C++ behavior primitives.

        Applies immediately per step. aim_noise 0 and cadence 1 = full
        strength. Only envs whose current opponent matches ``bot`` are
        affected. Trainer-side only — eval envs never see this setter, so
        matrix evals always measure the full-strength bot."""
        name = str(bot)
        if name not in VALID_OPPONENT_BOTS:
            raise ValueError(
                f"unknown opponent_bot {name!r}; valid: {sorted(VALID_OPPONENT_BOTS)}"
            )
        noise = float(aim_noise_radians)
        cadence = int(fire_cadence_ticks)
        if noise < 0.0:
            raise ValueError(f"aim_noise_radians must be >= 0, got {noise}")
        if cadence < 1:
            raise ValueError(f"fire_cadence_ticks must be >= 1, got {cadence}")
        self._opponent_handicap = (name, noise, cadence)

    def _apply_opponent_handicap(self, scripted, enemy_slot: int) -> None:
        handicap = self._opponent_handicap
        if handicap is None:
            return
        bot, noise_scale, cadence = handicap
        if bot != self._opponent_bot:
            return
        tick = int(self._sim.tick)
        if noise_scale > 0.0:
            # Deterministic per-(tick, slot) unit noise, mirroring the C++
            # weak_basic bots' deterministic_unit_noise: no RNG state, so
            # replays and reruns reproduce exactly.
            raw = math.sin(float(tick) * 12.9898 + float(enemy_slot) * 78.233) * 43758.5453
            unit = (raw - math.floor(raw)) * 2.0 - 1.0
            scripted.aim_delta = max(
                -_AIM_DELTA_LIMIT,
                min(_AIM_DELTA_LIMIT, scripted.aim_delta + unit * noise_scale),
            )
        if cadence > 1:
            scripted.primary_fire = bool(scripted.primary_fire) and (
                tick % cadence == 0
            )

    def close(self) -> None:
        self._sim = None

    def _build_actor_obs_all(self) -> None:
        for i, slot in enumerate(self._own_slots):
            _cpp.build_actor_obs(self._sim, slot, self._actor_obs_buf[i])

    def _objective_snapshot(self) -> dict[str, float | int]:
        assert self._sim is not None
        hp_slice = actor_field_slice("own_hp")
        on_point_slice = actor_field_slice("self_on_point")
        alive = np.zeros(_AGENTS_PER_MATCH, dtype=np.float32)
        on_point = np.zeros(_AGENTS_PER_MATCH, dtype=np.float32)
        for slot in range(_AGENTS_PER_MATCH):
            _cpp.build_actor_obs(self._sim, slot, self._objective_actor_obs_buf[slot])
            obs = self._objective_actor_obs_buf[slot]
            slot_alive = float(obs[hp_slice][0]) > 0.0
            alive[slot] = 1.0 if slot_alive else 0.0
            on_point[slot] = (
                1.0 if slot_alive and float(obs[on_point_slice][0]) > 0.5 else 0.0
            )
        _cpp.build_critic_obs(self._sim, _cpp.Team.A, self._objective_critic_obs_buf)
        cap_progress_ticks = int(
            self._objective_critic_obs_buf[critic_field_slice("cap_progress_ticks")][0]
        )
        return {
            "tick": int(self._sim.tick),
            "team_a_score_ticks": int(self._sim.team_a_score_ticks),
            "team_b_score_ticks": int(self._sim.team_b_score_ticks),
            "cap_progress_ticks": cap_progress_ticks,
            "alive_a": int(alive[0:3].sum()),
            "alive_b": int(alive[3:6].sum()),
            "alive_on_point_a": int(on_point[0:3].sum()),
            "alive_on_point_b": int(on_point[3:6].sum()),
        }

    def _objective_metrics_after_step(
        self, before: dict[str, float | int]
    ) -> dict[str, float]:
        assert self._sim is not None
        after = self._objective_snapshot()
        tick_delta = max(0, int(after["tick"]) - int(before["tick"]))
        seconds = float(tick_delta) / float(_cpp.TICK_HZ)
        score_a_delta = int(after["team_a_score_ticks"]) - int(before["team_a_score_ticks"])
        score_b_delta = int(after["team_b_score_ticks"]) - int(before["team_b_score_ticks"])
        cap_delta = int(after["cap_progress_ticks"]) - int(before["cap_progress_ticks"])

        alive_edge_a = int(after["alive_a"]) > int(after["alive_b"])
        alive_edge_b = int(after["alive_b"]) > int(after["alive_a"])
        if alive_edge_a and self._first_team_a_alive_edge_tick is None:
            self._first_team_a_alive_edge_tick = int(after["tick"])
        if (
            score_a_delta > 0
            and self._first_team_a_alive_edge_tick is not None
            and self._first_team_a_score_after_edge_tick is None
        ):
            self._first_team_a_score_after_edge_tick = int(after["tick"])

        first_a_edge_to_score_seconds = -1.0
        if (
            self._first_team_a_alive_edge_tick is not None
            and self._first_team_a_score_after_edge_tick is not None
        ):
            first_a_edge_to_score_seconds = (
                float(
                    self._first_team_a_score_after_edge_tick
                    - self._first_team_a_alive_edge_tick
                )
                / float(_cpp.TICK_HZ)
            )

        on_a = int(after["alive_on_point_a"])
        on_b = int(after["alive_on_point_b"])
        contested_majority_a = on_a > on_b and on_b > 0
        contested_majority_b = on_b > on_a and on_a > 0
        engagement = self._on_point_engagement_metrics()
        return {
            "uncontested_on_point_seconds_a": seconds if on_a > 0 and on_b == 0 else 0.0,
            "uncontested_on_point_seconds_b": seconds if on_b > 0 and on_a == 0 else 0.0,
            "majority_on_point_seconds_a": seconds if on_a > on_b else 0.0,
            "majority_on_point_seconds_b": seconds if on_b > on_a else 0.0,
            "alive_edge_no_score_seconds_a": (
                seconds if alive_edge_a and score_a_delta <= 0 else 0.0
            ),
            "alive_edge_no_score_seconds_b": (
                seconds if alive_edge_b and score_b_delta <= 0 else 0.0
            ),
            "cap_progress_gain_ticks": float(max(0, cap_delta)),
            "cap_progress_loss_ticks": float(max(0, -cap_delta)),
            "team_a_score_delta_ticks": float(max(0, score_a_delta)),
            "team_b_score_delta_ticks": float(max(0, score_b_delta)),
            "first_team_a_alive_edge_to_score_seconds": first_a_edge_to_score_seconds,
            "alive_on_point_a": float(on_a),
            "alive_on_point_b": float(on_b),
            "contested_majority_flag_a": 1.0 if contested_majority_a else 0.0,
            "contested_majority_flag_b": 1.0 if contested_majority_b else 0.0,
            "on_point_nearest_enemy_distance_sum_a": float(
                engagement["A"]["distance_sum"]
            ),
            "on_point_nearest_enemy_distance_count_a": float(
                engagement["A"]["distance_count"]
            ),
            "on_point_enemy_los_count_a": float(engagement["A"]["los_count"]),
            "on_point_total_count_a": float(engagement["A"]["on_point_count"]),
            "on_point_nearest_enemy_distance_sum_b": float(
                engagement["B"]["distance_sum"]
            ),
            "on_point_nearest_enemy_distance_count_b": float(
                engagement["B"]["distance_count"]
            ),
            "on_point_enemy_los_count_b": float(engagement["B"]["los_count"]),
            "on_point_total_count_b": float(engagement["B"]["on_point_count"]),
        }

    def _on_point_engagement_metrics(self) -> dict[str, dict[str, float]]:
        assert self._sim is not None
        critic = np.zeros(CRITIC_DIM, dtype=np.float32)
        _cpp.build_critic_obs(self._sim, _cpp.Team.A, critic)
        out = {
            "A": {"distance_sum": 0.0, "distance_count": 0.0, "los_count": 0.0, "on_point_count": 0.0},
            "B": {"distance_sum": 0.0, "distance_count": 0.0, "los_count": 0.0, "on_point_count": 0.0},
        }
        on_point_slice = actor_field_slice("self_on_point")
        hp_slice = actor_field_slice("own_hp")
        for slot in range(_AGENTS_PER_MATCH):
            _cpp.build_actor_obs(self._sim, slot, self._objective_actor_obs_buf[slot])
            obs = self._objective_actor_obs_buf[slot]
            alive = float(obs[hp_slice][0]) > 0.0
            on_point = float(obs[on_point_slice][0]) > 0.5
            if not alive or not on_point:
                continue
            team = self._team_for_slot(slot)
            out[team]["on_point_count"] += 1.0
            own_pos = self._slot_position(critic, slot)
            visible = list(_cpp.observable_enemy_slots(self._sim, slot))
            nearest: float | None = None
            for enemy in self._enemy_slots_for(slot):
                if not self._slot_alive(critic, enemy):
                    continue
                enemy_pos = self._slot_position(critic, enemy)
                dist = float(np.linalg.norm(enemy_pos - own_pos))
                if nearest is None or dist < nearest:
                    nearest = dist
                if visible[enemy]:
                    out[team]["los_count"] += 1.0
                    break
            if nearest is not None:
                out[team]["distance_sum"] += nearest
                out[team]["distance_count"] += 1.0
        return out

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
            # Gymnasium vector info collation stores Python ints in a C-long
            # array; uint64 hashes can overflow there, so expose hex text.
            "state_hash": f"0x{int(s.state_hash):016x}",
            "team_a_score": float(s.team_a_score),
            "team_b_score": float(s.team_b_score),
            "team_a_kills": int(s.team_a_kills),
            "team_b_kills": int(s.team_b_kills),
            "winner": winner_str,
            "learner_team": self._learner_team_str,
            "objective_unlock_ticks": int(s.objective_unlock_ticks),
            "objective_capture_ticks": int(s.objective_capture_ticks),
            "objective_unlock_seconds": float(s.objective_unlock_ticks)
            / float(_cpp.TICK_HZ),
            "objective_capture_seconds": float(s.objective_capture_ticks)
            / float(_cpp.TICK_HZ),
            "respawn_ticks": int(self._applied_respawn_ticks or 0),
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
            "contested_majority_fire_commands": 0,
            "contested_majority_damage_hits": 0,
            "contested_majority_damage_centi_hp": 0,
        }

    def _combat_metrics_before_step(self, actions: list[_cpp.Action]) -> dict[str, Any]:
        assert self._sim is not None
        critic = np.zeros(CRITIC_DIM, dtype=np.float32)
        _cpp.build_critic_obs(self._sim, _cpp.Team.A, critic)
        metrics = {
            "A": self._empty_team_combat_metrics(),
            "B": self._empty_team_combat_metrics(),
        }
        objective_before = self._objective_snapshot()
        contested_majority_team: str | None = None
        on_a = int(objective_before["alive_on_point_a"])
        on_b = int(objective_before["alive_on_point_b"])
        if on_a > 0 and on_b > 0:
            if on_a > on_b:
                contested_majority_team = "A"
            elif on_b > on_a:
                contested_majority_team = "B"
        for slot, action in enumerate(actions):
            if not action.primary_fire:
                continue
            team = self._team_for_slot(slot)
            team_metrics = metrics[team]
            team_metrics["fire_commands"] += 1
            if contested_majority_team == team:
                team_metrics["contested_majority_fire_commands"] += 1
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
        objective_before = self._objective_snapshot()
        contested_majority_team: str | None = None
        on_a = int(objective_before["alive_on_point_a"])
        on_b = int(objective_before["alive_on_point_b"])
        if on_a > 0 and on_b > 0:
            if on_a > on_b:
                contested_majority_team = "A"
            elif on_b > on_a:
                contested_majority_team = "B"
        for team in ("A", "B"):
            team_metrics = combat_metrics[team]
            for slot in self._team_slots(team):
                delta = int(damage_delta[slot])
                if delta <= 0:
                    continue
                team_metrics["damage_hits"] += 1
                team_metrics["damage_centi_hp"] += delta
                if contested_majority_team == team:
                    team_metrics["contested_majority_damage_hits"] += 1
                    team_metrics["contested_majority_damage_centi_hp"] += delta
