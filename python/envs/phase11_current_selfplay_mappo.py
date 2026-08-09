"""Six-agent current-vs-current MAPPO env.

The policy controls both teams in one match. Rewards and critic observations
stay per-team/per-agent so the trainer can learn from opposing returns without
averaging them into a single zero-sum scalar.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.phase4_mappo import Phase4MappoEnv
from xushi2 import xushi2_cpp as _cpp
from xushi2.multi_enemy_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM
from xushi2.map_randomization import (
    map_layout_hash,
    randomized_cover_markers,
    randomized_map_bounds,
    randomized_wall_segments,
    sim_cfg_with_map_bounds,
)
from xushi2.multi_enemy_obs import (
    actor_obs_to_multi_enemy_entity_grid_obs,
    normalize_world_for_team,
)
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM, actor_field_slice, critic_field_slice
from xushi2.reward import RewardCalculator
from xushi2.runner import _build_config
from xushi2.self_play_schedule import SelfPlayMatch, SelfPlaySchedule
from xushi2.snapshot_policy import SnapshotPolicy

__all__ = ["Phase11CurrentSelfplayMappoEnv"]

_AGENTS_PER_MATCH = _cpp.AGENTS_PER_MATCH
_PHASE11_PER_AGENT_REWARDS = False


class Phase11CurrentSelfplayMappoEnv(gym.Env):
    """3v3 current policy self-play with six controlled action rows."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    n_agents: int = _AGENTS_PER_MATCH
    actor_obs_dim: int = MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    critic_obs_dim: int = CRITIC_DIM
    action_dim: int = 6

    # See xushi2.env_capabilities. Every curriculum knob must be implemented or
    # declared here; silently lacking one drops the curriculum for a whole run.
    UNSUPPORTED_CURRICULUM_SETTERS: ClassVar[dict[str, str]] = {
        "set_team_spirit": (
            "team_spirit only shapes the per-agent reward path, and this env "
            "pins per_agent_rewards=False because step() duplicates team-scalar "
            "rewards across each team's three rows"
        ),
    }

    def __init__(
        self,
        sim_cfg: dict,
        *,
        reward_cfg: dict[str, Any] | None = None,
        fog_mode: str = "team_shared",
        visible_radius: float = 0.65,
        map_randomization: dict[str, Any] | None = None,
        self_play_schedule: dict[str, Any] | None = None,
        snapshot_league: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._base_sim_cfg = dict(sim_cfg)
        self._reward_cfg = dict(reward_cfg or {})
        cfg_per_agent = self._reward_cfg.get("per_agent_rewards")
        if cfg_per_agent is not None and bool(cfg_per_agent) != _PHASE11_PER_AGENT_REWARDS:
            raise ValueError(
                "Phase11CurrentSelfplayMappoEnv requires reward_cfg['per_agent_rewards']=False "
                "because step() duplicates team-scalar rewards across each team's three rows"
            )
        self._reward_cfg["per_agent_rewards"] = _PHASE11_PER_AGENT_REWARDS
        self._reward_calc = RewardCalculator(**self._reward_cfg)
        self._fog_mode = str(fog_mode)
        self._team_shared = self._fog_mode == "team_shared"
        self._visible_radius = float(visible_radius)
        self._map_randomization = dict(map_randomization or {})
        self._schedule = (
            SelfPlaySchedule(weights={"current": 1.0})
            if self_play_schedule is None
            else SelfPlaySchedule.from_config(self_play_schedule, snapshot_league)
        )
        self._sim: _cpp.Sim | None = None
        self._opponent_policy: SnapshotPolicy | None = None
        self._flat_actor_obs = np.zeros((_AGENTS_PER_MATCH, ACTOR_PHASE1_DIM), dtype=np.float32)
        self._last_map_bounds: dict[str, float] | None = None
        self._last_cover_markers: list[dict[str, float]] = []
        self._last_wall_segments: list[dict[str, float]] = []
        self._last_layout_hash: str | None = None
        self._last_match = SelfPlayMatch(match_type="current", group="current")
        self._last_loss_mask = np.ones(_AGENTS_PER_MATCH, dtype=np.float32)
        self._last_opponent_actions = np.zeros((3, 6), dtype=np.float32)
        self._last_seen_enemy_position = np.zeros((_AGENTS_PER_MATCH, 3, 2), dtype=np.float32)
        self._last_seen_valid = np.zeros((_AGENTS_PER_MATCH, 3), dtype=bool)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(_AGENTS_PER_MATCH, MULTI_ENEMY_ENTITY_GRID_OBS_DIM),
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

    def set_majority_on_point_alpha(self, value: float) -> None:
        self._reward_calc.set_majority_on_point_alpha(float(value))

    def set_uncontested_on_point_alpha(self, value: float) -> None:
        self._reward_calc.set_uncontested_on_point_alpha(float(value))

    def set_objective_timing_ticks(self, unlock_ticks: int, capture_ticks: int) -> None:
        """Objective-timing curriculum knob.

        Mirrors Phase4MappoEnv: the new timing is written into the base sim
        config so future resets pick it up, and pushed into the live sim so the
        in-flight episode also honors it.
        """
        unlock = int(unlock_ticks)
        capture = int(capture_ticks)
        if unlock <= 0 or capture <= 0:
            raise ValueError(
                f"objective timing ticks must be >0, got unlock={unlock} capture={capture}"
            )
        self._base_sim_cfg.pop("objective_unlock_seconds", None)
        self._base_sim_cfg.pop("objective_capture_seconds", None)
        nested = self._base_sim_cfg.get("objective_timing")
        if isinstance(nested, dict):
            nested.pop("unlock_seconds", None)
            nested.pop("capture_seconds", None)
        self._base_sim_cfg["objective_unlock_ticks"] = unlock
        self._base_sim_cfg["objective_capture_ticks"] = capture
        if self._sim is not None:
            self._sim.set_objective_timing_ticks(unlock, capture)

    def set_objective_timing_seconds(
        self, unlock_seconds: float, capture_seconds: float
    ) -> None:
        self.set_objective_timing_ticks(
            round(float(unlock_seconds) * float(_cpp.TICK_HZ)),
            round(float(capture_seconds) * float(_cpp.TICK_HZ)),
        )

    def set_respawn_ticks(self, respawn_ticks: int) -> None:
        """Respawn-time curriculum knob.

        As in Phase4MappoEnv there is no live-sim setter (respawn_tick is
        stamped at death), so the new value applies from the next reset onward.
        """
        ticks = int(respawn_ticks)
        if ticks <= 0:
            raise ValueError(f"respawn ticks must be >0, got {ticks}")
        mechanics = dict(self._base_sim_cfg.get("mechanics", {}))
        mechanics["respawn_ticks"] = ticks
        self._base_sim_cfg["mechanics"] = mechanics

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if "team_size" in self._base_sim_cfg:
            raise ValueError("sim_cfg must not carry 'team_size'; the env owns this knob")

        seed_int = 0 if seed is None else int(seed)
        bounds = randomized_map_bounds(seed_int, self._map_randomization)
        covers = randomized_cover_markers(seed_int, self._map_randomization)
        walls = randomized_wall_segments(seed_int, self._map_randomization)
        layout_hash = map_layout_hash(bounds, covers, walls)
        match = self._schedule.sample(seed_int)
        self._last_map_bounds = bounds
        self._last_cover_markers = covers
        self._last_wall_segments = walls
        self._last_layout_hash = layout_hash
        self._last_match = match
        self._last_opponent_actions = np.zeros((3, 6), dtype=np.float32)
        self._last_seen_enemy_position[:] = 0.0
        self._last_seen_valid[:] = False
        self._last_loss_mask = (
            np.ones(_AGENTS_PER_MATCH, dtype=np.float32)
            if match.match_type == "current"
            else np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
        )

        sim_cfg = sim_cfg_with_map_bounds(self._base_sim_cfg, bounds)
        sim_cfg["randomize_map"] = True
        sim_cfg["cover_circles"] = [dict(marker) for marker in covers]
        sim_cfg["wall_segments"] = [dict(wall) for wall in walls]
        cfg = _build_config(sim_cfg, seed_override=seed)
        cfg.team_size = 3
        self._sim = _cpp.Sim(cfg)
        self._opponent_policy = (
            SnapshotPolicy(match.snapshot_path)
            if match.snapshot_path is not None and match.match_type != "current"
            else None
        )
        if self._opponent_policy is not None:
            self._opponent_policy.reset(batch_size=3)
        self._reward_calc.reset(self._sim)
        self._build_actor_obs_all()
        return self._convert_obs(), self._make_info()

    def step(self, action: np.ndarray):
        if self._sim is None:
            raise RuntimeError("reset() must be called before step()")
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 2 or action.shape[0] != _AGENTS_PER_MATCH or action.shape[1] < 6:
            raise ValueError(f"action shape must be ({_AGENTS_PER_MATCH}, >=6), got {action.shape}")

        actions = [
            Phase4MappoEnv._action_to_cpp_for_slot(action[slot], slot)
            for slot in range(_AGENTS_PER_MATCH)
        ]
        self._last_opponent_actions = np.zeros((3, 6), dtype=np.float32)
        if self._last_match.match_type != "current":
            if self._opponent_policy is not None:
                opponent = np.asarray(
                    self._opponent_policy.act(
                        self._sim,
                        (3, 4, 5),
                        map_bounds=dict(self._last_map_bounds or {}),
                    ),
                    dtype=np.float32,
                )
                if opponent.shape[0] != 3 or opponent.shape[1] < 6:
                    raise ValueError(
                        "Phase-11 snapshot opponent must emit at least six "
                        f"controls per Team-B agent, got {opponent.shape}"
                    )
                opponent = opponent[:, :6]
                for idx, slot in enumerate((3, 4, 5)):
                    actions[slot] = Phase4MappoEnv._action_to_cpp_for_slot(opponent[idx], slot)
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
                            scripted.aim_delta / np.pi * 4.0,
                            float(scripted.primary_fire),
                            float(scripted.ability_1),
                            float(scripted.ability_2),
                        ],
                        dtype=np.float32,
                    )
        self._sim.step_decision(actions)

        r_a, r_b = self._reward_calc.step(self._sim)
        team_a_step, team_b_step = self._coerce_team_scalar_rewards(r_a, r_b, source="step")
        team_rewards = [team_a_step] * 3 + [team_b_step] * 3
        # terminated == the MDP genuinely ended (a team reached the score
        # threshold); truncated == the round timer cut it off. Deriving
        # these from `winner` labelled a timeout-with-a-winner as
        # terminated and a draw as truncated, which inverts the common
        # case: reaching the score threshold is rare, timing out is not.
        terminated = bool(self._sim.score_threshold_reached)
        truncated = bool(self._sim.episode_over) and not terminated
        if terminated or truncated:
            ta, tb = self._reward_calc.add_terminal(self._sim)
            team_a_term, team_b_term = self._coerce_team_scalar_rewards(
                ta, tb, source="add_terminal"
            )
            for i in range(3):
                team_rewards[i] += team_a_term
                team_rewards[i + 3] += team_b_term

        self._build_actor_obs_all()
        info = self._make_info()
        info["reward_team_a"] = float(team_a_step)
        info["reward_team_b"] = float(team_b_step)
        return (
            self._convert_obs(),
            np.asarray(team_rewards, dtype=np.float32),
            terminated,
            truncated,
            info,
        )

    def _coerce_team_scalar_rewards(
        self, reward_a: Any, reward_b: Any, *, source: str
    ) -> tuple[float, float]:
        if _PHASE11_PER_AGENT_REWARDS:
            raise RuntimeError(
                "Phase-11 current self-play env expects scalar team rewards contract"
            )
        team_a = self._coerce_single_team_scalar_reward(reward_a, team_name="A", source=source)
        team_b = self._coerce_single_team_scalar_reward(reward_b, team_name="B", source=source)
        return team_a, team_b

    @staticmethod
    def _coerce_single_team_scalar_reward(value: Any, *, team_name: str, source: str) -> float:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 1 and arr.shape[0] == 1:
            return float(arr[0])
        raise ValueError(
            "Phase11CurrentSelfplayMappoEnv expected scalar team reward from "
            f"RewardCalculator.{source} for Team {team_name}, got shape {arr.shape}"
        )

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

    def close(self) -> None:
        self._sim = None

    def _build_actor_obs_all(self) -> None:
        for slot in range(_AGENTS_PER_MATCH):
            _cpp.build_actor_obs(self._sim, slot, self._flat_actor_obs[slot])

    def _convert_obs(self) -> np.ndarray:
        critic = np.zeros((_AGENTS_PER_MATCH, CRITIC_DIM), dtype=np.float32)
        self.build_critic_obs(critic.reshape(-1))
        visible = self._enemy_visibility_matrix(critic)
        for row in range(_AGENTS_PER_MATCH):
            for enemy_idx in range(3):
                if visible[row, enemy_idx]:
                    self._last_seen_enemy_position[row, enemy_idx] = self._enemy_norm_position(
                        critic[row],
                        enemy_idx,
                        team_b_view=row >= 3,
                    )
                    self._last_seen_valid[row, enemy_idx] = True
        return actor_obs_to_multi_enemy_entity_grid_obs(
            self._flat_actor_obs,
            critic_obs=critic,
            map_bounds=dict(self._last_map_bounds or {}),
            visible_radius=self._visible_radius,
            visible_override=visible,
            last_seen_enemy_position=self._last_seen_enemy_position,
            last_seen_valid=self._last_seen_valid,
            team_b_view=np.array([False, False, False, True, True, True]),
        )

    def _enemy_visibility_matrix(self, critic: np.ndarray) -> np.ndarray:
        if self._sim is None:
            raise RuntimeError("reset() must be called before converting obs")
        own_pos = self._flat_actor_obs[:, actor_field_slice("own_position")]
        enemy_pos = np.zeros((_AGENTS_PER_MATCH, 3, 2), dtype=np.float32)
        alive = np.zeros((_AGENTS_PER_MATCH, 3), dtype=bool)
        for row in range(_AGENTS_PER_MATCH):
            for enemy_idx in range(3):
                enemy_pos[row, enemy_idx] = self._enemy_norm_position(
                    critic[row],
                    enemy_idx,
                    team_b_view=row >= 3,
                )
                alive[row, enemy_idx] = (
                    critic[row, critic_field_slice(f"enemy{enemy_idx}/alive_flag")][0] > 0.5
                )
        radius = np.linalg.norm(enemy_pos - own_pos[:, None, :], axis=2) <= float(
            self._visible_radius
        )
        los = np.zeros((_AGENTS_PER_MATCH, 3), dtype=bool)
        for slot in range(_AGENTS_PER_MATCH):
            enemy_slots = range(3, 6) if slot < 3 else range(0, 3)
            if self._team_shared:
                ally_slots = range(0, 3) if slot < 3 else range(3, 6)
                for enemy_idx, enemy_slot in enumerate(enemy_slots):
                    los[slot, enemy_idx] = any(
                        bool(_cpp.observable_enemy_slots(self._sim, ally)[enemy_slot])
                        for ally in ally_slots
                    )
                    team_rows = slice(0, 3) if slot < 3 else slice(3, 6)
                    radius[slot, enemy_idx] = bool(radius[team_rows, enemy_idx].any())
            else:
                native = _cpp.observable_enemy_slots(self._sim, slot)
                for enemy_idx, enemy_slot in enumerate(enemy_slots):
                    los[slot, enemy_idx] = bool(native[enemy_slot])
        return alive & radius & los

    def _enemy_norm_position(
        self,
        critic: np.ndarray,
        enemy_idx: int,
        *,
        team_b_view: bool,
    ) -> np.ndarray:
        return normalize_world_for_team(
            critic[critic_field_slice(f"enemy{enemy_idx}/world_position")],
            dict(self._last_map_bounds or {}),
            team_b_view=team_b_view,
        )

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
            "state_hash": f"0x{int(s.state_hash):016x}",
            "team_a_score": float(s.team_a_score),
            "team_b_score": float(s.team_b_score),
            "team_a_kills": int(s.team_a_kills),
            "team_b_kills": int(s.team_b_kills),
            "winner": winner_str,
            "learner_team": "both" if self._last_match.match_type == "current" else "A",
            "match_type": self._last_match.match_type,
            "schedule": self._schedule.summary,
            "loss_mask": self._last_loss_mask.copy(),
            "opponent_actions": self._last_opponent_actions.copy(),
            "map_bounds": dict(self._last_map_bounds or {}),
            "cover_markers": [dict(marker) for marker in self._last_cover_markers],
            "wall_segments": [dict(wall) for wall in self._last_wall_segments],
            "map_layout_hash": self._last_layout_hash,
            "snapshot_path": self._last_match.snapshot_path,
            "snapshot_group": self._last_match.group,
            "anchor_bot": self._last_match.anchor_bot,
        }
