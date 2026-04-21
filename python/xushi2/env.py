"""Single-env Gymnasium wrapper for Phase-1 1v1 Ranger.

Team A slot 0 is the learner. Team B slot 3 is driven by a configurable
scripted bot. All other slots are unoccupied at Phase 1.

The action space is a Dict with six scalar components matching the
`Action` schema in docs/action_spec.md — continuous ``move_x``,
``move_y``, ``aim_delta`` plus three Bernoulli booleans. ``target_slot``
is omitted (Phase 1-9).

The observation space is a flat Box of shape ``(ACTOR_PHASE1_DIM,)``,
float32. Layout matches ``obs_manifest.py``.

Reward is computed by ``RewardCalculator`` (rl_design.md §5): terminal
±10 / 0, shaped events clipped to ±3 per episode, symmetrized.

Determinism: the env does not reseed the sim other than through the
caller-provided seed. Calling ``reset(seed=S)`` is equivalent to the
Phase-0 determinism harness rerun — two envs seeded identically and
stepped with identical action streams produce identical
``info["state_hash"]`` trajectories.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium is a required dependency (see python/pyproject.toml). "
        "Install with `pip install gymnasium`."
    ) from exc

from . import xushi2_cpp as _cpp
from .obs_manifest import ACTOR_PHASE1_DIM
from .reward import RewardCalculator
from .runner import _build_config

__all__ = ["XushiEnv", "VALID_OPPONENT_BOTS"]

# Must match the bot names exposed by the C++ layer. Matching the set
# allowed by xushi2.runner so one failure mode (typo) fails the same
# way in both entry points.
VALID_OPPONENT_BOTS: frozenset[str] = frozenset({
    "walk_to_objective", "hold_and_shoot", "basic", "noop"
})

_TEAM_A_RANGER_SLOT = 0
_TEAM_B_RANGER_SLOT = 3
_AGENTS_PER_MATCH = _cpp.AGENTS_PER_MATCH


class XushiEnv(gym.Env):
    """Gymnasium Env for 1v1 Ranger vs a scripted opponent.

    Parameters
    ----------
    sim_cfg
        Dict matching the schema consumed by ``runner._build_config`` —
        requires a ``mechanics`` block; missing keys raise ``KeyError``.
    opponent_bot
        Name of the scripted bot controlling Team B. One of
        ``VALID_OPPONENT_BOTS``. No default — explicit per project
        convention; raise ``ValueError`` on unknown.
    learner_team
        Team controlled by the caller's ``action`` arg. ``"A"`` or
        ``"B"``. Default ``"A"``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sim_cfg: dict,
        *,
        opponent_bot: str,
        learner_team: str = "A",
    ):
        super().__init__()

        if opponent_bot not in VALID_OPPONENT_BOTS:
            raise ValueError(
                f"unknown opponent_bot {opponent_bot!r}; "
                f"valid: {sorted(VALID_OPPONENT_BOTS)}"
            )
        if learner_team not in ("A", "B"):
            raise ValueError(f"learner_team must be 'A' or 'B', got {learner_team!r}")

        self._sim_cfg = dict(sim_cfg)  # shallow copy; defensive
        self._opponent_bot = opponent_bot
        self._learner_team_str = learner_team
        self._learner_team = _cpp.Team.A if learner_team == "A" else _cpp.Team.B
        self._learner_slot = (
            _TEAM_A_RANGER_SLOT if learner_team == "A" else _TEAM_B_RANGER_SLOT
        )
        self._opponent_slot = (
            _TEAM_B_RANGER_SLOT if learner_team == "A" else _TEAM_A_RANGER_SLOT
        )

        self._sim: _cpp.Sim | None = None
        self._reward_calc = RewardCalculator()

        # Observation buffer, reused across steps — zero-copy into C++.
        self._obs_buf = np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32)

        self.action_space = spaces.Dict({
            "move_x":       spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
            "move_y":       spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
            "aim_delta":    spaces.Box(
                -np.pi / 4.0, np.pi / 4.0, shape=(), dtype=np.float32),
            "primary_fire": spaces.Discrete(2),
            "ability_1":    spaces.Discrete(2),
            "ability_2":    spaces.Discrete(2),
        })
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(ACTOR_PHASE1_DIM,), dtype=np.float32,
        )

    # --- Gymnasium API ---

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        cfg = _build_config(self._sim_cfg, seed_override=seed)
        self._sim = _cpp.Sim(cfg)
        self._reward_calc.reset(self._sim)

        _cpp.build_actor_obs(self._sim, self._learner_slot, self._obs_buf)
        info = self._make_info()
        return self._obs_buf.copy(), info

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._sim is None:
            raise RuntimeError("reset() must be called before step()")

        # Build the full 6-slot action list. Slots not occupied at Phase 1
        # still pass a no-op Action (the sim ignores non-present slots).
        actions_list: list[_cpp.Action] = [_cpp.Action() for _ in range(_AGENTS_PER_MATCH)]
        actions_list[self._learner_slot] = self._action_from_dict(action)
        actions_list[self._opponent_slot] = _cpp.scripted_bot_action(
            self._sim, self._opponent_slot, self._opponent_bot
        )

        self._sim.step_decision(actions_list)

        # Shaped reward from counter deltas.
        r_a, r_b = self._reward_calc.step(self._sim)
        step_reward = r_a if self._learner_team_str == "A" else r_b

        terminated = bool(self._sim.episode_over) and (
            self._sim.winner != _cpp.Team.Neutral
        )
        truncated = bool(self._sim.episode_over) and (
            self._sim.winner == _cpp.Team.Neutral
        )

        if terminated or truncated:
            ta, tb = self._reward_calc.add_terminal(self._sim)
            step_reward += ta if self._learner_team_str == "A" else tb

        _cpp.build_actor_obs(self._sim, self._learner_slot, self._obs_buf)
        info = self._make_info()
        # Publish both-team rewards so trainers that want them can symmetrize.
        info["reward_team_a"] = float(r_a)
        info["reward_team_b"] = float(r_b)

        return (
            self._obs_buf.copy(), float(step_reward),
            bool(terminated), bool(truncated), info,
        )

    def close(self) -> None:
        self._sim = None

    # --- helpers ---

    def _action_from_dict(self, a: dict[str, Any]) -> _cpp.Action:
        act = _cpp.Action()
        act.move_x       = float(a["move_x"])
        act.move_y       = float(a["move_y"])
        act.aim_delta    = float(a["aim_delta"])
        act.primary_fire = bool(a["primary_fire"])
        act.ability_1    = bool(a["ability_1"])
        act.ability_2    = bool(a["ability_2"])
        # target_slot stays 0 — not used pre-Phase 10.
        return act

    def _make_info(self) -> dict[str, Any]:
        return {
            "tick":         int(self._sim.tick),
            "state_hash":   int(self._sim.state_hash),
            "team_a_score": float(self._sim.team_a_score),
            "team_b_score": float(self._sim.team_b_score),
            "team_a_kills": int(self._sim.team_a_kills),
            "team_b_kills": int(self._sim.team_b_kills),
        }
