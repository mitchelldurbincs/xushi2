"""Curriculum-capability declaration and enforcement.

Regression cover for the failure recorded in
``envs/phase4_multi_enemy_mappo.py``: the vector env used to discover
curriculum setters with ``getattr(env, name, None)`` and skip envs that lacked
them, so a configured objective-timing anneal, team-spirit ramp, and eval
overrides were dropped for every multi-enemy run with no signal at all.
"""

from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest

from envs.phase4_aim_only_mappo import Phase4AimOnlyMappoEnv
from envs.phase4_cap_duel_mappo import Phase4CapDuelMappoEnv
from envs.phase4_combat_1v1_mappo import Phase4Combat1v1MappoEnv
from envs.phase4_mappo import Phase4MappoEnv
from envs.phase11_current_selfplay_mappo import Phase11CurrentSelfplayMappoEnv
from xushi2.env_capabilities import (
    CURRICULUM_SETTERS,
    UnsupportedCurriculumError,
    declared_unsupported_setters,
    require_curriculum_setters,
    resolve_curriculum_setter,
    supported_curriculum_setters,
)
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM
from xushi2.vector_env import XushiVectorEnv


def _make_phase4():
    return Phase4MappoEnv(_sim_cfg(), opponent_bot="noop")


def _make_phase11():
    return Phase11CurrentSelfplayMappoEnv(_sim_cfg())


# Every env the runtime factory can hand to a vector wrapper. Parametrizing by
# factory rather than class keeps the "has every env taken a position?" test
# honest as new envs are added.
ENV_FACTORIES = [
    ("Phase4MappoEnv", _make_phase4),
    ("Phase11CurrentSelfplayMappoEnv", _make_phase11),
    ("Phase4CapDuelMappoEnv", Phase4CapDuelMappoEnv),
    ("Phase4Combat1v1MappoEnv", Phase4Combat1v1MappoEnv),
    ("Phase4AimOnlyMappoEnv", Phase4AimOnlyMappoEnv),
]


def _sim_cfg() -> dict:
    return {
        "seed": 7,
        "round_length_seconds": 4,
        "fog_of_war_enabled": False,
        "randomize_map": False,
        "action_repeat": 3,
        "mechanics": {
            "revolver_damage_centi_hp": 7500,
            "revolver_fire_cooldown_ticks": 15,
            "revolver_hitbox_radius": 0.75,
            "respawn_ticks": 120,
        },
    }


class _SilentEnv(gym.Env):
    """An env that neither implements nor declares the curriculum setters.

    This is the shape that used to be silently skipped.
    """

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3, ACTOR_PHASE1_DIM), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3, 6), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        return np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32), {}

    def build_critic_obs(self, out: np.ndarray) -> None:
        out[:] = 0.0


class _PartialEnv(_SilentEnv):
    """Implements one knob and declares the rest unsupported."""

    UNSUPPORTED_CURRICULUM_SETTERS: ClassVar[dict[str, str]] = {
        name: "test env" for name in CURRICULUM_SETTERS if name != "set_team_spirit"
    }

    def __init__(self) -> None:
        super().__init__()
        self.team_spirit = None

    def set_team_spirit(self, value: float) -> None:
        self.team_spirit = float(value)


# --- declaration handling ----------------------------------------------


def test_undeclared_setter_raises_instead_of_being_skipped():
    env = _SilentEnv()
    with pytest.raises(AttributeError, match=r"neither implements .* nor declares it"):
        resolve_curriculum_setter(env, "set_team_spirit")


def test_declared_unsupported_setter_resolves_to_none():
    env = _PartialEnv()
    assert resolve_curriculum_setter(env, "set_respawn_ticks") is None
    assert resolve_curriculum_setter(env, "set_team_spirit") is not None


def test_supported_setters_reflects_declarations():
    assert supported_curriculum_setters(_PartialEnv()) == frozenset({"set_team_spirit"})


def test_unknown_setter_name_is_rejected():
    with pytest.raises(ValueError, match="unknown curriculum setter"):
        resolve_curriculum_setter(_PartialEnv(), "set_not_a_knob")


def test_declaration_naming_an_unknown_setter_is_rejected():
    class _Bad(_SilentEnv):
        UNSUPPORTED_CURRICULUM_SETTERS: ClassVar[dict[str, str]] = {"set_nonsense": "typo"}

    with pytest.raises(ValueError, match="names unknown setters"):
        declared_unsupported_setters(_Bad())


# --- every shipped env has taken a position ----------------------------


@pytest.mark.parametrize(
    "make_env", [f for _, f in ENV_FACTORIES], ids=[n for n, _ in ENV_FACTORIES]
)
def test_every_env_implements_or_declares_every_setter(make_env):
    """No shipped env may leave a curriculum knob undecided.

    supported_curriculum_setters raises on any knob that is neither
    implemented nor declared, so reaching the assert means all are accounted
    for.
    """
    env = make_env()
    try:
        supported = supported_curriculum_setters(env)
        declared = declared_unsupported_setters(env)
        assert supported | set(declared) == set(CURRICULUM_SETTERS)
    finally:
        env.close()


def test_phase4_supports_every_curriculum_setter():
    env = Phase4MappoEnv(_sim_cfg(), opponent_bot="noop")
    try:
        assert supported_curriculum_setters(env) == frozenset(CURRICULUM_SETTERS)
    finally:
        env.close()


def test_phase11_supports_everything_except_team_spirit():
    env = Phase11CurrentSelfplayMappoEnv(_sim_cfg())
    try:
        supported = supported_curriculum_setters(env)
        # team_spirit only shapes the per-agent path, which Phase 11 pins off.
        assert "set_team_spirit" not in supported
        # The rest reach a real RewardCalculator / C++ Sim and must work.
        assert "set_objective_timing_seconds" in supported
        assert "set_respawn_ticks" in supported
        assert "set_majority_on_point_alpha" in supported
        assert "set_uncontested_on_point_alpha" in supported
    finally:
        env.close()


def test_phase11_objective_timing_setter_reaches_the_sim():
    env = Phase11CurrentSelfplayMappoEnv(_sim_cfg())
    try:
        env.reset(seed=3)
        env.set_objective_timing_seconds(1.0, 2.0)
        assert env._base_sim_cfg["objective_unlock_ticks"] == 30
        assert env._base_sim_cfg["objective_capture_ticks"] == 60
    finally:
        env.close()


def test_phase11_respawn_setter_reaches_the_next_reset():
    env = Phase11CurrentSelfplayMappoEnv(_sim_cfg())
    try:
        env.set_respawn_ticks(45)
        assert env._base_sim_cfg["mechanics"]["respawn_ticks"] == 45
    finally:
        env.close()


# --- vector env wiring --------------------------------------------------


def test_vector_env_rejects_env_that_never_declared_capabilities():
    with pytest.raises(AttributeError, match="neither implements"):
        XushiVectorEnv([_SilentEnv], critic_obs_dim=CRITIC_DIM)


def test_vector_env_intersects_support_across_envs():
    vec = XushiVectorEnv([_PartialEnv, _PartialEnv], critic_obs_dim=CRITIC_DIM)
    try:
        assert vec.supported_curriculum_setters() == frozenset({"set_team_spirit"})
        # A declared-unsupported knob is a no-op, not an error.
        vec.set_respawn_ticks(60)
        vec.set_team_spirit(0.5)
        assert all(env.team_spirit == pytest.approx(0.5) for env in vec.envs)
    finally:
        vec.close()


# --- startup gate -------------------------------------------------------


def test_require_curriculum_setters_passes_when_supported():
    require_curriculum_setters(
        frozenset(CURRICULUM_SETTERS), ["set_team_spirit"], context="test"
    )


def test_require_curriculum_setters_raises_naming_the_missing_knob():
    with pytest.raises(UnsupportedCurriculumError) as exc:
        require_curriculum_setters(
            frozenset({"set_team_spirit"}),
            ["set_team_spirit", "set_respawn_ticks"],
            context="training config",
        )
    message = str(exc.value)
    assert "set_respawn_ticks" in message
    assert "training config" in message
    # The message must say why running anyway is not acceptable.
    assert "never actually applied" in message


# --- declarations beat inherited methods --------------------------------


def test_declaration_wins_over_an_inherited_setter():
    """A subclass can inherit a setter its own config makes inert.

    Phase4CurrentSelfplayMappoEnv inherits set_team_spirit from
    Phase4MappoEnv but builds its RewardCalculator on the scalar path, which
    team_spirit does not shape. If the inherited method won, the knob would
    report as supported and the ramp would be discarded anyway.
    """
    from envs.phase4_selfplay_mappo import Phase4CurrentSelfplayMappoEnv

    env = Phase4CurrentSelfplayMappoEnv(_sim_cfg())
    try:
        assert hasattr(env, "set_team_spirit")  # inherited, and present
        assert resolve_curriculum_setter(env, "set_team_spirit") is None
        assert "set_team_spirit" not in supported_curriculum_setters(env)
    finally:
        env.close()


def test_scalar_reward_calculator_rejects_a_nonzero_team_spirit():
    """Defense in depth for callers that bypass the vector env."""
    from xushi2.reward import RewardCalculator

    rc = RewardCalculator()  # per_agent_rewards=False
    rc.set_team_spirit(0.0)  # pushed unconditionally every update; must pass
    with pytest.raises(ValueError, match="requires per_agent_rewards=True"):
        rc.set_team_spirit(0.5)
