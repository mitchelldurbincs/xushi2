"""make_mappo_match_env must not silently drop what the caller asked for.

Each branch of the factory honors a different subset of its parameters. The
rest used to be dropped without a word, so a config asking for team-shared fog
on a self-play run produced a complete run with fog off and entirely plausible
metrics.
"""

from __future__ import annotations

import logging

import pytest

from envs.runtime_factory import make_mappo_match_env, mappo_env_fn_from_config


def _sim_cfg() -> dict:
    return {
        "seed": 11,
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


# --- semantic drops raise ------------------------------------------------


def test_self_play_rejects_fog_it_cannot_apply():
    with pytest.raises(ValueError, match=r"cannot honor.*fog_mode"):
        make_mappo_match_env(sim_cfg=_sim_cfg(), self_play=True, fog_mode="team_shared")


def test_self_play_rejects_a_widened_actor_obs():
    # runtime_specs would size the model at MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    # while this branch returns a flat 31-dim env.
    with pytest.raises(ValueError, match=r"cannot honor.*actor_obs"):
        make_mappo_match_env(
            sim_cfg=_sim_cfg(), self_play=True, actor_obs="multi_enemy_entity_grid"
        )


def test_cap_duel_rejects_map_randomization():
    with pytest.raises(ValueError, match=r"cannot honor.*map_randomization"):
        make_mappo_match_env(
            sim_cfg=_sim_cfg(),
            mini_game="cap_duel",
            map_randomization={"enabled": True},
        )


def test_flat_variant_rejects_fog_mode_it_does_not_read():
    with pytest.raises(ValueError, match=r"cannot honor.*fog_mode"):
        make_mappo_match_env(sim_cfg=_sim_cfg(), fog_mode="team_shared")


def test_error_names_what_the_variant_does_honor():
    with pytest.raises(ValueError) as exc:
        make_mappo_match_env(sim_cfg=_sim_cfg(), self_play=True, fog_mode="team_shared")
    assert "this variant honors" in str(exc.value)


# --- cosmetic drops warn but run -----------------------------------------


def test_cap_duel_warns_about_ignored_opponent_bot_but_still_builds(caplog):
    # Nine committed cap_duel configs set opponent_bot, which this mini-game
    # replaces with its own scripted enemy. That makes the config a misleading
    # description, not a wrong experiment, so it warns.
    with caplog.at_level(logging.WARNING, logger="envs.runtime_factory"):
        env = make_mappo_match_env(
            sim_cfg=_sim_cfg(), mini_game="cap_duel", opponent_bot="weak_basic_v2"
        )
    try:
        assert "opponent_bot" in caplog.text
        assert "ignores" in caplog.text
    finally:
        env.close()


# --- values equal to their default carry no intent -----------------------


def test_default_valued_params_do_not_trip_the_guard():
    # phases.py passes every parameter positionally, so most calls supply
    # defaults for parameters the chosen variant ignores.
    env = make_mappo_match_env(
        sim_cfg=_sim_cfg(),
        mini_game="cap_duel",
        fog_mode="none",
        actor_obs="flat",
        map_randomization={},
    )
    env.close()


def test_self_play_accepts_the_redundant_six_agent_count():
    # phases.py passes n_agents=6 for self-play; the env is inherently
    # six-agent, so that is agreement rather than an override.
    env = make_mappo_match_env(sim_cfg=_sim_cfg(), self_play=True, n_agents=6)
    env.close()


def test_phase11_honors_fog_and_map_so_neither_is_rejected():
    env = make_mappo_match_env(
        sim_cfg=_sim_cfg(),
        n_agents=6,
        fog_mode="team_shared",
        map_randomization={"enabled": False},
    )
    env.close()


# --- dead config keys ----------------------------------------------------


def test_unimplemented_snapshot_paths_key_is_rejected():
    with pytest.raises(ValueError, match="snapshot_paths is not implemented"):
        mappo_env_fn_from_config({"sim": _sim_cfg(), "snapshot_paths": ["a.pt"]})


def test_unimplemented_target_slot_key_is_rejected():
    with pytest.raises(ValueError, match="target_slot is not implemented"):
        mappo_env_fn_from_config({"sim": _sim_cfg(), "features": {"target_slot": True}})


def test_falsey_dead_key_requests_nothing_and_is_allowed():
    # experiments/configs/runtime/mappo_flat_smoke.yaml carries
    # `features.target_slot: false`, which asks for nothing.
    fn = mappo_env_fn_from_config({"sim": _sim_cfg(), "features": {"target_slot": False}})
    env = fn()
    env.close()
