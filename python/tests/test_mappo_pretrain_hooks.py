from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from train.mappo_pretrain_hooks import (
    maybe_run_bc_pretrain,
    maybe_run_composition_pretrain,
    maybe_run_full_env_rehearsal,
    maybe_run_multi_enemy_supervised_bridge,
    maybe_warm_start,
)


def _context(**run_overrides):
    run_cfg = {
        "composition_pretrain": False,
        "bc_pretrain_steps": 0,
        **run_overrides,
    }
    return SimpleNamespace(run_cfg=run_cfg, phase_label="phase4")


def _warm_context(**run_overrides):
    return SimpleNamespace(
        run_cfg=run_overrides,
        phase_label="phase4",
        cfg=SimpleNamespace(
            aim_aux_coef=0.0,
            target_selection_dim=0,
            target_conditioned_combat=False,
            mode_gated_combat=False,
        ),
    )


class _WarmStartToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.compatible = nn.Parameter(torch.ones(2))
        self.actor_entity_encoder = nn.Linear(2, 2)
        self.actor_body = nn.Linear(2, 2)


def test_warm_start_noops_without_checkpoint() -> None:
    maybe_warm_start(_context(), SimpleNamespace(model=object()))


def test_default_same_topology_warm_start_remains_strict(tmp_path) -> None:
    ckpt = tmp_path / "bad_shape.pt"
    torch.save(
        {
            "model_state_dict": {
                "compatible": torch.zeros(3),
            }
        },
        ckpt,
    )

    with pytest.raises(RuntimeError, match="size mismatch"):
        maybe_warm_start(
            _warm_context(init_from_checkpoint=str(ckpt)),
            SimpleNamespace(model=_WarmStartToyModel()),
        )


def test_warm_start_migration_loads_compatible_and_reports_skipped(tmp_path, capsys) -> None:
    ckpt = tmp_path / "flat_to_entity_grid.pt"
    torch.save(
        {
            "model_state_dict": {
                "compatible": torch.full((2,), 7.0),
                "actor_entity_encoder.weight": torch.zeros(3, 2),
                "actor_embed.0.weight": torch.zeros(2, 2),
            }
        },
        ckpt,
    )
    model = _WarmStartToyModel()

    maybe_warm_start(
        _warm_context(
            init_from_checkpoint=str(ckpt),
            warm_start_migration="compatible_exact",
        ),
        SimpleNamespace(model=model),
    )

    torch.testing.assert_close(model.compatible.detach(), torch.full((2,), 7.0))
    assert tuple(model.actor_entity_encoder.weight.shape) == (2, 2)
    out = capsys.readouterr().out
    assert "warm-start migration=compatible_exact" in out
    assert "actor_embed.0.weight" in out
    assert "actor_entity_encoder.weight(checkpoint=(3, 2), model=(2, 2))" in out


def test_warm_start_migration_resets_log_std_to_config_init(tmp_path) -> None:
    """Migration never carries the checkpoint's log_std: it is exploration
    tuned for the OLD architecture. The 2026-08-09 head-width probe migrated
    v5-0300's sharpened log_std onto fresh heads and exploration flatlined
    for 300 updates."""

    class _LogStdToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.compatible = nn.Parameter(torch.ones(2))
            self.log_std = nn.Parameter(torch.full((3,), -1.0))

    ckpt = tmp_path / "sharpened.pt"
    torch.save(
        {
            "model_state_dict": {
                "compatible": torch.full((2,), 7.0),
                "log_std": torch.full((3,), -2.14),
            }
        },
        ckpt,
    )
    model = _LogStdToyModel()

    maybe_warm_start(
        _warm_context(
            init_from_checkpoint=str(ckpt),
            warm_start_migration="compatible_exact",
        ),
        SimpleNamespace(model=model),
    )

    torch.testing.assert_close(model.compatible.detach(), torch.full((2,), 7.0))
    torch.testing.assert_close(model.log_std.detach(), torch.full((3,), -1.0))


def test_warm_start_migration_does_not_load_same_name_shape_mismatch(
    tmp_path, capsys
) -> None:
    ckpt = tmp_path / "shape_mismatch.pt"
    torch.save(
        {
            "model_state_dict": {
                "actor_body.bias": torch.full((3,), -5.0),
            }
        },
        ckpt,
    )
    model = _WarmStartToyModel()
    before = model.actor_body.bias.detach().clone()

    maybe_warm_start(
        _warm_context(
            init_from_checkpoint=str(ckpt),
            warm_start_migration="compatible_exact",
        ),
        SimpleNamespace(model=model),
    )

    torch.testing.assert_close(model.actor_body.bias.detach(), before)
    assert "actor_body.bias(checkpoint=(3,), model=(2,))" in capsys.readouterr().out


def test_warm_start_rejects_unknown_migration_mode(tmp_path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": {}}, ckpt)

    with pytest.raises(ValueError, match="unsupported warm_start_migration"):
        maybe_warm_start(
            _warm_context(init_from_checkpoint=str(ckpt), warm_start_migration="relaxed"),
            SimpleNamespace(model=_WarmStartToyModel()),
        )


def test_composition_pretrain_noops_when_disabled() -> None:
    result = maybe_run_composition_pretrain(
        _context(composition_pretrain=False),
        SimpleNamespace(model=object()),
        SimpleNamespace(),
    )

    assert result is True


def test_full_env_rehearsal_noops_when_disabled() -> None:
    result = maybe_run_full_env_rehearsal(
        _context(full_env_rehearsal={"enabled": False}),
        SimpleNamespace(model=object()),
        SimpleNamespace(),
    )

    assert result is True


def test_multi_enemy_supervised_bridge_noops_when_disabled() -> None:
    result = maybe_run_multi_enemy_supervised_bridge(
        _context(multi_enemy_supervised_bridge={"enabled": False}),
        SimpleNamespace(model=object()),
        SimpleNamespace(),
    )

    assert result is True


def test_multi_enemy_supervised_bridge_rejects_non_opt_in_shape() -> None:
    context = SimpleNamespace(
        run_cfg={"multi_enemy_supervised_bridge": {"enabled": True}},
        cfg=SimpleNamespace(obs_encoder="flat", target_action_dim=0),
    )

    with pytest.raises(ValueError, match="entity_attention_grid"):
        maybe_run_multi_enemy_supervised_bridge(
            context,
            SimpleNamespace(model=object()),
            SimpleNamespace(),
        )


def test_multi_enemy_supervised_bridge_dispatches_closed_loop_opt_in(
    tmp_path, monkeypatch
) -> None:
    calls: list[str] = []

    def fake_closed_loop(model, env_fn, config):
        calls.append("closed_loop")
        assert config["closed_loop"]["enabled"] is True
        return {"loss": 0.25, "final_agreement_fire_accuracy": 0.75}

    def fake_gate(model, env_fn, *, gate, output_dir, seed, checkpoint_path):
        calls.append("gate")
        return SimpleNamespace(
            status="NOT_REACHED",
            passed=False,
            path=tmp_path / "gate.json",
            metrics={
                "team_a_visible_fire_rate": 1.0,
                "team_a_hit_fire": 0.0,
                "objective_on_point": 0.0,
                "mean_score_a": 0.0,
                "losses": 50.0,
            },
            thresholds={
                "min_team_a_visible_fire_rate": 0.01,
                "min_team_a_hit_fire": 0.04,
                "min_objective_on_point": 0.25,
                "min_mean_score_a": 1.0,
                "max_losses": 49.0,
            },
        )

    monkeypatch.setattr(
        "train.mappo_pretrain_hooks.closed_loop_supervised_bridge_pretrain",
        fake_closed_loop,
    )
    monkeypatch.setattr("train.mappo_pretrain_hooks.run_full_env_rehearsal_gate", fake_gate)
    monkeypatch.setattr(
        "train.mappo_pretrain_hooks.save_mappo_checkpoint", lambda *args, **kwargs: None
    )

    context = SimpleNamespace(
        run_cfg={
            "multi_enemy_supervised_bridge": {
                "enabled": True,
                "teacher": "multi_enemy_visible",
                "closed_loop": {"enabled": True, "rounds": 2, "updates_per_round": 3},
                "gate": {},
            }
        },
        cfg=SimpleNamespace(obs_encoder="entity_attention_grid", target_action_dim=0),
        env_fn=lambda: None,
        seed_base=123,
        output_dir=tmp_path,
        ckpt_env_cfg={},
        phase=4,
        phase_label="phase4",
    )
    trainer = SimpleNamespace(model=SimpleNamespace(state_dict=lambda: {}, cfg=SimpleNamespace()))
    logger = SimpleNamespace(log=lambda *args, **kwargs: None)

    assert maybe_run_multi_enemy_supervised_bridge(context, trainer, logger) is False
    assert calls == ["closed_loop", "gate"]


def test_bc_pretrain_noops_when_steps_zero() -> None:
    result = maybe_run_bc_pretrain(
        _context(bc_pretrain_steps=0),
        SimpleNamespace(model=object()),
        SimpleNamespace(),
    )

    assert result is True
