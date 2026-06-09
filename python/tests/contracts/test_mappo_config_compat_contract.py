import pytest
import torch
import yaml
from tests._paths import config_path

from train.mappo import make_mappo_config
from train.mappo_bc_pretrain import _walk_to_objective_targets
from train.phases import resolve_phase
from xushi2.multi_enemy_obs import GRID_CHANNELS, GRID_SIZE, MULTI_ENEMY_ENTITY_GRID_OBS_DIM

pytestmark = pytest.mark.contract_fast


def test_phase4_smoke_config_builds_mappo_config() -> None:
    with open(config_path("phase4/smoke/phase4_mappo_smoke.yaml"), encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    cfg = make_mappo_config(config)
    assert cfg.num_envs == 2
    assert cfg.n_agents == 3
    assert cfg.obs_dim == 31
    assert cfg.critic_obs_dim == 135
    assert cfg.vector_env == "sync"


def test_phase4_config_can_select_async_vector_backend() -> None:
    with open(config_path("phase4/smoke/phase4_mappo_smoke.yaml"), encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    config["ppo"] = dict(config["ppo"])
    config["ppo"]["vector_env"] = "async"
    cfg = make_mappo_config(config)
    assert cfg.vector_env == "async"


@pytest.mark.parametrize(
    ("key", "value", "msg"),
    [
        ("gamma", 0.0, "ppo.gamma"),
        ("gamma", 1.1, "ppo.gamma"),
        ("gae_lambda", -0.1, "ppo.gae_lambda"),
        ("gae_lambda", 1.1, "ppo.gae_lambda"),
        ("clip_ratio", 0.0, "ppo.clip_ratio"),
        ("value_clip_ratio", 0.0, "ppo.value_clip_ratio"),
        ("entropy_coef", -1.0e-4, "ppo.entropy_coef"),
        ("value_coef", -1.0, "ppo.value_coef"),
        ("aim_aux_coef", -0.1, "ppo.aim_aux_coef"),
        ("mode_aux_coef", -0.1, "ppo.mode_aux_coef"),
        ("target_selection_aux_coef", -0.1, "ppo.target_selection_aux_coef"),
        (
            "team_spirit_ramp_fraction",
            1.2,
            "ppo.team_spirit_ramp_fraction",
        ),
        (
            "team_spirit_ramp_fraction",
            -0.01,
            "ppo.team_spirit_ramp_fraction",
        ),
    ],
)
def test_mappo_config_rejects_out_of_range_hyperparameters(
    key: str, value: float, msg: str
) -> None:
    with open(config_path("phase4/smoke/phase4_mappo_smoke.yaml"), encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    config["ppo"] = dict(config["ppo"])
    config["ppo"][key] = value
    with pytest.raises(ValueError, match=msg):
        make_mappo_config(config)


def test_phase4_basic_config_builds_mappo_config() -> None:
    with open(config_path("phase4/baseline/phase4_mappo_basic.yaml"), encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    cfg = make_mappo_config(config)
    assert cfg.num_envs == 8
    assert cfg.n_agents == 3
    assert cfg.rollout_len == 128
    assert cfg.critic_obs_dim == 135


def test_phase4_noop_probe_config_builds_mappo_config() -> None:
    with open(
        config_path("phase4/probe/phase4_mappo_noop_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    cfg = make_mappo_config(config)
    assert cfg.num_envs == 8
    assert cfg.n_agents == 3
    assert cfg.lr_schedule == "constant"
    assert cfg.entropy_coef == 0.001
    assert config["env"]["opponent_bot"] == "noop"
    assert config["env"]["reward"]["distance_shaping_coef"] == 0.05
    assert config["env"]["reward"]["on_point_shaping_coef"] == 0.02
    assert config["run"]["bc_pretrain_steps"] == 500


def test_phase4_objective_probe_config_is_compact() -> None:
    with open(
        config_path("phase4/probe/phase4_mappo_objective_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    cfg = make_mappo_config(config)
    assert cfg.num_envs == 2
    assert cfg.rollout_len == 32
    assert cfg.gru_hidden == 32
    assert cfg.learning_rate == 1.0e-5
    assert config["run"]["bc_pretrain_steps"] == 200
    assert config["run"]["total_updates"] == 1


def test_phase11_mixed_league_probe_config_is_compact() -> None:
    with open(
        config_path("phase11/probe/phase11_mixed_league_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)
    cfg = make_mappo_config(config)
    assert phase == 11
    assert cfg.n_agents == 6
    assert cfg.value_per_agent is True
    assert ckpt_env_cfg["self_play_schedule"]["weights"]["current"] == 0.34
    assert ckpt_env_cfg["self_play_schedule"]["weights"]["snapshot"] == 0.33
    assert ckpt_env_cfg["self_play_schedule"]["weights"]["anchor"] == 0.33
    assert ckpt_env_cfg["snapshot_league"]["weights"]["latest"] == 0.7
    assert config["run"]["matrix_eval"]["current_selfplay"] is True
    assert config["run"]["matrix_eval"]["anchor_bots"] == ["noop"]
    assert len(config["run"]["matrix_eval"]["opponent_checkpoints"]) == 1
    assert config["run"]["total_updates"] == 1


def test_phase4_walk_bc_target_points_toward_objective() -> None:
    with open(config_path("phase4/smoke/phase4_mappo_smoke.yaml"), encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    cfg = make_mappo_config(config)
    obs = torch.tensor(
        [
            [0.0] * 31,
            [0.0] * 31,
        ],
        dtype=torch.float32,
    )
    obs[0, 5:7] = torch.tensor([0.0, -0.8])
    obs[1, 5:7] = torch.tensor([0.6, 0.0])
    target = _walk_to_objective_targets(obs, cfg)
    assert target[0, 0].item() == 0.0
    assert target[0, 1].item() == 1.0
    assert target[1, 0].item() == -1.0
    assert target[1, 1].item() == 0.0
