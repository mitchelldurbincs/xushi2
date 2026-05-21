from pathlib import Path

import pytest
import yaml
from tests._paths import config_path

from train.mappo import train_phase4_from_config

pytestmark = [pytest.mark.slow, pytest.mark.smoke_behavior, pytest.mark.bc_probe]


@pytest.mark.parametrize(
    ("config_name", "output_name"),
    [
        ("phase4/probe/phase4_mappo_objective_probe.yaml", "phase4_objective"),
        ("phase5_entity_attention_probe.yaml", "phase5_entity"),
        ("phase6_entity_grid_probe.yaml", "phase6_grid"),
        ("phase7_team_fog_probe.yaml", "phase7_fog"),
        ("phase7_per_agent_fog_probe.yaml", "phase7_per_agent_fog"),
        ("phase8_random_map_probe.yaml", "phase8_random_map"),
    ],
)
def test_mappo_bc_probe_can_be_best_result(
    tmp_path: Path, config_name: str, output_name: str
) -> None:
    with open(config_path(config_name), encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    config["run"] = dict(config["run"])
    config["run"]["output_dir"] = str(tmp_path / output_name)
    result = train_phase4_from_config(config)
    assert result["mappo"] > 10.0
    assert (tmp_path / output_name / "mappo" / "ckpt_final.pt").exists()
