# Experiment Config Layout

## Naming convention

Config filenames keep the existing format:

- `<phase>_<track>_<purpose>.yaml`
- Examples: `phase4_mappo_smoke.yaml`, `phase11_current_selfplay_probe.yaml`

Directory layout is now phase + intent:

- `experiments/configs/<phase>/smoke/`
- `experiments/configs/<phase>/probe/`
- `experiments/configs/<phase>/baseline/`
- `experiments/configs/<phase>/legacy/`

## Canonical configs by phase

### Phase 3
- **Smoke:** `phase3/smoke/phase3_ranger_smoke.yaml`
- **Probe:** `phase3/probe/phase3_ranger_noop_probe.yaml`
- **Baseline (primary):** `phase3/baseline/phase3_ranger_recurrent.yaml`

### Phase 4
- **Smoke:** `phase4/smoke/phase4_mappo_smoke.yaml`
- **Probe:** `phase4/probe/phase4_mappo_objective_probe.yaml`
- **Baseline (primary):** `phase4/baseline/phase4_mappo_basic.yaml`

### Phase 11
- **Probe (current self-play):** `phase11/probe/phase11_current_selfplay_probe.yaml`
- **Probe (mixed league):** `phase11/probe/phase11_mixed_league_probe.yaml`

## Legacy snapshots

Historical snapshots and superseded iteration configs (for example warm-start branches, versioned ladder configs, and one-off variants) should live under:

- `experiments/configs/<phase>/legacy/`

Keep legacy filenames unchanged so past run logs and artifact references remain easy to map.
