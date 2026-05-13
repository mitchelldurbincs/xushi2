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

## How to pick a config

1. Start from the phase **baseline** for real training runs.
2. Use the phase **smoke** config for fast sanity checks.
3. Use the phase **probe** config for targeted behavioral checks.
4. Escalate to `legacy/` only when you must reproduce a historical run or run a known diagnostic-only variant.

## Canonical configs by phase

### Phase 3
- **Baseline:** `phase3/baseline/phase3_ranger_recurrent.yaml`
- **Smoke:** `phase3/smoke/phase3_ranger_smoke.yaml`
- **Probe:** `phase3/probe/phase3_ranger_noop_probe.yaml`

### Phase 4
- **Baseline:** `phase4/baseline/phase4_mappo_basic.yaml`
- **Smoke:** `phase4/smoke/phase4_mappo_smoke.yaml`
- **Probe:** `phase4/probe/phase4_mappo_objective_probe.yaml`

### Phase 11
- **Probe:** `phase11/probe/phase11_current_selfplay_probe.yaml`

## Legacy snapshots

Historical snapshots and superseded iteration configs (for example warm-start branches, versioned ladder configs, and one-off variants) should live under:

- `experiments/configs/<phase>/legacy/`

For phase-specific move history and old-name path mapping, see each phase's legacy README (for example `phase4/legacy/README.md`).


## Config metadata schema

Each actively maintained config should include a top-level `metadata` mapping:

- `phase`: phase identifier (`phase3`, `phase4`, etc.)
- `purpose`: run intent (`baseline`, `smoke`, `probe`, `legacy`)
- `status`: lifecycle state (`active`, `deprecated`, `diagnostic`)
- `expected_runtime`: coarse runtime class (`short`, `medium`, `long`, `extended`)
- `gate_relevance`: list of gate tags this config informs
- `lineage`: short free-text provenance summary

Use `python/scripts/list_configs.py` to discover and filter configs from metadata and print canonical train commands.
