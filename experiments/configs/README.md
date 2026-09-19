# Experiment Config Layout

## Naming convention

Config filenames keep the existing format:

- `<phase>_<track>_<purpose>.yaml`
- Examples: `phase4_mappo_smoke.yaml`, `phase11_current_selfplay_probe.yaml`

Directory layout is now runtime-first, then phase + intent for experiment
metadata:

- `experiments/configs/runtime/`
- `experiments/configs/<phase>/smoke/`
- `experiments/configs/<phase>/probe/`
- `experiments/configs/<phase>/baseline/`
- `experiments/configs/<phase>/legacy/`

## How to pick a config

1. Use `runtime/` configs when you are testing current runtime-spec dispatch.
2. Start from the phase **baseline** for real training runs that still belong to a phase gate.
3. Use the phase **smoke** config for fast sanity checks.
4. Use the phase **probe** config for targeted behavioral checks.
5. Escalate to `legacy/` only when you must reproduce a historical run or run a known diagnostic-only variant.

Phases are progress metadata. New runtime behavior should be selected by explicit
`learner.kind` and `env.kind` specs, not by adding new phase-number branches.

## Canonical configs by phase

### Explicit runtime specs
- **MAPPO flat smoke:** `runtime/mappo_flat_smoke.yaml`

### Phase 4
- **Baseline:** `phase4/baseline/phase4_mappo_basic.yaml`
- **Smoke:** `phase4/smoke/phase4_mappo_smoke.yaml`
- **Probe:** `phase4/probe/phase4_mappo_objective_probe.yaml`

The many Phase 4 probe files are historical experiment records unless a current
plan or card names them. Keep them for reproducibility, but do not treat the
whole probe directory as the active surface area.

### Phase 11
- **Probe:** `phase11/probe/phase11_current_selfplay_probe.yaml`

## Legacy snapshots

Historical snapshots and superseded iteration configs (for example warm-start branches, versioned ladder configs, and one-off variants) should live under:

- `experiments/configs/<phase>/legacy/`

For phase-specific move history and old-name path mapping, see each phase's legacy README (for example `phase4/legacy/README.md`).

Configs outside `legacy/` can still be historical if their `metadata.status` is
`diagnostic` or a plan/journal entry treats them as completed evidence. Prefer
clarifying metadata before moving paths that are referenced by docs, journals,
or old W&B run records.


## Config metadata schema

Each actively maintained config should include a top-level `metadata` mapping:

- `phase`: phase identifier (`phase4`, `phase11`, etc.)
- `purpose`: run intent (`baseline`, `smoke`, `probe`, `legacy`)
- `status`: lifecycle state (`active`, `deprecated`, `diagnostic`)
- `expected_runtime`: coarse runtime class (`short`, `medium`, `long`, `extended`)
- `gate_relevance`: list of gate tags this config informs
- `lineage`: short free-text provenance summary

Use `python/scripts/list_configs.py` to discover and filter configs from metadata and print canonical module-entrypoint commands.
