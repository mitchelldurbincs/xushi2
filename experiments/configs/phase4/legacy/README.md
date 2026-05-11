# Phase 4 legacy configs

Use `phase4/baseline/`, `phase4/smoke/`, and `phase4/probe/` first. Only use legacy configs for explicit reproduction/debug requests.

## Status legend
- **superseded**: historical variant, replaced by canonical baseline/smoke/probe.
- **diagnostic-only**: purpose-built for targeted debugging checks.
- **reusable template**: old but still useful as a starting point for similar experiments.

## Legacy file index
| Legacy filename | Status | Current location / note |
|---|---|---|
| `phase4_mappo_basic_v2.yaml` | superseded | `legacy/archive/phase4_mappo_basic_v2.yaml` |
| `phase4_mappo_basic_v3.yaml` | superseded | `legacy/archive/phase4_mappo_basic_v3.yaml` |
| `phase4_mappo_basic_v3_noop.yaml` | diagnostic-only | `legacy/archive/phase4_mappo_basic_v3_noop.yaml` |
| `phase4_mappo_basic_v4.yaml` | superseded | `legacy/archive/phase4_mappo_basic_v4.yaml` |
| `phase4_mappo_basic_v5.yaml` | superseded | `legacy/archive/phase4_mappo_basic_v5.yaml` |
| `phase4_mappo_basic_v6.yaml` | superseded | `legacy/archive/phase4_mappo_basic_v6.yaml` |
| `phase4_mappo_basic_v6_5.yaml` | superseded | `legacy/archive/phase4_mappo_basic_v6_5.yaml` |
| `phase4_mappo_basic_v7_holdshoot.yaml` | diagnostic-only | `legacy/archive/phase4_mappo_basic_v7_holdshoot.yaml` |
| `phase4_mappo_basic_v8.yaml` | reusable template | `legacy/phase4_mappo_basic_v8.yaml` |
| `phase4_mappo_noop.yaml` | diagnostic-only | `legacy/phase4_mappo_noop.yaml` |

## Migration map for old run logs
If an old run references a legacy filename, resolve it with the table above. Paths moved into `legacy/archive/` are unchanged except for that inserted folder segment.
