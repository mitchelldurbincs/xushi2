# scripts/

One-off utilities — data munging, replay conversion, sanity-check dumps,
etc. Nothing long-lived lives here; promote anything stable to `train/`,
`eval/`, or (if it's a C++ offline tool) `src/tools/`.

For Phase 4 training/eval configuration context, see
[`experiments/configs/README.md`](../../experiments/configs/README.md).

## Phase 4 quick script chooser

| Script path | Purpose | Typical Phase 4 usage | Key arguments | Expected output files |
| --- | --- | --- | --- | --- |
| `python/scripts/diag_phase4_walk_objective.py` | Environment sanity-check using a hardcoded walk-to-objective policy versus noop Team B (no learning required). | Verify the env/reward path is healthy before trusting MAPPO regressions. | `--seed`, `--round-length`, `--max-decisions`, `--dump-replay` | Optional text replay file from `--dump-replay` (viewer-consumable). |
| `python/scripts/eval_mappo_matrix.py` | Evaluate one or more MAPPO checkpoints against anchor bots and/or snapshot checkpoints; emits matchup rows with win/draw/loss and score stats. | Generate a compact league/matrix artifact for Phase 4+ checkpoint quality and comparison. | `--checkpoint` (repeat), `--anchor-bot` (repeat), `--opponent-checkpoint` (repeat), `--episodes`, `--seed`, `--output` | Optional matrix JSON file from `--output` (list of matchup rows). |
| `python/scripts/check_mappo_matrix.py` | Gate a matrix JSON artifact against threshold rules (minimum rows, min win-rate by opponent type, max draw-rate). | CI/automation pass-fail check after matrix eval. | `--matrix`, `--min-rows`, `--min-win-rate opponent_type=value` (repeat), `--max-draw-rate opponent_type=value` (repeat), `--output` | Optional gate summary JSON from `--output`; non-zero exit on gate failure. |
| `python/scripts/dump_replay.py` | Run checkpoint eval rollout(s) and dump deterministic text replay for viewer playback. Supports Phase 3 and MAPPO phases. | Produce artifact to inspect model behavior visually in the replay viewer after eval/gating. | `--checkpoint`, `--output`, `--seed`, `--episodes`, `--max-decisions`, `--stochastic` | Text replay file at `--output` (ASCII line-delimited replay format). |

## Other utility scripts

| Script path | Purpose |
| --- | --- |
| `python/scripts/list_configs.py` | Discover experiment configs by metadata (`phase`, `purpose`, `status`) and print canonical train commands. See `experiments/configs/README.md` §"Config metadata schema". |
| `python/scripts/check_import_boundaries.py` | Enforce Python package import-direction rules (`xushi2` → `envs` → `train`). See `docs/architecture/python_layers.md`. |
| `python/scripts/check_viewer_bench.py` | Compare a viewer-benchmark result JSON against the baseline within the 15% tolerance band. Used by `make bench-viewer`. |
| `python/scripts/diag_phase3_plumbing.py` | Phase-3 plumbing probe: drives `XushiEnv` with hand-written actions vs a noop opponent and prints objective-state-machine diagnostics. |

## Golden workflow (Phase 4)

Train checkpoint → matrix eval/gate → replay dump for viewer:

```bash
# 1) Evaluate checkpoint against anchors and optional frozen opponents.
python -m scripts.eval_mappo_matrix \
  --checkpoint runs/phase4_mappo/ckpt_0600.pt \
  --anchor-bot noop \
  --anchor-bot striker \
  --opponent-checkpoint runs/phase4_mappo/league/snapshot_0400.pt \
  --episodes 8 \
  --seed 0xE0A17 \
  --output artifacts/phase4/matrix_ckpt_0600.json

# 2) Gate the matrix artifact.
python -m scripts.check_mappo_matrix \
  --matrix artifacts/phase4/matrix_ckpt_0600.json \
  --min-rows 3 \
  --min-win-rate bot=0.55 \
  --min-win-rate snapshot=0.50 \
  --max-draw-rate bot=0.20 \
  --output artifacts/phase4/matrix_ckpt_0600_gate.json

# 3) Dump a replay for viewer inspection.
python -m scripts.dump_replay \
  --checkpoint runs/phase4_mappo/ckpt_0600.pt \
  --output ../data/replays/phase4_ckpt_0600_eval.replay \
  --episodes 1 \
  --seed 0xD1CEDA7A
```

Tip: if Step 2 fails, inspect printed failures, tune training/config, and rerun.
