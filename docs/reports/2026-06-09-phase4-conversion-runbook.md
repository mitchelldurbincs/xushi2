# Phase 4 Conversion Run — MacBook Runbook

Date: 2026-06-09
Companion analysis: `docs/reports/2026-06-09-phase4-breakthrough-analysis.md`

## What's already done (on the Windows machine, ready to commit)

- **New reward terms** in `python/xushi2/reward.py` + `python/xushi2/reward_components.py`
  (config-gated, off by default):
  - `cap_progress_potential_coef` — potential-based shaping on the objective state
    machine, `Phi_A = owner_sign + cap_sign * cap_progress`. Pays every tick of
    uncontested capture progress; charges decay/ownership loss symmetrically.
    Policy-invariant (PBRS), safe to leave on permanently.
  - `capture_completed_bonus` — one-time team bonus when objective ownership flips.
  - Metrics (`captures_a/b`, `conversion_phi_a`, per-step contributions) flow into
    `info["reward_metrics"]` alongside the majority/uncontested metrics.
- **Tests**: 7 new unit tests in `python/tests/test_reward.py`, 1 real-sim
  integration test in `python/tests/test_phase4_mappo_env.py` (drives agents onto
  the point vs noop and asserts a capture completes and pays). All pass.
- **Configs**:
  - `experiments/configs/phase4/probe/phase4_mappo_conversion_v1.yaml` — the real run.
  - `experiments/configs/phase4/smoke/phase4_mappo_conversion_smoke.yaml` — 4-update
    wiring check (~7 min on the Windows box).
- **Warm-start checkpoint** copied to a git-tracked path:
  `data/checkpoints/phase4_multi_enemy_closed_loop_bridge_v1.pt` (836 KB — make sure
  this file is included in the commit, since `runs/` and `python/runs/` are ignored).

## Smoke result already observed (2026-06-09, Windows)

The smoke run validated the whole pipeline AND the core hypothesis:

- **Update 2 eval (eased timing 10s unlock / 5s capture): 4/4 WINS vs
  weak_basic_v2, score 3.10/2.93, kills 10.0/0.0, mean_reward +10.05.**
  First Team A wins against an objective-contesting bot in the project's history.
- Update 4 eval (smoke anneals to canonical 15s/8s in just 4 updates): 0/4 wins —
  expected; the real config takes 150 updates to anneal so PPO can track the
  tightening timing.

## MacBook steps

```bash
# 0) Pull, then build + install (from repo root)
make build-cpp
make py-install

# 1) Smoke first (~5-10 min, no W&B). MUST be launched from repo root —
#    config paths (data/checkpoints/..., runs/...) resolve from CWD.
xushi2-train --config experiments/configs/phase4/smoke/phase4_mappo_conversion_smoke.yaml
#    Pass criteria: "warm-start: loaded data/checkpoints/..." in the log,
#    finite rewards, eval lines print, checkpoints written under
#    runs/phase4_mappo_conversion_smoke/.

# 2) Optional: wandb login   (config has required: false, so it won't block)

# 3) The real run (~2-4 h depending on the machine)
xushi2-train --config experiments/configs/phase4/probe/phase4_mappo_conversion_v1.yaml
```

If `xushi2-train` isn't on PATH, the equivalent is
`python -m train.train --config <path>` using the venv that `make py-install`
populated (still from repo root).

## What to watch (leading indicators, not score)

Eval lines print `score`, `uncont` (uncontested seconds), `cap_gain`
(capture-progress ticks), `maj_sec`, `hit_fire`, `kills`. Baselines from the
bridge checkpoint pre-PPO: uncont 4.9s, cap_gain ~238, kills 13.4/0, hit_fire 0.047.

| Checkpoint | Healthy | Act |
|---|---|---|
| update 25–50 (eased timing) | wins/score > 0 (smoke says yes), uncont trending up | — |
| update 75–150 (timing annealing) | score persists as capture window tightens; cap_gain stays high while cap loss shrinks | if score collapses exactly as timing tightens, slow the anneal (`anneal_updates: 300`) and rerun from the last good ckpt |
| update 150–300 (canonical 15s/8s) | **mean_score_a > 0 at canonical timing = the actual breakthrough** | — |
| any | onpt/maj_sec collapsing toward 0 (flee), or hit_fire < 0.015 twice in a row | stop; drop LR to 3e-6; rerun |

The `canonical_eval` lines (every 25 updates) are the truth — they evaluate at
real 15s/8s timing regardless of where the curriculum currently is.

## After the run

- Matrix eval vs `noop` / `weak_basic_v2` / `basic` runs automatically
  (`matrix_eval.json` in the output dir).
- Dump and watch a replay (greedy and stochastic):
  `python python/scripts/dump_replay.py --checkpoint runs/phase4_mappo_conversion_v1/mappo/ckpt_best_eval.pt --output data/replays/phase4_conversion_v1_best_greedy.replay --episodes 5`
  (add `--stochastic` for the sampled variant). Reward-hacking check: agents
  should kill-then-HOLD, not cycle capture progress without scoring.
- Phase gate: thresholds are in the config's `phase_gate:` block
  (weak_basic_v2 score >= 3, wins >= 5/50, hit/fire >= 0.04, matrix weak_basic_v2
  score >= 3).
- Losing to `basic` is expected and is the next curriculum rung, not a
  falsification.
- Journal the result either way (`docs/journal/reinforcement_learning_journal.md`).

## Knobs, in the order to try them if it stalls

1. `objective_timing_curriculum.anneal_updates: 150 -> 300` (slower tightening).
2. `learning_rate: 1e-5 -> 3e-6` (if eval degrades early) or `-> 3e-5`
   (if everything is healthy but slow).
3. `capture_completed_bonus: 2.0 -> 4.0` and/or `uncontested_on_point_coef:
   0.15 -> 0.3` (if kills stay high but holding doesn't emerge).
4. `kill_bonus: 0.5 -> 0.25` (if agents still chase kills off the point).
5. If none of that produces captures even at eased timing: the fallback is the
   objective-conversion DAgger bridge from the 2026-05-22 audit
   (`docs/plans/active/2026-05-22-phase4-multi-enemy-closed-loop-zero-score-audit.md`),
   then PPO from that checkpoint with this same config.
