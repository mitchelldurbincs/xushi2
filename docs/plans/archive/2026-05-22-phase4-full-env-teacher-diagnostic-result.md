# Phase 4 full-env teacher diagnostic result

Date: 2026-05-22

## Summary

Added a direct teacher-action diagnostic for full-env Phase 4. The diagnostic
runs a teacher policy directly through `Phase4MappoEnv` without neural
training, so the project can check whether a proposed teacher can hit, hold
point, and avoid losing before using it as a supervised target.

Status: `EVIDENCE_INSUFFICIENT` for Phase 4 progress, but sufficient to reject
the actor-observation-only v2 teacher as a useful next training target.

## Evidence

- Diagnostic script: `python/scripts/diagnose_full_env_teacher.py`
- Tests: `python/tests/test_full_env_teacher_diagnostic.py`
- Source config:
  `experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v2.yaml`
- Seed: `3519994490`
- Actor-observation teacher output:
  `python/runs/phase4_mappo_full_env_rehearsal_v2/scripted_teacher_diagnostic.json`
- Full-state scripted baseline output:
  `python/runs/phase4_mappo_full_env_rehearsal_v2/cpp_basic_teacher_diagnostic.json`
- No W&B run was created; this is an offline diagnostic, not a training run.

## Results

Actor-observation-only teacher, `50` episodes vs `weak_basic_v2`:

```json
{
  "teacher": "actor_obs_scripted",
  "wins": 0.0,
  "losses": 0.0,
  "draws": 50.0,
  "mean_score_a": 0.0,
  "mean_score_b": 0.0,
  "team_a_hit_fire": 0.0,
  "team_a_visible_fire_rate": 1.0,
  "objective_on_point": 0.0
}
```

The same actor-observation teacher against `noop` won `10/10` with
`mean_score_a=37.0` and `objective_on_point=0.9333`, so it can walk to and hold
the point when uncontested. Its failure is contested shooting/majority pressure,
not basic movement.

Full-state `cpp_basic` teacher, `10` episodes vs `weak_basic_v2`:

```json
{
  "teacher": "cpp_basic",
  "wins": 10.0,
  "losses": 0.0,
  "draws": 0.0,
  "mean_score_a": 12.7,
  "mean_score_b": 0.0,
  "team_a_hit_fire": 0.09166666666666666,
  "team_a_visible_fire_rate": 1.0,
  "objective_on_point": 0.8666666666666745
}
```

## Interpretation

The v2 neural failure is now explained at the teacher level: the actor-visible
scripted teacher itself cannot create damage or majority point pressure against
`weak_basic_v2`. Training another neural policy against that same teacher, or
only extending rehearsal length, is not defensible.

The contrast with `cpp_basic` shows that the full Phase 4 wrapper and mechanics
can support a teacher that wins, hits, and holds point. The gap is teacher
fidelity and information surface. `cpp_basic` can choose the nearest visible
enemy across all enemy slots, while current Phase 4 flat actor observations
still expose only the counterpart enemy through `visible_enemy_1v1` in
`src/sim/src/obs_utils.cpp`. This is a real actor-observation limitation for
imitating nearest-enemy behavior.

## Verification

- `py -3.13 -m pytest tests/test_full_env_teacher_diagnostic.py
  tests/test_full_env_rehearsal.py -q` -> `9 passed`.
- `py -3.13 -m pytest tests/test_full_env_teacher_diagnostic.py
  tests/test_full_env_rehearsal.py tests/test_mappo_pretrain_hooks.py
  tests/test_mappo_focus_fire.py tests/test_mappo_aux_aim.py
  tests/test_phase7_partial_obs.py tests/test_phase4_mappo_env.py
  tests/test_mappo_matrix_eval.py -q` -> `74 passed`.
- `py -3.13 -m scripts.check_import_boundaries` -> PASS.

## Decision

Do not rerun actor-observation-only full-env rehearsal v1/v2, do not extend its
length, and do not force PPO after its failed pre-PPO gate.

Recommended next assignment: implement a bounded v3 pretrain design around a
higher-fidelity teacher, with an explicit preflight diagnostic. The v3 design
should either:

- train from a full-state scripted teacher such as `cpp_basic` while clearly
  marking it as privileged training-time imitation and proving no inference
  path reads hidden state; or
- switch to an existing wider actor-observation path that exposes multiple
  visible enemies, if the master/user approves changing the Phase 4 observation
  surface.

Do not change C++ sim rules, reward formulas, replay format, action semantics,
or phase-gate thresholds.

## Completion metadata

```json
{
  "changed_files": [
    "docs/plans/archive/2026-05-22-phase4-full-env-teacher-diagnostic-result.md",
    "python/scripts/diagnose_full_env_teacher.py",
    "python/tests/test_full_env_teacher_diagnostic.py"
  ],
  "verification": [
    "9 passed diagnostic/full-env rehearsal tests",
    "74 passed broader focused suite",
    "check_import_boundaries PASS",
    "actor_obs_scripted direct teacher diagnostic completed",
    "cpp_basic direct teacher diagnostic completed"
  ],
  "commit": "f776104eb95f64bea44975f0050af29f595f46af",
  "config_path": "experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v2.yaml",
  "seeds": [3519994490],
  "wandb_run_url": null,
  "replay_artifacts": [
    "data/replays/phase4_full_env_rehearsal_v2_ckpt_final_greedy.replay",
    "data/replays/phase4_full_env_rehearsal_v2_ckpt_final_stochastic.replay"
  ],
  "viewer_command": "xushi2-viewer --replay data/replays/phase4_full_env_rehearsal_v2_ckpt_final_greedy.replay",
  "tests_run": [
    "py -3.13 -m pytest tests/test_full_env_teacher_diagnostic.py tests/test_full_env_rehearsal.py -q",
    "py -3.13 -m pytest tests/test_full_env_teacher_diagnostic.py tests/test_full_env_rehearsal.py tests/test_mappo_pretrain_hooks.py tests/test_mappo_focus_fire.py tests/test_mappo_aux_aim.py tests/test_phase7_partial_obs.py tests/test_phase4_mappo_env.py tests/test_mappo_matrix_eval.py -q",
    "py -3.13 -m scripts.check_import_boundaries",
    "py -3.13 -m scripts.diagnose_full_env_teacher --config ..\\experiments\\configs\\phase4\\probe\\phase4_mappo_full_env_rehearsal_v2.yaml --episodes 50 --seed 3519994490 --teacher actor_obs_scripted --output runs\\phase4_mappo_full_env_rehearsal_v2\\scripted_teacher_diagnostic.json",
    "py -3.13 -m scripts.diagnose_full_env_teacher --config ..\\experiments\\configs\\phase4\\probe\\phase4_mappo_full_env_rehearsal_v2.yaml --episodes 10 --seed 3519994490 --teacher cpp_basic --output runs\\phase4_mappo_full_env_rehearsal_v2\\cpp_basic_teacher_diagnostic.json"
  ],
  "behavior_changes": [
    "Adds an offline diagnostic script for direct full-env teacher action streams"
  ],
  "reward_changes": [],
  "config_changes": [],
  "blocked_reason": null,
  "residual_risk": [
    "cpp_basic uses full state and is not actor-observation-only.",
    "No training run was launched from this diagnostic.",
    "The worktree was dirty, so this evidence is tied to commit plus explicit working-tree delta."
  ]
}
```
