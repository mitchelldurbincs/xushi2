# Phase 4 Cap-Training Escalation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Get the Phase 4 MAPPO policy to actually train against the `basic` opponent. Climb a 3-rung escalation, advancing only when the prior rung empirically fails.

**Architecture:** Each rung is a config edit that adds *one* mechanism: (1) shaping fix, (2) BC pretrain, (3) two-stage warm-start from a noop-trained checkpoint. After each rung's run, evaluate at update 100 against an explicit "frozen vs. learning" check; abort and escalate if frozen. Design rationale: `docs/plans/2026-05-08-phase4-cap-training-escalation-design.md`.

**Tech Stack:** YAML, Python, PyTorch. Empirical loop — most tasks here are running a long training command, watching its eval line at update 100, and deciding whether to escalate.

**Pre-flight:** Branch `goalTest` already has uncommitted changes from the previous session (per the user's "commit-the-whole-delta-at-end" workflow — that's intentional, not blocking). Run from `C:\Users\mitchell.durbin\source\repos\cartridgeRepos\xushi2\python` unless noted. Use `py -3.13` (Python 3.13 launcher) — not the default `python`.

**Stopping criterion** (used after every rung's eval-at-update-100):
- **PASS-through:** Any of `mean_reward > -8`, `onpt > 0` in trainer log, eval `kills > 0` for our team. Continue the run to update 250.
- **FAIL-escalate:** All three at frozen-agent values (`-11.0` / `0.000` / `0.0`). Kill the run, advance to next rung.

---

## Task 1: Rung 1 — create shaping-fix config

**Why:** Cheapest possible intervention. Same `phase4_mappo_basic.yaml` config but with the shaping coefs that `phase4_mappo_objective_probe.yaml` (a known-working config) uses: `distance_shaping_coef: 0.05, on_point_shaping_coef: 0.02`.

**Files:**
- Create: `experiments/configs/phase4/legacy/phase4_mappo_basic_v2.yaml`

**Step 1: Read the baseline config**

Read `experiments/configs/phase4/baseline/phase4_mappo_basic.yaml`. Note these will be the differences:
- `env.reward.distance_shaping_coef: 0.005` → `0.05`
- `env.reward.on_point_shaping_coef: <absent>` → `0.02`
- `run.output_dir: runs/phase4_mappo_basic` → `runs/phase4_mappo_basic_v2`

Everything else (model, ppo, team_spirit ramp, opponent=basic, sim) stays identical.

**Step 2: Write the new config**

Create `experiments/configs/phase4/legacy/phase4_mappo_basic_v2.yaml` with the exact contents of `phase4_mappo_basic.yaml`, modified per Step 1.

**Step 3: Verify the config parses**

Run from `python/`:

```
py -3.13 -c "import yaml; print(yaml.safe_load(open('../experiments/configs/phase4/legacy/phase4_mappo_basic_v2.yaml'))['env']['reward'])"
```

Expected output: `{'distance_shaping_coef': 0.05, 'on_point_shaping_coef': 0.02}`.

---

## Task 2: Rung 1 — run training

**Why:** Empirical test of "does shaping alone fix it?"

**Step 1: Start training in foreground**

Run from `python/`:

```
py -3.13 -m train.train --config ../experiments/configs/phase4/legacy/phase4_mappo_basic_v2.yaml
```

This is a long-running task (~30 minutes on the user's machine for 250 updates). Do not run in background — we want to see the eval lines stream live so we can hit the decision gate at update 100.

**Step 2: Wait for the eval line at update 100**

Watch for a log line of the form `[phase4/mappo] eval update=100/250 mean_reward=... wins=.../10 ... score=.../...  kills=.../...`. There are evals at updates 25, 50, 75, 100. The update-100 eval is the gate.

**Step 3: Apply the stopping criterion**

Compute (visually from the eval line):
- `mean_reward > -8` ?
- `onpt > 0` (this is in the per-update training metric line, look at the most recent one before update 100)?
- our team `kills > 0` in the eval line (first number in `kills=X.X/Y.Y`)?

If **any** are true → PASS-through. Continue to Step 4.
If **all** at frozen values (`-11.0` / `0.000` / `0.0`) → FAIL-escalate. Skip to Task 5.

**Step 4: Let it finish**

Wait for `[phase4] mappo_final=...` line (final eval at update 250). Note the final values:
- `mappo_final` (target: > 0 for "actually winning")
- Eval `wins` count
- Eval `score=A/B` ratio

If `mappo_final > 0` and `wins > 0`: rung 1 succeeded, skip to Task 8.
If `mappo_final < 0` but the run wasn't frozen at update 100 (i.e. some intermediate signal): note the partial progress, then advance to Task 3 (rung 2).

---

## Task 3: Rung 1 — view the result (only if Rung 1 ran to update 250)

**Why:** Visual confirmation of what the policy is doing, to inform whether rung 2 is the right next step or whether we need a different intervention than what's planned.

**Files:**
- Output: `replays/phase4_basic_v2_final.replay`

**Step 1: Dump replay from the final checkpoint**

Run from `python/`:

```
py -3.13 -m scripts.dump_replay --checkpoint runs/phase4_mappo_basic_v2/mappo/ckpt_final.pt --output ../replays/phase4_basic_v2_final.replay --episodes 1
```

Expected: `[dump_replay] wrote N decisions to ..\replays\phase4_basic_v2_final.replay` where N is between 100 and 300 (one episode of 30s × ~10 decisions/s).

**Step 2: View it**

Run from repo root:

```
build\bin\Release\xushi2_viewer.exe --replay replays\phase4_basic_v2_final.replay
```

If the policy looks like it's actually playing (walking to point, sometimes shooting), great. If it's still spinning, the shaping coefs may need tuning rather than escalation — flag this to the user before proceeding.

---

## Task 4: Rung 1 SUCCESS path — document and stop

**Why:** If rung 1 succeeded, this is the new Phase 4 baseline. Stop the escalation here.

**Files:**
- Modify: `docs/plans/2026-05-08-phase4-cap-training-escalation.md` (this file) — append a "Result" section noting rung 1 succeeded.

**Step 1: Append result section**

Add to the end of this plan:

```markdown
---

## Result

Rung 1 succeeded. New Phase 4 baseline: `experiments/configs/phase4/legacy/phase4_mappo_basic_v2.yaml`.
Final eval: wins=N/10, mean_reward=X, score=A/B. See `runs/phase4_mappo_basic_v2/mappo/ckpt_final.pt`.
```

**Step 2: Stop the plan execution.** Skip remaining tasks. Tell the user.

---

## Task 5: Rung 2 — create BC-pretrain config

**Why:** Rung 1 alone didn't break the impasse. The likely missing ingredient is a non-random initialization that's already cap-pointed. BC pretrain (200 steps imitating a "walk to objective" expert) gives PPO a starting policy that has at least *some* baseline cap-seeking behavior, so the shaping signal has something to grade.

**Files:**
- Create: `experiments/configs/phase4/legacy/phase4_mappo_basic_v3.yaml`

**Step 1: Write the new config**

Same as `phase4_mappo_basic_v2.yaml`, with these additions to the `run:` block:

```yaml
run:
  total_updates: 250
  eval_every: 25
  eval_episodes: 10
  checkpoint_every: 25
  log_every: 5
  bc_pretrain_steps: 200
  bc_batch_size: 192
  bc_learning_rate: 1.0e-3
  output_dir: runs/phase4_mappo_basic_v3
```

The `bc_*` keys are read by `python/train/mappo.py:1404-1412` (already wired). No code changes needed.

**Step 2: Verify the config parses**

```
py -3.13 -c "import yaml; cfg=yaml.safe_load(open('../experiments/configs/phase4/legacy/phase4_mappo_basic_v3.yaml')); print('bc_steps=', cfg['run'].get('bc_pretrain_steps')); print('shaping=', cfg['env']['reward'])"
```

Expected: `bc_steps= 200` and `shaping= {'distance_shaping_coef': 0.05, 'on_point_shaping_coef': 0.02}`.

---

## Task 6: Rung 2 — run training

Same procedure as Task 2.

**Step 1: Start training**

```
py -3.13 -m train.train --config ../experiments/configs/phase4/legacy/phase4_mappo_basic_v3.yaml
```

**Step 2: Watch for the BC pretrain log**

Before the first `update=1/250` line, you should see lines like `[phase4/mappo] bc_pretrain step=N/200 ...`. If those don't appear, BC isn't engaging — abort and check that `bc_pretrain_steps` is reaching mappo.py (look at `python/train/mappo.py:1404`).

**Step 3: Apply the stopping criterion at update 100**

Same as Task 2 Step 3. PASS-through → continue. FAIL-escalate → Task 8.

**Step 4: Let it finish (PASS-through path)**

Same as Task 2 Step 4. If `mappo_final > 0` and `wins > 0`: rung 2 succeeded, skip to Task 7. Otherwise advance to Task 8 (rung 3).

---

## Task 7: Rung 2 view + document SUCCESS

Mirror of Tasks 3–4 but for rung 2.

**Step 1: Dump and view**

```
py -3.13 -m scripts.dump_replay --checkpoint runs/phase4_mappo_basic_v3/mappo/ckpt_final.pt --output ../replays/phase4_basic_v3_final.replay --episodes 1
```

```
build\bin\Release\xushi2_viewer.exe --replay replays\phase4_basic_v3_final.replay
```

**Step 2: Append result to this plan**

```markdown
---

## Result

Rung 2 succeeded. New Phase 4 baseline: `experiments/configs/phase4/legacy/phase4_mappo_basic_v3.yaml`.
BC pretrain was the necessary ingredient. Final eval: wins=N/10, mean_reward=X, score=A/B.
```

Stop. Tell the user.

---

## Task 8: Rung 3 prep — add warm-start support to MAPPO trainer

**Why:** Rung 1 + BC pretrain still didn't crack it. Rung 3 needs to first train a policy against `noop` (no enemy fire — easier learning environment), then warm-start that checkpoint into a run against `basic`. `python/train/mappo.py` does **not** currently support warm-starting (only `python/train/ppo_recurrent/orchestration.py:278-284` does, for a different trainer). This task adds the same pattern to the MAPPO trainer.

**Files:**
- Modify: `python/train/mappo.py` (around the `MappoTrainer.__init__` or trainer setup; placement TBD by reading the file)
- Test: `python/tests/test_mappo_warm_start.py` (new)

**Step 1: Read the existing warm-start pattern**

Open `python/train/ppo_recurrent/orchestration.py` and read lines 167-180 (`_load_init_checkpoint` definition) and lines 278-284 (call site). Note:
- It expects `run_cfg.get("init_from_checkpoint")` (string path).
- Validates the architecture matches (see `_load_init_checkpoint` body).
- Loads `state_dict` into the trainer's model.

**Step 2: Write the failing test**

Create `python/tests/test_mappo_warm_start.py`:

```python
"""Test that MAPPO trainer warm-starts from a checkpoint when
``run.init_from_checkpoint`` is set in the config."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from train.phases import train_phase4_from_config


def _phase4_smoke_cfg(tmp_path: Path, output_dir: Path, **run_overrides):
    return {
        "phase": 4,
        "env": {
            "seed_base": 0,
            "opponent_bot": "noop",
            "learner_team": "A",
            "sim": {
                "round_length_seconds": 3,
                "fog_of_war_enabled": False,
                "randomize_map": False,
                "seed": 0,
                "action_repeat": 3,
                "mechanics": {
                    "revolver_damage_centi_hp": 7500,
                    "revolver_fire_cooldown_ticks": 15,
                    "revolver_hitbox_radius": 0.75,
                    "respawn_ticks": 240,
                },
            },
        },
        "model": {
            "use_recurrence": True,
            "embed_dim": 16,
            "gru_hidden": 8,
            "head_hidden": 16,
            "action_log_std_init": -1.0,
        },
        "ppo": {
            "num_envs": 2, "rollout_len": 16, "num_epochs": 1,
            "minibatch_size": 1, "learning_rate": 3.0e-4,
            "value_normalization": True, "vector_env": "sync",
            "torch_num_threads": 1, "lr_schedule": "constant",
            "lr_final_ratio": 1.0, "warmup_updates": 0,
            "clip_ratio": 0.2, "value_clip_ratio": 0.2,
            "gamma": 0.997, "gae_lambda": 0.95,
            "entropy_coef": 0.01, "value_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "run": {
            "total_updates": 1, "eval_every": 1,
            "eval_episodes": 1, "checkpoint_every": 1,
            "log_every": 1,
            "output_dir": str(output_dir),
            **run_overrides,
        },
    }


def test_mappo_warm_starts_from_init_checkpoint(tmp_path: Path) -> None:
    # Stage 1: train one update, save checkpoint.
    stage1 = tmp_path / "stage1"
    train_phase4_from_config(_phase4_smoke_cfg(tmp_path, stage1))
    ckpt_path = stage1 / "mappo" / "ckpt_final.pt"
    assert ckpt_path.exists(), "stage 1 must produce a checkpoint"

    # Capture the stage-1 model's first-layer weights to compare later.
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw_ckpt["model"]
    expected_first_param_name = next(iter(state.keys()))
    expected_value = state[expected_first_param_name].clone()

    # Stage 2: train another run, warm-starting from stage 1.
    stage2 = tmp_path / "stage2"
    cfg = _phase4_smoke_cfg(
        tmp_path, stage2,
        init_from_checkpoint=str(ckpt_path),
    )
    # Set total_updates to 0 so we don't *update* — just load and snapshot.
    # If 0 isn't supported, use 1 and accept that the post-update weights
    # differ slightly from the init.
    cfg["run"]["total_updates"] = 1
    train_phase4_from_config(cfg)

    stage2_ckpt = torch.load(
        stage2 / "mappo" / "ckpt_final.pt",
        map_location="cpu",
        weights_only=False,
    )
    # Stage 2 init must have come from stage 1: at the very least the
    # post-1-update weights must be CLOSER to stage-1 weights than a fresh
    # random init would be. Use a small tolerance.
    actual = stage2_ckpt["model"][expected_first_param_name]
    diff = (actual - expected_value).abs().mean().item()
    assert diff < 0.5, (
        f"warm-started weights diverged too far from init "
        f"(diff={diff}); expected near-zero update"
    )
```

**Step 3: Run the failing test**

```
py -3.13 -m pytest tests/test_mappo_warm_start.py -v
```

Expected: FAIL — test_mappo_warm_starts_from_init_checkpoint either (a) errors because `init_from_checkpoint` is unrecognized, or (b) the diff is much larger than 0.5 because warm-start was silently ignored.

**Step 4: Add warm-start support to mappo.py**

In `python/train/mappo.py`, find the trainer setup site (around `trainer = MappoTrainer(env_fn, cfg, seed=seed_base)`, near line 1359). Right after trainer creation, before the training loop, add:

```python
init_ckpt = run_cfg.get("init_from_checkpoint")
if init_ckpt:
    raw = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    trainer.model.load_state_dict(raw["model"], strict=True)
    print(f"[{phase_label}/mappo] warm-start: loaded {init_ckpt}", flush=True)
```

Verify the checkpoint format matches the existing save path (look for `torch.save(...)` calls in mappo.py to confirm the dict shape).

**Step 5: Run the test, verify it passes**

```
py -3.13 -m pytest tests/test_mappo_warm_start.py -v
```

Expected: PASS.

---

## Task 9: Rung 3a — train against noop with shaping

**Why:** Build a checkpoint that's at least competent at the cap-holding objective without combat, before introducing the basic bot.

**Files:**
- Create: `experiments/configs/phase4_mappo_basic_noop_pretrain.yaml`

**Step 1: Write the config**

Same as `phase4_mappo_basic_v2.yaml` (rung 1's config), but:
- `env.opponent_bot: basic` → `noop`
- `run.output_dir: runs/phase4_mappo_basic_v2` → `runs/phase4_mappo_basic_noop_pretrain`

Keep the team_spirit ramp, BC pretrain *off* (we're testing pure RL against noop), shaping on.

**Step 2: Run training**

```
py -3.13 -m train.train --config ../experiments/configs/phase4_mappo_basic_noop_pretrain.yaml
```

**Step 3: Verify it actually learned to hold the cap**

Watch for evals where `wins > 0` and `score > 0`. If by update 100 we're still at `wins=0`, abort — even noop isn't trainable from this init, which would mean the issue is deeper than curriculum. Tell the user before continuing.

**Step 4: Confirm `runs/phase4_mappo_basic_noop_pretrain/mappo/ckpt_final.pt` exists.**

---

## Task 10: Rung 3b — warm-start from noop checkpoint into basic-opponent run

**Files:**
- Create: `experiments/configs/phase4/legacy/phase4_mappo_basic_v4.yaml`

**Step 1: Write the config**

Same as `phase4_mappo_basic_v2.yaml`, plus add to `run:`:

```yaml
  init_from_checkpoint: runs/phase4_mappo_basic_noop_pretrain/mappo/ckpt_final.pt
  output_dir: runs/phase4_mappo_basic_v4
```

Keep `opponent_bot: basic`. The warm-start gives us a non-random policy that already knows the cap is good; the run learns to defend it under fire.

**Step 2: Run training**

```
py -3.13 -m train.train --config ../experiments/configs/phase4/legacy/phase4_mappo_basic_v4.yaml
```

Expect to see `warm-start: loaded ...` in the log within the first few lines.

**Step 3: Apply the stopping criterion at update 100**

Same as Task 2 Step 3.

**Step 4: Let it finish**

If `mappo_final > 0` and `wins > 0`: rung 3 succeeded, advance to Task 11.
If still failing: this is no longer a config issue — escalate to the user with all three rungs' eval data and discuss next steps.

---

## Task 11: Rung 3 view + document SUCCESS

Mirror of Tasks 3–4 / 7 for rung 3.

**Step 1: Dump and view**

```
py -3.13 -m scripts.dump_replay --checkpoint runs/phase4_mappo_basic_v4/mappo/ckpt_final.pt --output ../replays/phase4_basic_v4_final.replay --episodes 1
```

```
build\bin\Release\xushi2_viewer.exe --replay replays\phase4_basic_v4_final.replay
```

**Step 2: Append result section**

```markdown
---

## Result

Rung 3 succeeded. New Phase 4 baseline: two-stage warm-start from
`runs/phase4_mappo_basic_noop_pretrain/mappo/ckpt_final.pt` into
`experiments/configs/phase4/legacy/phase4_mappo_basic_v4.yaml`.
Final eval: wins=N/10, mean_reward=X, score=A/B.
```

Tell the user.

---

## Out of scope (explicit non-goals)

- Tuning `entropy_coef`, `learning_rate`, `value_coef`, `clip_ratio`, etc. — separate axis if all 3 rungs fail.
- Modifying the per-agent reward path or `team_spirit` ramp.
- Phase 5+ envs.
- Defining a numerical "Phase 4 acceptance gate" — separate plan.
- Per-component reward weight retuning beyond what's already in the configs.
