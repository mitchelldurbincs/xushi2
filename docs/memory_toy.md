# Memory-toy environment (Phase 2)

> Referenced as TBD from `observation_spec.md` §"Phase 2" and
> `rl_design.md` §6 Phase 2 and §10 Memory sanity test.

## Purpose

Validate that our recurrent PPO training machinery is correct *before* we
depend on it for the real game. The toy environment is orthogonal to the
C++ sim — it exists solely to exercise GRU hidden-state handling, BPTT,
and rollout/training consistency in isolation.

Failure modes caught here, per `rl_design.md` §10:

- Stale hidden state across PPO epochs.
- BPTT truncation boundary `detach()` bugs.
- Episode reset not zeroing hidden state.
- Hidden state divergence between rollout sampling pass and training pass.

If any of these exist, Phase 2 will fail *loudly* on a trivial env where
we know the optimum analytically. If we skip Phase 2 and jump to Phase 3
(recurrent PPO on the real game), these bugs silently degrade training
to effectively-feedforward with no error — you would waste days training
a policy that looks fine but isn't using memory.

## The task (D2: aim at remembered position)

At reset, the env samples a target point on the unit circle and shows it
to the agent for a brief visible window. After the window, the cue
disappears (observation goes to zero). At the final tick of the episode,
the agent must output a 2D action matching the target. Reward is the
negative Euclidean distance between the action and the target, awarded
only at the terminal tick.

This rehearses the exact Phase 7 memory pattern ("I saw the enemy at
position X; they are now out of sight; remember where they were") on a
task simple enough that the optimal behavior is analytically known, and
where a recurrent policy is *provably* required to achieve nonzero
reward in the post-cue regime.

### Dynamics

- Tick rate: unitless. One step == one tick.
- `T = 64` (episode length in ticks).
- `k = 4` (cue visibility window).
- At reset: sample `theta ~ Uniform[0, 2π]`; set
  `target = (cos(theta), sin(theta))`.
- Per-tick observation: `[vis_x, vis_y, visible_flag]` (shape `(3,)`,
  dtype `float32`).
  - For `t < k`: `(vis_x, vis_y) = target`, `visible_flag = 1.0`.
  - For `t ≥ k`: `(vis_x, vis_y) = (0.0, 0.0)`, `visible_flag = 0.0`.
- Action space: `Box(low=-1, high=1, shape=(2,), dtype=float32)`.
- Reward:
  - For `t < T-1`: `0.0`.
  - For `t = T-1`: `-||action_{T-1} - target||_2`, clamped to `≥ -2.0`.
- `terminated = (t == T-1)`; `truncated = False`.

### Analytic reward bounds

| Policy | Best achievable terminal reward |
|---|---|
| Optimal recurrent | `0.0` (outputs target exactly) |
| Optimal feedforward | `≈ -1.0` (observes `(0, 0, 0)` at terminal; best action is `(0, 0)`; expected `E[-‖target - 0‖] = -1.0` since `‖target‖ = 1`) |
| Random (uniform in unit ball) | `≈ -1.27` |

The ~1.0 reward gap between recurrent and feedforward is the **memory
proof**. If that gap doesn't appear in training, the recurrent trainer
is broken.

## Repo layout

The toy env is a standalone Gymnasium env with no dependency on xushi2
or the C++ sim.

```
python/envs/memory_toy.py          # MemoryToyEnv(gym.Env)
python/envs/__init__.py
python/tests/test_memory_toy_env.py
```

The recurrent PPO trainer, on the other hand, is designed to be reused
at Phase 3 with an env swap:

```
python/train/models.py             # RecurrentActorCritic + FeedforwardActorCritic
python/train/ppo_recurrent.py      # CleanRL-style recurrent PPO
python/train/train.py              # existing entrypoint — add phase=2 branch
python/tests/test_ppo_recurrent_invariants.py
```

Eval (memory-proof ablation harness):

```
python/eval/eval_memory_toy.py
python/tests/test_eval_memory_toy.py
```

Config:

```
experiments/configs/phase2_memory_toy.yaml
```

## Config schema

```yaml
phase: 2
env:
  episode_length: 64
  cue_visible_ticks: 4
  seed_base: 0x4d454d54   # "MEMT"
model:
  use_recurrence: true      # false for the feedforward baseline
  embed_dim: 64
  gru_hidden: 64
  head_hidden: 64
  action_log_std_init: -1.0
ppo:
  num_envs: 16
  rollout_len: 256          # ticks per env per PPO update
  num_epochs: 4
  minibatch_size: 64        # in units of whole episodes
  learning_rate: 3.0e-4
  clip_ratio: 0.2
  value_clip_ratio: 0.2
  gamma: 0.99
  gae_lambda: 0.95
  entropy_coef: 0.01
  value_coef: 0.5
  max_grad_norm: 0.5
run:
  total_updates: 500
  eval_every: 25
  eval_episodes: 100
  checkpoint_every: 50
  log_every: 1
  output_dir: runs/phase2_memory_toy
```

## Policy architecture

Shared GRU trunk between actor and critic (per `rl_design.md` §2):

```
obs (3) ─► Linear(3→64) ─► ReLU ─┐
                                  ├─► GRUCell(64, hidden=64) ─► h_t
                    h_{t-1} ──────┘                             │
                                                                ├─► Linear(64→64)→ReLU → Linear(64→2) ─► action mean
                                                                │                          + learned log_std param
                                                                └─► Linear(64→64)→ReLU → Linear(64→1) ─► value
```

- Action head outputs a 2D diagonal Gaussian mean. Learned per-dim
  `log_std` parameter (not state-dependent).
- Actions are sampled, then squashed via `tanh` to `[-1, 1]²` with the
  standard `log_prob` correction term.
- `use_recurrence: false` replaces the GRUCell with an identity
  (`h_t = embed(obs_t)`), keeping total parameter count comparable.

## Hidden-state management rules

These are the things that must be correct for recurrent PPO to work at
all. Each gets an explicit unit test.

1. **Reset zeroing.** On `env.reset()`, the trainer's stored hidden state
   for that env index is zeroed. Asserted on every reset in the trainer.
2. **Rollout causality.** `h_{t+1} = GRU(obs_t, h_t)`. At each rollout
   tick, the buffer stores `h_t` (the input to the GRU cell at that tick).
3. **Segment boundary detach.** `h` is `detach()`ed exactly once per
   segment, at `h_0_of_segment` (i.e., at the start of each episode and
   at rollout-buffer boundaries that align with episode boundaries).
   Gradients flow within a segment via full BPTT; they do not flow
   across segments.
4. **Cross-epoch consistency.** Across PPO epochs, the forward pass at
   training time is re-run from the stored `h_0_of_segment`. A hidden
   state from epoch `n` is never reused as initial state at epoch `n+1`.
5. **Minibatch composition.** Minibatches are composed of whole episodes
   (not random tick strides), so the GRU state trajectory in each
   minibatch is contiguous within each episode.

## Memory-proof gate

A single exit-code-gated command decides whether Phase 2 passes.

```
python -m eval.eval_memory_toy --checkpoint runs/phase2_memory_toy/recurrent/ckpt_final.pt
```

The harness runs 500 eval episodes in each of three ablation modes and
asserts:

| Mode | `h` behavior | Required terminal reward |
|---|---|---|
| `normal` | carried as during rollout | `> -0.15` (within 15% of optimal) |
| `zero_every_tick` | `h ← 0` before every forward | `∈ [-1.2, -0.8]` (collapses to feedforward) |
| `random_every_tick` | `h ← N(0, 1)` before every forward | `∈ [-1.5, -0.8]` |

Plus a final cross-check: `normal_mean - zero_every_tick_mean > 0.5`
(the memory effect dominates noise).

In addition, the training run itself produces two curves — one for the
recurrent policy, one for the feedforward baseline (trained from the same
code, same seed base, same wall-clock run for fairness). The feedforward
curve must plateau at `≥ -1.1` (confirming the analytic bound) while the
recurrent curve converges near `0`.

## What Phase 2 does NOT include

Every item below is deliberately deferred. Adding any of them blends
multiple deltas into one and defeats the point of the phase ladder
(`rl_design.md` §6).

- No variable-length episodes or variable cue-visible window
  (randomization is Phase 8).
- No wandb logging — tensorboard only. Wandb can be added later as a
  thin adapter.
- No CUDA-specific code. `torch.device("cuda" if available else "cpu")`
  and nothing more.
- No resume-from-checkpoint mid-run. The full training run is ~10 min on
  CPU; resume is YAGNI.
- No linear-probe-on-hidden-state analysis. The ablation eval gives the
  same yes/no signal with less code.
- No multi-agent, centralized critic, entity attention, grid obs, map
  randomization, or snapshot pool. These are Phases 4-9.

## Success-criteria checklist

Phase 2 is complete when all of these pass:

1. `cmake --build build --config Release && ctest --test-dir build -C Release` → 87/87 (no C++ regression).
2. `cd python && python -m pytest tests/` → 59 existing + 13 new tests pass.
3. `python -m train.train --config experiments/configs/phase2_memory_toy.yaml` exits 0 and prints a final `[phase2] recurrent_final=... feedforward_final=... gap=...` line.
4. `python -m eval.eval_memory_toy --checkpoint runs/phase2_memory_toy/recurrent/ckpt_final.pt` exits 0 (i.e., all ablation assertions hold).
