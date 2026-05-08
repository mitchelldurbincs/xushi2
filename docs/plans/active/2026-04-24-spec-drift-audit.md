# Spec Drift Audit — Phase 3 → Phase 4 cleanup

**Date:** 2026-04-24
**Purpose:** Compare design docs against the post-Phase-2-refactor codebase; surface deltas for resolution. This memo is the input to task B7 (apply spec-side edits) and to any code-side follow-ups raised to the user.

**Scope of "code":**
- Reward: `python/xushi2/reward.py`
- PPO trainer: `python/train/ppo_recurrent/{trainer,losses,config,orchestration,lr_schedule}.py`, `python/train/rollout_buffer.py`
- Phase-3 run config: `experiments/configs/phase3_ranger_recurrent.yaml`
- Observations: `src/sim/src/actor_obs.cpp`, `src/sim/src/critic_obs.cpp`, `src/sim/src/obs_utils.cpp`, `python/xushi2/obs_manifest.py`, `src/sim/include/xushi2/sim/obs.h`
- Actions: `src/common/include/xushi2/common/types.h`, `src/common/include/xushi2/common/action_canon.hpp`, `python/xushi2/env.py`

Resolution legend:
- **edit spec** — code is right; doc needs to be updated in B7.
- **edit code** — doc is right; code drifted and should be fixed (surfaced to user, not applied here).
- **intentional drift** — a deliberate Phase-ladder decision; doc should note the ladder context but the current code is correct for the current phase.

---

## rl_design.md

### §5 — Reward

| Spec says | Code does | Resolution |
|---|---|---|
| Terminal: +10 win / −10 loss / 0 draw, "not clipped" implicitly since it must dominate. | `RewardCalculator.add_terminal` returns `(+10, −10)` / `(−10, +10)` / `(0, 0)` by `sim.winner`; only shaped is clipped. | (matches) no edit |
| `+0.01 per objective score point own team gains`, `−0.01 per enemy team gains`. | `SCORE_PER_SECOND_DEFAULT = 0.01`; reward uses `0.01 * (score_ticks_delta / TICK_HZ)`, i.e. 0.01/second, not 0.01 per integer score point. Since score ticks accumulate at 1/tick while controlling (and 1 score point = `kWinTicks` ticks), the effective magnitude is different from "per score point." Spec note at the bottom of §5 acknowledges "per score point" ≈ 0.01/sec under the 1/sec scoring rule; the code literally implements the 0.01/sec form. | **edit spec** — reword §5 to lead with "0.01 per second while controlling the objective" (which is what code does) and keep the "per score point" framing only as an approximate-equivalent footnote. The per-score-point framing is misleading given game-design §3 uses `team_score_ticks`, not integer score points. |
| `+0.25 enemy kill`, `−0.25 ally death`. | `KILL_BONUS_DEFAULT = 0.25`, `DEATH_PENALTY_DEFAULT = 0.25`. Kills use `sim.team_X_kills` counter delta. Deaths are *not* separately tracked; instead, the symmetrized form `raw_b = -raw_a` makes "enemy kill" and "ally death" equivalent by construction. | (matches numerically and semantically) no edit |
| `+small useful healing delivered (overheal excluded, capped)`. | Not implemented. No Mender in Phase 1–3, no healing counter is read. | **intentional drift** — Phase 1–3 is Ranger-only per curriculum §6; healing reward is a Phase-4+ (once Mender is in play) item. Spec §5 should note that the healing / damage-blocked shaped components are unimplemented until their hero is in the roster. |
| `+small damage blocked near ally (Vanguard shield, capped)`. | Not implemented. No Vanguard in Phase 1–3. | **intentional drift** — same as above. Flag in B7 as "Phase 4+ when Vanguard joins". |
| `total non-terminal shaping clipped to [-3.0, +3.0] per episode per team`. | `SHAPING_CLIP_DEFAULT = 3.0`; `_apply_clip` clips the cumulative running total to `[-3, +3]` and returns the step delta consistent with that cap. | (matches) no edit |
| `team_reward = own_events − enemy_events` applied per shaped event. | `RewardCalculator.step` computes `raw_a` from `a_score_seconds − b_score_seconds + kill_bonus * a_kills_delta − death_penalty * b_kills_delta`, then sets `raw_b = -raw_a` after clipping each side. | (matches) no edit, though note the A/B clips are applied *independently* — a pedantic reader might expect exact antisymmetry of the clipped output. In practice the two running totals stay equal-and-opposite until one side hits the cap. **edit spec** — add a one-line note that "the ±3 clip is applied per team independently; once one team's running total saturates the cap the other side's clipped return will no longer be the exact negation." |
| Team spirit scalar `team_spirit ∈ [0,1]` introduced at Phase 4, ramp 0.3 → 0.9 over first ~30% of training. | Not implemented anywhere in `reward.py` or the trainer. Env returns per-team reward; orchestration hands the learner's team slice directly to PPO. | **intentional drift** — Phase 4 is the introduction point; Phase 3 is single-agent on Team A. Note in B7 that team_spirit wiring is a Phase 4 task, and flag to user as a code-side TODO tracked against Phase 4 kickoff. |
| "Applies to shaped rewards only. The ±10 terminal reward is already a team-outcome signal." | With no team_spirit, this is moot; the env just adds terminal to shaped. | (no drift) — will be relevant once team_spirit lands. |
| Periodic evaluation with shaping disabled. | `evaluate.py` / orchestration run eval with the same `RewardCalculator` defaults; no "shaping off" eval mode exists. | **edit code** (Phase-4 scope) — surface to user: add a `shaping_clip=0` or explicit "shaping disabled" eval pass somewhere in the eval harness before Phase 4 scale-up. Not urgent for Phase 3. |
| `enemy_alive = 0` zeros enemy pos/HP/vel (obs side, shows up in §5's "no reward without context" discipline via kill events). | Actor obs: when `enemy_alive=false`, enemy_pos/vel set to 0, enemy_hp=0. | (matches; see observation_spec.md row) no edit |

### §6 — Curriculum

| Spec says | Code does | Resolution |
|---|---|---|
| "Phase 2 — Recurrent PPO on 1v1 memory toy." | `phase2_memory_toy.yaml` + `envs/memory_toy.py` + `orchestration._run_variant(use_recurrence=True/False)` runs both recurrent and feedforward to produce the gate. | (matches) no edit |
| "Phase 3 — Recurrent PPO, 1v1 Ranger, flat obs." | `phase3_ranger_recurrent.yaml` + `envs/phase3_ranger.py` + `XushiEnv` wraps 1v1 Ranger with `ACTOR_PHASE1_DIM=31` flat obs and GRU trainer. `orchestration._phase_task_spec` hardcodes `phase=3 → obs_dim=31, action_dim=6, continuous=3, binary=3`. | (matches) no edit |
| "Fixed map" for Phase 3. | `phase3_ranger_recurrent.yaml` sets `fog_of_war_enabled: false` and `randomize_map: false`. | (matches) no edit |
| "Phase 1 — Feedforward PPO, flat obs, 1v1 Ranger." | No single-file CleanRL-style FF PPO trainer for the Ranger env; the recurrent trainer with `use_recurrence=False` is the only FF path, and Phase 1 proper was effectively skipped because the recurrent trainer is what shipped. `phase1b_env_smoke.yaml` is an env smoke test, not a learning run. | **edit spec** — the ladder text in §6 should note that Phase 1 (pure FF on 1v1 Ranger) was subsumed by the Phase 2 gate — the recurrent trainer with `use_recurrence=False` serves as the FF baseline, and the "small MLP" single-file trainer described in §9 Trainer rationale was never built as a separate artifact. This is **intentional drift** — acceptable shortcut given Phase 2's recurrent-vs-FF ablation covered the pipeline validation that Phase 1 was meant to cover. |
| Phase 2/3 use of `team_spirit`. | N/A — single-agent Phase 3. | (matches — team_spirit is Phase 4+) no edit |
| Team-relative coordinate normalization for all spatial features. | `actor_obs.cpp` uses `obs_utils::mirror_position_for_team`, `mirror_velocity_for_team`, `mirror_angle_for_team` for self and enemy pos/vel/aim. Critic obs uses the actor prefix for team-frame and then adds `world_*` fields un-mirrored. | (matches) no edit |

### PPO hyperparameters (rl_design.md §8 vs `phase3_ranger_recurrent.yaml` + PPOConfig defaults)

| Spec says (§8) | Code does (Phase-3 config) | Resolution |
|---|---|---|
| Algorithm: Recurrent MAPPO | Recurrent PPO (single agent; MAPPO is Phase 4+). | **intentional drift** — note in spec §8 that the starting hyperparameters table targets the Phase 4+ MAPPO configuration; at Phase 3 we run single-agent recurrent PPO with the same hyperparameter profile minus the centralized critic. |
| Optimizer: Adam | `torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)` | (matches) no edit |
| γ = 0.997 | `phase3_ranger_recurrent.yaml`: `gamma: 0.99` | **edit spec** OR **edit code** — deliberate-ish drift: shorter horizon than spec value because Phase 3 episodes are 30s (900 ticks) and 0.997 effective horizon ≈ 333 policy steps is much longer than an episode. 0.99 gives an effective horizon ≈ 100, closer to the episode length. Recommend **edit spec** to say "γ = 0.99 for Phase-1 through Phase-3 short-horizon episodes (30s rounds); ramp to γ = 0.997 at Phase 4+ when round length grows." Flag to user: if they intended 0.997, this is a code-side change. |
| GAE λ = 0.95 | `gae_lambda: 0.95` | (matches) no edit |
| PPO clip ε = 0.10 | `clip_ratio: 0.2`, `value_clip_ratio: 0.2` | **edit spec** OR **edit code** — drift. Spec §8 explicitly calls out "keep PPO clip well under 0.2"; code uses exactly 0.2. The value-clip is a separate knob the spec doesn't enumerate. Recommend: surface to user. If the user wants conservative Phase-3 training per spec, this is an **edit code** (set 0.1). If 0.2 is a deliberate Phase-3 choice for faster convergence, **edit spec** to record the actual value and the reasoning. |
| PPO epochs per batch = 5 | `num_epochs: 4` | **edit spec** — minor drift; code uses 4. Either is within the "conservative" intent. Update §8 to 4 to match, or widen to "4–5" with a note. |
| Minibatches per epoch = 1–2 | `minibatch_size` = segments per minibatch; `num_minibatches = ceil(num_segments / minibatch_size)`. | **edit spec** — "Minibatches per epoch" is ambiguous in the spec. Replace with the explicit formulation used by the code: `minibatch_size` = segments per minibatch; `num_minibatches = ceil(num_segments / minibatch_size)`. |
| Entropy coef = 0.01, slow anneal | `entropy_coef: 0.01`; no anneal is implemented in trainer or lr_schedule. Only LR is scheduled (cosine, via `lr_for_update`). | **edit code** (non-urgent) — surface to user: spec calls for slow anneal on entropy coef; currently static. Either add an `entropy_coef_final` / `entropy_schedule` knob or **edit spec** to drop the anneal until Phase 4+. |
| Value loss coef = 0.5 | `value_coef: 0.5` | (matches) no edit |
| Gradient clipping = 0.5 | `max_grad_norm: 0.5`; `nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)` | (matches) no edit |
| Value normalization = yes | `value_normalization: true`; per-rollout return mean/std used in `_ppo_minibatch_step` to normalize value target and clip in normalized space. | (matches in spirit; different mechanism) — spec does not specify the flavor. Implementation uses per-rollout running stats, not an EMA PopArt-style normalizer. **edit spec** — add one line noting "per-rollout return-space z-scoring with clip in normalized space" so the spec agrees with the concrete implementation. |
| Advantage normalization = yes | `_ppo_minibatch_step`: `norm_adv = (advantage − adv_mean) / adv_std` over valid mask per minibatch. | (matches) no edit |
| RNN type = GRU | `models.build_model` uses GRU. | (matches) no edit |
| RNN hidden size = 256 | `phase3_ranger_recurrent.yaml`: `gru_hidden: 64`. | **edit spec** — drift: Phase 3 uses 64 (small task, 31-dim obs, shorter memory horizon). Spec §2 already allows "scale to 512 if underfitting", so adding "64 for Phase 1–3 flat-obs; scale up with observation complexity at Phase 5+" is a one-line clarification. |
| Unroll length = 64–96 | `phase3_ranger_recurrent.yaml`: `rollout_len: 256`. Note: this is *rollout length*, not BPTT unroll. BPTT unroll = segment length (episode length or up to rollout_len for non-terminating segments). Episode length at Phase 3 ≈ 100 decisions (30s × 10 Hz policy rate). | **edit spec** — disambiguate. The 64–96 figure refers to BPTT unroll per segment; with our 30-s episodes at 10 Hz we get segments ≤ 100 decisions naturally, which brackets 64–96. The trainer does not truncate BPTT within a segment. One-line note: "unroll length = full episode for Phase 1–3 (episode ≤ 100 decisions); truncate to 64–96 at Phase 5+ once episodes grow." |
| Action repeat = 2–3 ticks | `phase3_ranger_recurrent.yaml`: `action_repeat: 3`. | (matches) no edit |
| Parallel envs = 128 | `phase3_ranger_recurrent.yaml`: `num_envs: 16`. | **edit spec** — drift: Phase 3 uses 16. 128 is the scaled-training target; 16 is what the user's machine handles for the interactive-training loop. Add a phase-by-phase note. Alternatively, add a "Phase-3 overrides" subsection in §8 that lists the actual values used at each phase. |

_Audited: rollout truncation vs terminal-done handling._ With `round_length_seconds: 30`, `rollout_len: 256`, `action_repeat: 3` the rollout covers ~768 sim ticks (256 decisions × 3) vs a 900-tick episode — rollouts typically truncate mid-episode. `python/xushi2/env.py:165-183` distinguishes `terminated` (true terminal: a team wins) from `truncated` (timeout draw) and emits the ±10/0 terminal reward in both cases. `python/train/rollout_buffer.py` collapses the two into a single `done = terminated or truncated` signal and zeros the GAE bootstrap `V_{t+1}` at every `done`. This is **technically incorrect** in the classic sense (truncation-by-time-limit should bootstrap through `V_{t+1}` rather than zero it), but in this codebase the terminal reward is unconditionally added at both terminated-and-truncated episode ends, so the intended terminal signal is always present and the bootstrap-zeroing at a truncation is a conservative simplification rather than a signal-destroying bug. `docs/rl_design.md` does not currently describe this truncated-vs-terminal distinction. One-line **edit spec** candidate (not promoted to a full row): "rollouts may truncate mid-episode; the env emits a terminal reward at both true-done and timeout, and the rollout buffer zeros the GAE bootstrap at either kind of done. A future PopArt-style bootstrap through time-limit truncations is a Phase-5+ refinement."

---

## observation_spec.md

Phase-1 layout table walked against `actor_obs.cpp` / `critic_obs.cpp` / `obs_manifest.py`:

| Spec says | Code does | Resolution |
|---|---|---|
| Actor total: 31 floats (top-of-file status note) | `kActorObsPhase1Dim = 31` in `obs.h`; `ACTOR_PHASE1_DIM` sums to 31. | (matches) no edit |
| Critic total: 45 floats (top-of-file status note) | `kCriticObsPhase1Dim = 45` in `obs.h`; `CRITIC_PHASE1_DIM` sums to 45. | (matches) no edit |
| "~28 floats" legacy total in the Phase-1 table body. | Actual total 31; status note already calls out the legacy estimate. | **edit spec** — drop the "~28 floats" line from the Phase-1 table or replace with "31 (see canonical totals at top)". The status banner covers it, but the line inside the table invites future drift. |
| Field order (top of Phase-1 table): own HP, own velocity, own aim direction, own position, own revolver ammo, own reloading, own combat-roll cd, enemy alive, enemy respawn timer, enemy relative position, enemy HP, enemy velocity, objective owner one-hot, cap_team one-hot, cap progress, contested, objective unlocked, own score, enemy score, self on point, enemy on point, round timer. | `actor_obs.cpp` writes in exactly this order (see line-by-line comments). `ACTOR_PHASE1_FIELDS` tuple order matches one-for-one. | (matches) no edit |
| "own aim direction: unit vector (sin θ, cos θ)" | `actor_obs.cpp`: `angle_to_unit` then `push2(aim_unit[0], aim_unit[1])`. `obs_manifest.py` names it `own_aim_unit` "as (sin, cos)". | matches; `(sin, cos)` order confirmed at `src/sim/src/obs_utils.cpp:81-82`. No edit needed. |
| "own position: team-frame, [-1, 1] normalized to map extent." | `actor_obs.cpp` uses `mirror_position_for_team` then `normalize_position_to_map`. | (matches) no edit |
| "own revolver ammo: [0,1] = magazine / 6." | `push1(magazine / kRangerMaxMagazine)`. | (matches assuming `kRangerMaxMagazine == 6`) no edit |
| "own reloading: {0, 1}." | `push1(self.weapon.reloading ? 1.0 : 0.0)`. | (matches) no edit |
| "own combat-roll cd: [0,1] = ticks_remaining / max_cd." | `clamp01(cd_ability_1 / kRangerCombatRollCooldownTicks)`. | (matches) no edit |
| "enemy alive: {0, 1}." | `enemy_alive = enemy.present && enemy.alive`; pushed as 0/1. | (matches; note the code defines it as "present AND alive", which is the intended semantic for a 1v1 where "present=false" never happens in Phase 3 but is kept as a guard.) |
| "enemy respawn timer: [0,1]." | `respawn_norm = clamp01((respawn_tick − sim.tick) / mechanics.respawn_ticks)` when `enemy.present && !enemy.alive`, else 0. | (matches) no edit |
| "enemy relative position: team-frame, [-1, 1]." | Code computes `enemy_pos_norm − own_pos_norm` (both team-frame + map-normalized). This is enemy relative position expressed as the *delta of map-normalized coordinates*. Range is `[-2, 2]` in principle (if both are at opposite edges), not `[-1, 1]`. | **edit spec** — the spec's "[-1, 1]" bound is loose. Either (a) tighten spec to say "team-frame, delta of map-normalized positions, nominal range [-1, 1] in practice but can reach ±2 at map edges" OR (b) flag to user as **edit code** to clamp or re-normalize. Recommend (a) since learned policies handle ±2 floats fine and clamping loses information near map edges. |
| "enemy HP (normalized)." | `clamp01(enemy.health_centi_hp / enemy.max_health_centi_hp)` when alive, else 0. | (matches) no edit |
| "enemy velocity: team-frame." (no range) | `mirror_velocity_for_team(enemy.velocity, viewer)` scaled by `ranger_max_speed()`. When dead: `(0, 0)`. | (matches; spec could note the "/ ranger_max_speed" normalization, same as own velocity, for completeness.) **edit spec** — add the normalization to the Range/Encoding column. |
| "objective owner one-hot: {Neutral, Us, Them}." | `onehot_team(objective.owner, viewer)` writes `(neutral, us, them)`. Width = 3. | (matches) no edit |
| "cap_team one-hot: {None, Us, Them}." | `onehot_team(objective.cap_team, viewer)` — uses the same helper, so order is `(neutral, us, them)`. Width = 3. | (matches; spec says "None" where code says Neutral — same semantics.) no edit |
| "cap progress: [0,1]." | `clamp01(cap_progress_ticks / kCaptureTicks)`. | (matches) no edit |
| "contested: {0,1}." | `contested = a_on && b_on` derived by iterating all 6 hero slots for `position_on_objective`. | (matches; note this iteration over `s.heroes` — not a hidden-enemy iteration since at Phase 3 there's no fog and "hidden" is not defined. This will need scrutiny at Phase 7 when fog arrives; see leak-prevention invariant in observation_spec.md.) **intentional drift** — note this only for the Phase-7 handoff. |
| "objective unlocked: {0,1}." | `s.objective.unlocked ? 1 : 0`. | (matches) no edit |
| "own score: [0,1]." | `clamp01(own_score_ticks / kWinTicks)`. | (matches) no edit |
| "enemy score: [0,1]." | `clamp01(enemy_score_ticks / kWinTicks)`. | (matches) no edit |
| "self on point: {0,1}." | `self.alive && position_on_objective(self.position, map)`. Note: uses **world-frame** position, not team-frame. That is semantically fine because `position_on_objective` is team-symmetric (objective is at map center). | (matches) no edit, but note for Phase-7 reviewer: if objectives ever become team-specific, this check will need a mirror. |
| "enemy on point (public / no-fog phase): {0,1}." | `enemy_alive && position_on_objective(enemy.world_position, map)`. | (matches) no edit |
| "round timer: [0,1] = elapsed / total." | `clamp01(s.tick / round_len_ticks)` with `round_len_ticks = round_length_seconds * kTickHz`. | (matches) no edit |
| Critic always sees: true hidden-enemy positions, all cooldowns, ammo, weapon states, beam-lock targets, objective state machine internals (cap_team, cap_progress_ticks, team_score_ticks), map layout/seed, team side. | Critic obs (`CRITIC_PHASE1_FIELDS`): actor-prefix (team-frame view), plus world_own_position, world_enemy_position, world_own_velocity, world_enemy_velocity, cap_progress_ticks, team_a_score_ticks, team_b_score_ticks, tick_raw, seed_hi, seed_lo. Missing from critic: beam-lock target (Phase 4+), weapon state (Phase 4+, Mender), team side marker (team perspective is implicit in which slot's actor prefix is used — there is no explicit `team_side` scalar). | **intentional drift** — weapon/beam-lock fields are Phase 4+ once Mender enters; the spec enumerates those under "all phases" which is technically wrong for Phase 3. **edit spec**: narrow the "Critic always sees" list to a per-phase subset, or add a one-line "at Phase 3 the critic sees the flat actor prefix for team-perspective plus world-frame pos/vel + raw counters + seed; per-hero cooldown/ammo/weapon enter with the corresponding heroes in Phase 4+." Also **edit code** or **edit spec** for the missing explicit team-side scalar — if the critic is intended to learn team-conditioned values without inferring team from which slot it got its prefix, adding a `team_side` float to the critic makes training more robust. Surface to user. |
| "Enemy presence becomes three separate fields: enemy_visible, enemy_last_seen_valid, enemy_alive_public_if_known" at Phase 7. | Not implemented — Phase 7 scope. | (no drift) — flagged here only to confirm the Phase-3 code has the single `enemy_alive` field as the spec dictates. |

---

## action_spec.md

Walked against `Action` struct, `action_canon.hpp`, and `XushiEnv.action_space`.

| Spec says | Code does | Resolution |
|---|---|---|
| Field list: `move_x`, `move_y`, `aim_delta`, `primary_fire`, `ability_1`, `ability_2`, `target_slot`. | `struct Action` has exactly these 7 fields in this order. | (matches) no edit |
| `move_x: float32, [-1, 1], tanh-squashed Gaussian from policy`. | Struct field `float move_x = 0.0F`. Canonicalization clamps to `[-1, 1]` via `quantize_action_float(v, 1.0F)`. Policy emits tanh-squashed Gaussian. | (matches) no edit |
| `move_y: float32, [-1, 1]`. | Same; `quantize_action_float(v, 1.0F)`. | (matches) no edit |
| `aim_delta: float32, [-π/4, π/4]; applied once per policy decision`. | Struct field `float aim_delta = 0.0F`. Canonicalization uses `kAimDeltaMax = kPiOver4`; quantized via `quantize_action_float(v, kAimDeltaMax)`. Tick pipeline applies aim only on first sub-tick of decision (comment: "sub-tick (aim_consumed false) — impulse semantics"). | (matches) no edit |
| `primary_fire: bool (Bernoulli), {0,1}, held-semantics`. | Struct `bool primary_fire`. Held is enforced in sim (primary-fire is re-evaluated per tick during decision window). | (matches) no edit |
| `ability_1: bool, held or impulse depending on hero`. | Struct `bool ability_1`. Ranger's `ability_1` = Combat Roll = impulse, enforced in `sim_tick_pipeline.cpp` step 7 ("Combat Roll (impulse, first tick of decision)"). | (matches for Ranger; Vanguard/Mender variants are Phase 4+ scope, not yet implemented) **intentional drift** — note in spec that Vanguard Barrier (held) and Mender Weapon Swap (impulse) will be wired in at Phase 4+. |
| `ability_2: bool, held or impulse depending on hero`. | Struct `bool ability_2`. Ranger's `ability_2` is deferred per spec; Phase-3 policy still emits a Bernoulli for it (orchestration has `binary_action_dim=3`) and the sim treats it as a no-op since Ranger has no `ability_2` wired. | **edit spec** — add a note that in Phase 3 the policy still emits `ability_2` (to keep the action head shape stable) but the sim no-ops it for Ranger. Or **edit code** to drop the `ability_2` head in Phase 3 only. Recommend edit spec; the stable action head is the right call for phase transitions. |
| `target_slot: uint8 (Categorical), {0..K-1}, Phase 1–9 omitted or always 0`. | Struct `uint8_t target_slot = 0`. Canonicalization note: "target_slot pass-through for now; Phase-gate enforcement happens in sim." `XushiEnv.action_space` does **not** expose `target_slot` in the Dict (intentional, per Phase-1–9 spec). `_action_from_dict` leaves `target_slot = 0`. | (matches) no edit |
| Canonicalization: continuous int16 scale 1/10000; booleans bitfield; idempotent. | `action_canon.hpp`: `kCanonScale=10000`, `quantize_action_float` + `dequantize_action_float` round-trips. Booleans are already normalized by the `bool` type. Idempotency follows from quantize → dequantize → quantize stability. | (matches) no edit |
| Per-decision: `aim_delta` applied **once** at start of window. | Tick pipeline comment: "sub-tick (aim_consumed false) — impulse semantics." | (matches) no edit |
| Per-decision: `move_x`, `move_y` applied every tick. | Implemented in the tick pipeline movement step. | (matches) no edit |
| Per-decision: held booleans re-evaluated every tick; impulse booleans evaluated once at decision start. | Enforced in `sim_tick_pipeline.cpp` (Combat Roll gated to first sub-tick). No previous-button state is stored in `MatchState` or `HeroState` — Markovian as specified. `Grep` for `prev_primary_fire`, `prev_ability_1`, `prev_ability_2`, `rising_edge` in `src/sim` returned zero hits, confirming the "no previous-button state" invariant. | (matches) no edit |
| Invalid actions are no-ops: firing while dead / on cooldown / Barrier+Warhammer / empty mag not reloading / Weapon-Swap within 0.5s / Tether with no valid ally. | Ranger-specific invalids (cooldown, empty mag) no-op in the sim. Vanguard mutual-exclusion and Mender Weapon Swap / Tether gating are Phase 4+. | **intentional drift** — the spec lists the full Phase 1 roster's invalid rules; the code currently implements only Ranger's subset because Phase 3 is Ranger-only. Note in B7 that the Vanguard/Mender invalid-action gating lands with their respective hero kits in Phase 4+. |
| "Action distribution factorization: π_move · π_aim · π_primary · π_ability_1 · π_ability_2 [· π_target]." | Trainer: `action_logprob_and_entropy` sums `_tanh_squashed_logprob` over the 3 continuous dims (move_x, move_y, aim_delta) and `Bernoulli(logits).log_prob` over 3 binary dims (primary_fire, ability_1, ability_2). target_slot not in the distribution. | (matches; note the spec separates `π_move(move_x, move_y)` as a 2-D head and `π_aim(aim_delta)` as a 1-D head. Code puts all three continuous dims under one shared head with a per-dim log_std. This is observationally equivalent — the joint is still factorized as independent tanh-Gaussians — but doesn't match the literal "π_move · π_aim" split.) **edit spec** — reword to "π_continuous(move_x, move_y, aim_delta)" with a note that the heads are a single continuous action vector in code (same factorization). |
| Action space human-play viewer mapping (WASD / mouse / shift-space / E). | Out of scope for Phase 3 training code; viewer lives in `src/viewer/`. Not audited here. | — no row, handled elsewhere |
| `XushiEnv.action_space` is a Dict with `move_x`, `move_y`, `aim_delta`, `primary_fire`, `ability_1`, `ability_2`. | Exactly matches the spec's Phase-1–9 action schema minus `target_slot`. | (matches) no edit |
| `spaces.Box(-π/4, π/4, shape=(), dtype=float32)` for `aim_delta`. | Matches `kAimDeltaMax = kPiOver4`. | (matches) no edit |

_Audited: `aim_delta` absolute-aim accumulation / wrapping._ `action_spec.md` does not explicitly describe the wrap semantics for the accumulated absolute aim; `src/sim/src/internal/sim_tick_pipeline.cpp` (`stage_validate_and_aim`) clamps each delta to `[-kAimDeltaMax, kAimDeltaMax]` and updates `h.aim_angle = wrap_angle(h.aim_angle + delta)`, keeping the stored aim in `[-π, π]`. No drift relative to the spec's explicit claims; the spec is merely silent on the wrap. Optional spec clarification only — no row added.

---

## Summary

Row tallies (counting each concrete row above, ignoring "(matches) no edit"):

**rl_design.md §5 — Reward:** 4 drift rows
- edit spec: 2 (per-second vs per-point framing; independent-clip footnote)
- edit code: 1 (eval-with-shaping-off harness; non-urgent)
- intentional drift: 3 (healing, damage-blocked, team_spirit — all Phase 4+)

**rl_design.md §6 — Curriculum:** 1 drift row
- intentional drift: 1 (Phase 1 subsumed by Phase 2 gate)

**rl_design.md §8 — PPO hyperparameters:** 10 drift rows
- edit spec: 6 (γ, epochs, minibatches, value normalization flavor, RNN hidden, unroll length, parallel envs — "edit spec" is the default because the Phase-3 YAML represents the deliberate Phase-3 tuning)
- edit code: 2 (PPO clip 0.2 vs 0.10; entropy anneal missing — both surface to user for decision)
- intentional drift: 2 (MAPPO → single-agent PPO; 0.997 → 0.99 for short episodes)

**observation_spec.md:** 5 drift rows
- edit spec: 4 (drop "~28 floats" legacy line; tighten enemy-relative-position range; add velocity normalization note; narrow "critic always sees" list to per-phase)
- edit code: 0-1 (optional explicit `team_side` critic scalar)
- intentional drift: 2 (Phase-7 three-field enemy-presence; Phase-4+ beam/weapon critic fields)

**action_spec.md:** 3 drift rows
- edit spec: 2 (Phase-3 `ability_2` no-op note; "π_move · π_aim" factorization wording)
- edit code: 0
- intentional drift: 2 (Vanguard/Mender held/impulse and invalid-action rules — Phase 4+)

**Totals:**
- Rows marked **edit spec**: 14
- Rows marked **edit code**: 3
- Rows marked **intentional drift**: 10

**Concerning findings surfaced to user (not applied here):**

1. **PPO clip ratio 0.2 vs spec's 0.10** — the spec explicitly says "keep PPO clip well under 0.2" and cites the MAPPO paper. The code uses 0.2. If Phase-3 training is working well, this is fine and the spec should be updated to record the actual setting; if training has been unstable, this is a plausible root cause.
2. **γ = 0.99 vs spec 0.997** — justifiable given 30-s episodes at Phase 3 but worth confirming with the user that this was a deliberate choice rather than a typo on the 0.997 → 0.99 path.
3. **Entropy anneal not implemented** — §8 calls for "slow anneal" on entropy coef; trainer only schedules LR. Worth adding before Phase 4+ scale-up or removing from the spec.
4. **Critic has no explicit `team_side` scalar** — rl_design.md §3 includes "team side (A or B)" in the critic input list; current critic obs infers team from which slot's actor prefix is used. Consider adding an explicit one-hot or bit to make team-conditioned values more learnable.
5. **Reward's `_apply_clip` is applied per team independently (confirm intent)** — each team's shaped running total is clipped to `[-3, +3]` independently. Per-step antisymmetry holds until one team's cumulative shaping saturates the cap, after which the clipped step deltas are no longer exact negations. This is consistent with reading the `±3`/episode clip as an episode-scale safety net bounding the total shaping contribution per team (rather than a step-level antisymmetry guarantee). Surface to user to confirm intent; the alternative (clip the symmetrized raw value once and split) would preserve antisymmetry but changes the meaning of the cap.
6. **Phase 1 FF trainer never materialized as a separate artifact** — the recurrent trainer with `use_recurrence=False` covers the FF baseline. Curriculum §6 and §9 Trainer language still describe a separate single-file CleanRL trainer. This is acceptable path-dependency but the spec should acknowledge it.

---

## Skipped tests

Audit of `skip` / `xfail` / `GTEST_SKIP()` / `DISABLED_` markers across `tests/` (C++ GoogleTest) and `python/tests/` (pytest) as of 2026-04-24. Read-only: no markers were modified.

Search patterns used: `skip`, `xfail`, `SKIP`, `DISABLED_` (case-insensitive for the Python side; also checked `pytest.skip`, `pytest.xfail`, `pytest.importorskip`, `@pytest.mark.skip`, `@pytest.mark.xfail`, `@pytest.mark.skipif`, `unittest.skip`, `@skip`).

| File:line | Skip type | Reason | Still valid? |
|---|---|---|---|
| `tests/replay/test_golden_replay.cpp:49` | `GTEST_SKIP` | `"no golden file at " << kGoldenPath << " — generate with \`xushi2-eval --dump-golden\`"` | yes — defensive fallback only. The golden artifact `data/replays/golden_phase0_basic.txt` exists in the repo today, so this branch is not taken in normal CI; the skip triggers only if a developer deletes the file locally. Keep as-is (legitimate "fixture missing" skip, not a phase-gated skip). |
| `python/tests/test_phase3_ranger.py:10` | `pytest.importorskip("xushi2.xushi2_cpp")` | (no string message — falls through to pytest's default "could not import 'xushi2.xushi2_cpp'" message). The test imports `envs.phase3_ranger` immediately after, which hard-depends on the native C++ bindings. | yes — the native `xushi2_cpp` extension is a build artifact that is not guaranteed to be importable in a pure-Python-only environment (e.g., a doc-only venv). The skip is a correct environmental guard, not a Phase-N deferral. No change needed. |

### Summary

**Totals:** 2 skips total — 1 in C++ (`GTEST_SKIP`), 1 in Python (`pytest.importorskip`). No `DISABLED_` prefixed tests, no `@pytest.mark.skip`/`xfail` decorators, no `pytest.skip()`/`pytest.xfail()` inline calls, no `skipif` / `unittest.skip` anywhere.

**Re-enableable now:** 0. Neither skip is phase-gated or tied to a "skipped until Phase N lands" rationale — both are environmental guards (missing fixture file; missing native extension module) that correctly remain active. There is nothing for Phase 3 cleanup to re-enable.

**Surprising findings:**

- The test suite is remarkably clean of skip markers given the Phase-ladder cadence. There are **no** commented-out tests, no `TODO`-flavored skips, and no `xfail` waivers for "known bugs." This is healthy — but it also means any Phase-3/4 regressions will surface as hard failures rather than silent skips, so CI signal stays trustworthy.
- No Windows-vs-POSIX platform skips despite the repo running on Windows 11 in the dev environment and on Linux (via CMake CI) elsewhere. If platform-specific behavior emerges later (e.g., floating-point determinism across compilers), expect `skipif(sys.platform == ...)` markers to appear; none today.
- No `slow` / `integration` / `gpu` marker-based skips, and no `pytest.ini` / `pyproject.toml` entries that would auto-skip a class of tests. Every test in `python/tests/` runs on every invocation.

