"""Deterministic per-env opponent assignment for the opponent-mix curriculum.

`env.opponent_bot_mix` maps scripted-bot names to positive weights. The
trainer assigns one bot per vector-env slot so that slot counts match the
weight proportions as closely as integer slots allow. The assignment must be
a pure function of (mix, num_envs): no RNG, so runs reproduce and the async
backend (which pickles env state per worker) sees a stable assignment.
"""

from __future__ import annotations

from collections.abc import Sequence

from xushi2.runner import VALID_BOT_NAMES as VALID_OPPONENT_BOTS


def parse_opponent_bot_mix(raw: object) -> dict[str, float]:
    """Validate an `env.opponent_bot_mix` config block. Empty dict = disabled."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"env.opponent_bot_mix must be a mapping, got {type(raw)!r}")
    mix: dict[str, float] = {}
    for bot, weight in raw.items():
        name = str(bot)
        if name.startswith("snapshot:"):
            if not name[len("snapshot:"):]:
                raise ValueError(
                    "env.opponent_bot_mix: snapshot entry requires a checkpoint path"
                )
        elif name not in VALID_OPPONENT_BOTS:
            raise ValueError(
                f"env.opponent_bot_mix: unknown bot {name!r}; "
                f"valid: {sorted(VALID_OPPONENT_BOTS)} or snapshot:<path>"
            )
        value = float(weight)
        if value <= 0.0:
            raise ValueError(
                f"env.opponent_bot_mix: weight for {name!r} must be > 0, got {value}"
            )
        mix[name] = value
    return mix


def recent_self_mix(
    update_idx: int,
    *,
    checkpoint_every: int,
    lags: Sequence[int],
    share: float,
    anchor_mix: dict[str, float],
    output_dir: str,
    fallback_path: str,
) -> dict[str, float]:
    """Opponent mix for iterated self-play against the learner's own recent
    checkpoints. Each lag targets the checkpoint-grid update at
    ``update_idx - lag`` (falling back to ``fallback_path`` — normally the
    warm start — before that checkpoint exists), so the skill gap stays
    small by construction. Pure function of its inputs: refreshing at every
    checkpoint event is deterministic and resume-safe. Duplicate paths
    merge their weights; ``anchor_mix`` entries are added as-is."""
    if checkpoint_every <= 0:
        raise ValueError(f"checkpoint_every must be > 0, got {checkpoint_every}")
    lag_list = [int(lag) for lag in lags]
    if not lag_list or any(lag <= 0 for lag in lag_list):
        raise ValueError(f"lags must be positive, got {list(lags)}")
    if not 0.0 < float(share) <= 1.0:
        raise ValueError(f"share must be in (0, 1], got {share}")
    mix: dict[str, float] = {}
    each = float(share) / len(lag_list)
    for lag in lag_list:
        grid = ((int(update_idx) - lag) // checkpoint_every) * checkpoint_every
        if grid >= checkpoint_every:
            path = f"{output_dir}/ckpt_{grid:04d}.pt"
        else:
            path = str(fallback_path)
        key = f"snapshot:{path}"
        mix[key] = mix.get(key, 0.0) + each
    for bot, weight in anchor_mix.items():
        mix[str(bot)] = mix.get(str(bot), 0.0) + float(weight)
    return mix


def opponent_mix_assignment(mix: dict[str, float], num_envs: int) -> list[str]:
    """Deterministic, interleaved apportionment of bots over env slots.

    Greedy largest-deficit choice per slot: slot k goes to the bot whose
    assigned count lags its weight share of (k+1) the most, ties broken by
    bot name. Interleaving falls out naturally (a 10% bot lands mid-sequence,
    not bunched at the end), which keeps every rollout batch mixed.
    """
    if num_envs <= 0:
        raise ValueError(f"num_envs must be > 0, got {num_envs}")
    if not mix:
        raise ValueError("opponent_mix_assignment requires a non-empty mix")
    items = sorted((str(bot), float(weight)) for bot, weight in mix.items())
    total = sum(weight for _, weight in items)
    counts = {bot: 0 for bot, _ in items}
    assignment: list[str] = []
    for k in range(int(num_envs)):
        best_bot = items[0][0]
        best_deficit = None
        for bot, weight in items:
            deficit = (weight / total) * (k + 1) - counts[bot]
            if best_deficit is None or deficit > best_deficit + 1e-12:
                best_bot = bot
                best_deficit = deficit
        counts[best_bot] += 1
        assignment.append(best_bot)
    return assignment
