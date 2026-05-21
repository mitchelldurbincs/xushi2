"""Build a phase-gate RunEvidence JSON from a run directory.

Pulls eval/canonical_eval metrics from a W&B run history, matrix/anchor/<bot>/*
metrics from local anchor_eval_*.json artifacts, and identity/artifact metadata
from launch.log and the checkpoint manifest. One-shot helper for Phase 4 gate
runs; not wired into general training.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import yaml


_WANDB_URL_RE = re.compile(r"(https://wandb\.ai/[A-Za-z0-9\-_/]+/runs/[A-Za-z0-9]+)")
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CRASH_RE = re.compile(
    r"Traceback|(?:^|\b)(?:Error:|error:)|\bFAILED\b|assert\s|Killed|OOM|RuntimeError|ImportError",
    re.IGNORECASE,
)
_NAN_RE = re.compile(r"\bnan\b|NaN", re.IGNORECASE)
_IMPORT_ERR_RE = re.compile(r"ImportError|ModuleNotFoundError")


def _scan_launch_log(path: Path) -> tuple[str | None, bool, bool, bool]:
    """Return (wandb_url, crashed, saw_nan, import_error) from launch.log."""
    if not path.exists():
        return None, False, False, False
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    text = _ANSI_RE.sub("", raw.decode(encoding, errors="replace"))
    wandb_match = _WANDB_URL_RE.search(text)
    wandb_url = wandb_match.group(1) if wandb_match else None
    crashed = bool(_CRASH_RE.search(text))
    import_err = bool(_IMPORT_ERR_RE.search(text))
    # Ignore "nan" appearing in random ASCII; only treat as real signal if line
    # contains explicit nan-loss markers.
    saw_nan = any(
        "nan" in line.lower() and any(tag in line for tag in ("loss", "grad", "reward"))
        for line in text.splitlines()
    )
    return wandb_url, crashed, saw_nan, import_err


def _matrix_metrics_from_anchor_files(
    files: Iterable[Path],
) -> dict[str, list[float]]:
    """Aggregate matrix/anchor/<bot>/<key> metrics across one or more JSON files.

    Each file is a list of matchup rows (dicts) with at minimum:
      learner, opponent, opponent_type='bot', win_rate, loss_rate, draw_rate,
      mean_reward, mean_score_a, mean_score_b, wins, losses, mean_kills_a, ...
    The gate cares about a few keys per bot. We export everything that looks
    numeric so the gate can read whatever metric IDs the config references.
    """
    metrics: dict[str, list[float]] = {}
    for file_path in files:
        rows = json.loads(file_path.read_text(encoding="utf-8"))
        for row in rows:
            if row.get("opponent_type") != "bot":
                continue
            bot = str(row.get("opponent", "?"))
            for key, value in row.items():
                if key in {"learner", "learner_team", "opponent", "opponent_type"}:
                    continue
                if not isinstance(value, (int, float)):
                    continue
                metric_name = f"matrix/anchor/{bot}/{key}"
                metrics.setdefault(metric_name, []).append(float(value))
    return metrics


def _wandb_history_metrics(wandb_url: str | None) -> dict[str, list[float]]:
    """Fetch eval/* and canonical_eval/* metrics from a W&B run history.

    Returns empty dict if wandb_url is None or the run can't be reached.
    """
    if not wandb_url:
        return {}
    try:
        import wandb
    except ImportError:
        print("[evidence] wandb not installed; skipping history fetch", file=sys.stderr)
        return {}
    # Parse entity/project/run_id out of the URL.
    match = re.match(r"https://wandb\.ai/([^/]+)/([^/]+)/runs/([^/?]+)", wandb_url)
    if match is None:
        print(f"[evidence] could not parse wandb URL: {wandb_url}", file=sys.stderr)
        return {}
    entity, project, run_id = match.groups()
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        history = list(run.scan_history())
    except Exception as exc:  # noqa: BLE001 - W&B raises many error types
        print(f"[evidence] wandb history fetch failed: {exc}", file=sys.stderr)
        return {}
    metrics: dict[str, list[float]] = {}
    for entry in history:
        numeric_entry: dict[str, float] = {}
        for key, value in entry.items():
            if not isinstance(value, (int, float)):
                continue
            if not (key.startswith("eval/") or key.startswith("canonical_eval/")):
                continue
            numeric_entry[key] = float(value)
            metrics.setdefault(key, []).append(float(value))
        for prefix in ("eval", "canonical_eval"):
            mean_kills_a = numeric_entry.get(f"{prefix}/mean_kills_a")
            mean_kills_b = numeric_entry.get(f"{prefix}/mean_kills_b")
            episodes = numeric_entry.get(f"{prefix}/episodes")
            if mean_kills_a is not None and episodes is not None:
                metrics.setdefault(f"{prefix}/team_a_kills", []).append(
                    float(mean_kills_a) * float(episodes)
                )
            if mean_kills_b is not None and episodes is not None:
                metrics.setdefault(f"{prefix}/team_b_kills", []).append(
                    float(mean_kills_b) * float(episodes)
                )
    return metrics


def _config_seeds(config_path: Path) -> list[int]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    env = cfg.get("env", {}) or {}
    seed = env.get("seed_base")
    if seed is None:
        sim = env.get("sim", {}) or {}
        seed = sim.get("seed")
    return [int(seed)] if seed is not None else []


def build_evidence(
    *,
    run_dir: Path,
    config_path: Path,
    git_commit: str,
    replay_artifacts: list[Path],
    viewer_command_template: str = "xushi2-viewer --replay {replay_path}",
) -> dict:
    launch_log = run_dir / "launch.log"
    wandb_url, crashed, saw_nan, import_err = _scan_launch_log(launch_log)
    matrix_files = [
        *sorted((run_dir / "mappo").glob("anchor_eval_*.json")),
        *sorted((run_dir / "mappo").glob("matrix_eval*.json")),
    ]
    matrix_metrics = _matrix_metrics_from_anchor_files(matrix_files)
    wandb_metrics = _wandb_history_metrics(wandb_url)
    metrics: dict[str, list[float]] = {**wandb_metrics, **matrix_metrics}

    replay_paths = [str(p.as_posix()) for p in replay_artifacts]
    viewer_command = (
        viewer_command_template.format(replay_path=replay_paths[0])
        if replay_paths
        else None
    )
    seeds = _config_seeds(config_path)
    evidence = {
        "run_id": f"{run_dir.name}+{(wandb_url or 'no-wandb-id').rsplit('/', 1)[-1]}",
        "git_commit": git_commit,
        "config_path": str(config_path.as_posix()),
        "seeds": seeds,
        "wandb_run_url": wandb_url,
        "replay_artifacts": replay_paths,
        "viewer_command": viewer_command,
        "crashed": crashed,
        "saw_nan": saw_nan,
        "import_error": import_err,
        "timed_out_before_evidence": False,
        "metrics": metrics,
    }
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Build phase-gate RunEvidence JSON")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--git-commit", type=str, required=True)
    parser.add_argument(
        "--replay",
        type=Path,
        action="append",
        default=[],
        help="Path to a replay artifact (repeatable)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evidence = build_evidence(
        run_dir=args.run_dir,
        config_path=args.config,
        git_commit=args.git_commit,
        replay_artifacts=list(args.replay),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"[evidence] wrote {args.output}")
    metric_summary = {
        key: len(values) for key, values in sorted(evidence["metrics"].items())
    }
    print("[evidence] metrics:")
    for key, count in metric_summary.items():
        print(f"  {key}: {count} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
