from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ENTRYPOINTS = (
    "python/scripts/bench_eval.py",
    "python/scripts/bench_rollout.py",
    "python/scripts/dump_replay.py",
    "python/scripts/eval_mappo_matrix.py",
    "python/train/train.py",
    "python/train/mappo_eval_checkpoint.py",
    "python/train/mappo_matrix_eval.py",
)



def _has_adapter_import(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "train.runtime_adapter":
            if any(alias.name == "resolve_runtime_env_factory" for alias in node.names):
                return True
    return False



def _has_adapter_call(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "resolve_runtime_env_factory":
                return True
    return False


def test_entrypoints_use_public_runtime_adapter_contract() -> None:
    missing: list[str] = []
    for rel_path in ENTRYPOINTS:
        source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel_path)
        if not _has_adapter_import(tree) or not _has_adapter_call(tree):
            missing.append(rel_path)

    assert not missing, (
        "entrypoints must import and call train.runtime_adapter.resolve_runtime_env_factory: "
        + ", ".join(missing)
    )
