"""Check Python package import-direction boundaries."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_XUSHI2_TRAIN_IMPORTERS = {"xushi2.snapshot_policy"}


@dataclass(frozen=True)
class Violation:
    path: Path
    lineno: int
    message: str


def _iter_py_files() -> list[Path]:
    return [p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts]


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def _import_targets(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.module is None:
            return []
        return [node.module]
    return []


def _is_phase_private_env_module(name: str) -> bool:
    return name.startswith("envs.phase")


def _check_file(path: Path) -> list[Violation]:
    rel = path.relative_to(ROOT)
    mod = _module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[Violation] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for target in _import_targets(node):
            if mod.startswith("xushi2.") and target.startswith("envs"):
                violations.append(Violation(rel, node.lineno, f"xushi2 layer cannot import {target!r}"))
            if mod.startswith("xushi2.") and target.startswith("train") and mod not in ALLOWED_XUSHI2_TRAIN_IMPORTERS:
                violations.append(Violation(rel, node.lineno, f"xushi2 layer cannot import {target!r}"))
            if mod.startswith("train.") and _is_phase_private_env_module(target):
                violations.append(Violation(rel, node.lineno, f"train layer cannot import phase-private env module {target!r}"))
            if (mod.startswith("scripts.") or mod.startswith("eval.")) and _is_phase_private_env_module(target):
                violations.append(Violation(rel, node.lineno, f"scripts/eval cannot import phase-private env module {target!r}"))

    return violations


def main() -> int:
    violations: list[Violation] = []
    for path in _iter_py_files():
        violations.extend(_check_file(path))
    if not violations:
        print("[check_import_boundaries] PASS")
        return 0
    print(f"[check_import_boundaries] FAIL violations={len(violations)}")
    for v in sorted(violations, key=lambda x: (str(x.path), x.lineno, x.message)):
        print(f"{v.path}:{v.lineno}: {v.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
