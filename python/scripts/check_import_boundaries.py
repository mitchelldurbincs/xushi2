"""Check Python package import-direction boundaries."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_XUSHI2_TRAIN_IMPORTERS = {"xushi2.snapshot_policy"}
PHASE_RUNTIME_BRANCH_ALLOWLIST = {
    "train.phases",
    "train.runtime_specs",
    "train.checkpoint_runtime",
}
TRAIN_PHASE_IMPORT_ALLOWLIST = {
    "train.runtime_specs",
}
PHASE_PRIVATE_ENV_IMPORT_ALLOWLIST = {
    "envs.__init__",
    "envs.runtime_factory",
}


@dataclass(frozen=True)
class Violation:
    path: Path
    lineno: int
    message: str


def _iter_py_files() -> list[Path]:
    return [
        p
        for p in ROOT.rglob("*.py")
        if "__pycache__" not in p.parts and ".venv" not in p.parts
    ]


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


def _is_test_file(path: Path) -> bool:
    return "tests" in path.relative_to(ROOT).parts


def _name_text(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_text(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        if node.args:
            inner = _name_text(node.args[0])
            if inner is not None:
                return inner
        return _name_text(node.func)
    return None


def _looks_like_phase_runtime_compare(node: ast.Compare) -> bool:
    names = [_name_text(node.left), *(_name_text(c) for c in node.comparators)]
    if not any(name in {"phase", "wanted_phase", "raw_phase"} for name in names):
        return False
    candidates = [node.left, *node.comparators]
    for candidate in candidates:
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, int):
            return True
        if isinstance(candidate, ast.Tuple) and any(
            isinstance(elt, ast.Constant) and isinstance(elt.value, int)
            for elt in candidate.elts
        ):
            return True
    return False


def _check_file(path: Path) -> list[Violation]:
    rel = path.relative_to(ROOT)
    mod = _module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[Violation] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for target in _import_targets(node):
                if mod.startswith("xushi2.") and target.startswith("envs"):
                    violations.append(Violation(rel, node.lineno, f"xushi2 layer cannot import {target!r}"))
                if mod.startswith("xushi2.") and target.startswith("train") and mod not in ALLOWED_XUSHI2_TRAIN_IMPORTERS:
                    violations.append(Violation(rel, node.lineno, f"xushi2 layer cannot import {target!r}"))
                if mod.startswith("train.") and _is_phase_private_env_module(target):
                    violations.append(Violation(rel, node.lineno, f"train layer cannot import phase-private env module {target!r}"))
                if (mod.startswith("scripts.") or mod.startswith("eval.")) and _is_phase_private_env_module(target):
                    violations.append(Violation(rel, node.lineno, f"scripts/eval cannot import phase-private env module {target!r}"))
                if (
                    not _is_test_file(path)
                    and target == "train.phases"
                    and mod not in TRAIN_PHASE_IMPORT_ALLOWLIST
                ):
                    violations.append(Violation(rel, node.lineno, "production code must not import train.phases outside the legacy adapter"))
                if (
                    not _is_test_file(path)
                    and _is_phase_private_env_module(target)
                    and mod not in PHASE_PRIVATE_ENV_IMPORT_ALLOWLIST
                    and not mod.startswith("envs.phase")
                ):
                    violations.append(Violation(rel, node.lineno, f"production code must use envs.runtime_factory instead of importing {target!r}"))
        if (
            isinstance(node, ast.Compare)
            and not _is_test_file(path)
            and mod not in PHASE_RUNTIME_BRANCH_ALLOWLIST
            and _looks_like_phase_runtime_compare(node)
        ):
            violations.append(Violation(rel, node.lineno, "production code must not branch runtime behavior on numeric phase"))

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
