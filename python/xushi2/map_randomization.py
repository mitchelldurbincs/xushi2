"""Deterministic map randomization for Phase 8 diagnostics."""

from __future__ import annotations

import numpy as np

DEFAULT_MAP_BOUNDS: dict[str, float] = {
    "min_x": 0.0,
    "min_y": 0.0,
    "max_x": 50.0,
    "max_y": 50.0,
}

_FNV64_OFFSET = 0xCBF29CE484222325
_FNV64_PRIME = 0x100000001B3


def _fnv1a64_update(h: int, text: str) -> int:
    for byte in text.encode("ascii"):
        h ^= byte
        h = (h * _FNV64_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    return h


def randomized_map_bounds(seed: int, cfg: dict | None = None) -> dict[str, float]:
    """Return deterministic symmetric arena bounds for one episode.

    This first Phase-8 probe randomizes arena scale while keeping the center
    fixed. Topology helpers can then add deterministic cover and wall segments
    inside those bounds.
    """
    cfg = dict(cfg or {})
    base = dict(DEFAULT_MAP_BOUNDS)
    base.update(dict(cfg.get("base_bounds", {})))
    jitter = float(cfg.get("span_jitter", 5.0))
    min_span = float(cfg.get("min_span", 40.0))
    max_span = float(cfg.get("max_span", 60.0))

    cx = 0.5 * (float(base["min_x"]) + float(base["max_x"]))
    cy = 0.5 * (float(base["min_y"]) + float(base["max_y"]))
    base_w = float(base["max_x"]) - float(base["min_x"])
    base_h = float(base["max_y"]) - float(base["min_y"])

    rng = np.random.default_rng(int(seed) & 0xFFFF_FFFF_FFFF_FFFF)
    width = float(np.clip(base_w + rng.uniform(-jitter, jitter), min_span, max_span))
    height = float(np.clip(base_h + rng.uniform(-jitter, jitter), min_span, max_span))
    return {
        "min_x": cx - 0.5 * width,
        "min_y": cy - 0.5 * height,
        "max_x": cx + 0.5 * width,
        "max_y": cy + 0.5 * height,
    }


def randomized_cover_markers(seed: int, cfg: dict | None = None) -> list[dict[str, float]]:
    """Return deterministic symmetric cover marker positions.

    These markers are Phase-8 topology metadata and native circular cover
    pillars when passed through ``sim.cover_circles``.
    """
    cfg = dict(cfg or {})
    bounds = randomized_map_bounds(seed, cfg)
    count_per_side = int(cfg.get("cover_count_per_side", 2))
    jitter = float(cfg.get("cover_jitter", 2.0))
    radius = float(cfg.get("cover_radius", 1.0))
    min_x = float(bounds["min_x"])
    max_x = float(bounds["max_x"])
    min_y = float(bounds["min_y"])
    max_y = float(bounds["max_y"])
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    half_w = 0.5 * (max_x - min_x)
    half_h = 0.5 * (max_y - min_y)

    rng = np.random.default_rng((int(seed) ^ 0xC0A5_7EED) & 0xFFFF_FFFF_FFFF_FFFF)
    markers: list[dict[str, float]] = []
    for idx in range(max(0, count_per_side)):
        frac = (idx + 1) / float(count_per_side + 1)
        x = cx + (0.18 + 0.22 * frac) * half_w
        y = cy + (0.16 + 0.30 * frac) * half_h
        x += float(rng.uniform(-jitter, jitter))
        y += float(rng.uniform(-jitter, jitter))
        x = float(np.clip(x, min_x + 2.0, max_x - 2.0))
        y = float(np.clip(y, min_y + 2.0, max_y - 2.0))
        markers.append({"x": x, "y": y, "radius": radius})
        markers.append({"x": 2.0 * cx - x, "y": 2.0 * cy - y, "radius": radius})
    markers.sort(key=lambda p: (p["x"], p["y"]))
    return markers


def randomized_wall_segments(seed: int, cfg: dict | None = None) -> list[dict[str, float]]:
    """Return deterministic symmetric wall segments for Phase-8 topology probes."""
    cfg = dict(cfg or {})
    bounds = randomized_map_bounds(seed, cfg)
    count_per_side = int(cfg.get("wall_count_per_side", 1))
    jitter = float(cfg.get("wall_jitter", 1.0))
    half_width = float(cfg.get("wall_half_width", 0.25))
    length = float(cfg.get("wall_length", 5.0))
    min_x = float(bounds["min_x"])
    max_x = float(bounds["max_x"])
    min_y = float(bounds["min_y"])
    max_y = float(bounds["max_y"])
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    half_w = 0.5 * (max_x - min_x)
    half_h = 0.5 * (max_y - min_y)

    rng = np.random.default_rng((int(seed) ^ 0xA11E_5EED) & 0xFFFF_FFFF_FFFF_FFFF)
    walls: list[dict[str, float]] = []
    for idx in range(max(0, count_per_side)):
        frac = (idx + 1) / float(count_per_side + 1)
        x = cx + (0.24 + 0.16 * frac) * half_w
        y = cy + (0.02 + 0.20 * frac) * half_h
        x += float(rng.uniform(-jitter, jitter))
        y += float(rng.uniform(-jitter, jitter))
        y1 = float(np.clip(y - 0.5 * length, min_y + 2.0, max_y - 2.0))
        y2 = float(np.clip(y + 0.5 * length, min_y + 2.0, max_y - 2.0))
        x = float(np.clip(x, min_x + 2.0, max_x - 2.0))
        wall = {"x1": x, "y1": y1, "x2": x, "y2": y2, "half_width": half_width}
        mirror = {
            "x1": 2.0 * cx - x,
            "y1": 2.0 * cy - y1,
            "x2": 2.0 * cx - x,
            "y2": 2.0 * cy - y2,
            "half_width": half_width,
        }
        if mirror["y1"] > mirror["y2"]:
            mirror["y1"], mirror["y2"] = mirror["y2"], mirror["y1"]
        walls.append(wall)
        walls.append(mirror)
    walls.sort(key=lambda p: (p["x1"], p["y1"], p["x2"], p["y2"]))
    return walls


def map_layout_hash(
    bounds: dict[str, float],
    covers: list[dict[str, float]] | tuple[dict[str, float], ...],
    walls: list[dict[str, float]] | tuple[dict[str, float], ...] = (),
) -> str:
    """Return a stable 64-bit hash for rounded Phase-8 layout geometry."""
    h = _FNV64_OFFSET
    for key in ("min_x", "min_y", "max_x", "max_y"):
        h = _fnv1a64_update(h, f"{key}={float(bounds[key]):.3f};")
    for cover in sorted(covers, key=lambda p: (float(p["x"]), float(p["y"]))):
        h = _fnv1a64_update(
            h,
            (
                f"c={float(cover['x']):.3f}:"
                f"{float(cover['y']):.3f}:"
                f"{float(cover.get('radius', 1.0)):.3f};"
            ),
        )
    for wall in sorted(walls, key=lambda p: (float(p["x1"]), float(p["y1"]))):
        h = _fnv1a64_update(
            h,
            (
                f"w={float(wall['x1']):.3f}:"
                f"{float(wall['y1']):.3f}:"
                f"{float(wall['x2']):.3f}:"
                f"{float(wall['y2']):.3f}:"
                f"{float(wall.get('half_width', 0.25)):.3f};"
            ),
        )
    return f"0x{h:016x}"


def sim_cfg_with_map_bounds(sim_cfg: dict, bounds: dict[str, float]) -> dict:
    out = dict(sim_cfg)
    out["map"] = {k: float(bounds[k]) for k in ("min_x", "min_y", "max_x", "max_y")}
    return out
