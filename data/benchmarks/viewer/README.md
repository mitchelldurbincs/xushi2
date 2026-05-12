# Viewer benchmark fixtures

These fixtures are committed and versioned so viewer benchmark runs are stable across time.

- `minimal_scene.replay` — Minimal visual complexity: no fog/overlays, no map markers, and very short decision stream. Exists to track baseline render/playback overhead.
- `typical_match_scene.replay` — Typical complexity: fog + target-slot metadata, mixed hero roster, a handful of cover/wall markers, and representative per-agent action tokens. Exists to represent common replay inspection workloads.
- `stress_scene.replay` — High complexity: fog + last-seen + target-slot enabled with dense cover/wall marker lists and full six-agent tokenized actions. Exists to stress debug overlay drawing and metadata parsing paths.
