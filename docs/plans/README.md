# Plans Folder Routing

This directory is split into two buckets:

- `active/`: currently relevant plans/design notes that should guide ongoing implementation and review work.
- `archive/`: dated result logs (`*-result.md`) plus superseded design/implementation notes that are kept only for historical traceability.

## Naming convention

Use date-first filenames so files sort chronologically:

- Format: `YYYY-MM-DD-topic.md`
- Result logs: `YYYY-MM-DD-topic-result.md` (store in `archive/`)
- Design/plan notes: `YYYY-MM-DD-topic-design.md` or `YYYY-MM-DD-topic.md` (store in `active/` unless superseded)

## Where to put new docs (decision checklist)

1. Is this document intended to drive current or near-term implementation decisions?
   - **Yes** → put it in `active/`.
2. Is this a completed run log, experiment outcome, or phase result summary (`*-result.md`)?
   - **Yes** → put it in `archive/`.
3. Is this doc no longer the source of truth because a newer plan/design replaced it?
   - **Yes** → move it to `archive/` and link to the replacement from commit/PR context.
4. Unsure?
   - Start in `active/`, then move to `archive/` once it is clearly superseded or only historical.

## Current canonical plans

Keep this list intentionally small and update it whenever canonical planning docs change.

- [2026-05-08-phase4-cap-training-escalation.md](active/2026-05-08-phase4-cap-training-escalation.md)
- [2026-05-08-phase4-cap-training-escalation-design.md](active/2026-05-08-phase4-cap-training-escalation-design.md)
- [2026-05-08-team-spirit-per-agent-rewards.md](active/2026-05-08-team-spirit-per-agent-rewards.md)
