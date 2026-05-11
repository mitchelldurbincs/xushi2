# Plans Folder Policy

## Routing rules

Place plan files in `docs/plans/active/` when they are still driving current work.
Move plan files to `docs/plans/archive/` when any of the following is true:

1. **Completed**: implementation and validation are done (`Status: done`).
2. **Superseded**: a newer plan replaced it as source of truth.
3. **Blocked for more than 14 days**: unresolved blockers have paused execution for >14 days.
4. **Historical-only**: the document is no longer used for active decisions.

## Required active-plan header

Every file in `docs/plans/active/` must include this header near the top:

- `Status: active|blocked|done`
- `Owner: <name or team>`
- `Last-updated: YYYY-MM-DD`

When `Status: done`, move the file to `archive/` in the same change.
When `Status: blocked`, update `Last-updated` on each blocker review; move to `archive/` once blocked for >14 days unless reactivated.

## Naming

- Use date-first names: `YYYY-MM-DD-topic.md`
- Result summaries belong in `archive/`: `YYYY-MM-DD-topic-result.md`
