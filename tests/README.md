# tests/

C++ tests for the simulation core, wired up via `tests/CMakeLists.txt` and run
under CTest. Layout mirrors `src/`:

- `sim/` — deterministic sim, config validation
- `observations/` — actor/critic obs builders, obs dims, leak checks
- `bots/` — scripted bot runners
- `replay/` — replay format round-trips
- `integration/` — end-to-end smoke
- `common/` — shared test helpers (e.g. `test_config.hpp`)

Run with:

```bash
make test-cpp
# or
ctest --test-dir build --output-on-failure
```

Python tests live separately under [`python/tests/`](../python/tests/) and run
via `pytest python/tests`.
