# Local Build & Test Guide

## Prerequisites

### System packages (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y cmake g++ libpython3.12-dev python3.12 python3.12-venv
```

Minimum versions verified on this machine:

| Tool | Version |
|------|---------|
| CMake | 3.31.10 (≥ 3.24 required) |
| GCC | 13.3.0 (C++20 required) |
| Python | 3.12.3 (≥ 3.10 supported) |
| libpython3.12-dev | 3.12.3-1ubuntu0.13 |

### Optional: viewer X11 dependencies

The raylib viewer needs X11 development headers. If they are missing,
configure with `-DXUSHI2_BUILD_VIEWER=OFF` (the default CI configuration).

```bash
sudo apt install libx11-dev libxrandr-dev libxi-dev libxcursor-dev libxinerama-dev
```

---

## C++ Build (single fixed directory)

Use **`build/`** as the only build directory. Do not create ad-hoc directories
like `build-py312` or `build_fpic`.

### Configure

```bash
cd /home/aspect/source/personal/xushi2
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=/usr/bin/python3.12 \
    -DXUSHI2_BUILD_VIEWER=OFF
```

Flags explained:
- `-DCMAKE_BUILD_TYPE=Release` — disables asserts, enables `-O3`.
- `-DPYTHON_EXECUTABLE=/usr/bin/python3.12` — pins the Python version for the
  pybind11 extension. Omit this only if your default `python3` is already 3.12.
- `-DXUSHI2_BUILD_VIEWER=OFF` — skips raylib when X11 headers are absent.

### Build

```bash
cmake --build build -j$(nproc)
```

### C++ tests

```bash
ctest --test-dir build --output-on-failure
```

Expected: 86/87 pass. The single pre-existing failure is
`GoldenReplay.BasicVsBasicMatchesGoldenTrajectory` — the golden replay was
recorded before a recent sim change and needs regeneration (not a build issue).

---

## Python Extension Build

The C++ extension `xushi2_cpp` is produced by the CMake step above and placed
in `python/xushi2/` automatically (`src/python_bindings/CMakeLists.txt` sets
`XUSHI2_PY_EXT_OUTPUT_DIR`).

### Virtual environment

```bash
cd /home/aspect/source/personal/xushi2
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

> **Note:** On Debian/Ubuntu `python3.12 -m venv` requires the
> `python3.12-venv` package. If it is not installed, use
> `apt install python3.12-venv`.

### Install Python package (editable)

```bash
cd python
pip install -e ".[dev]"
```

This installs the pure-Python trainer, eval harness, and test dependencies.
The binary `xushi2_cpp` is already in `python/xushi2/` from the CMake build.

### Verify the extension loads

```bash
cd /home/aspect/source/personal/xushi2
python3.12 -c "import sys; sys.path.insert(0, 'python'); import xushi2.xushi2_cpp; print('OK')"
```

### Python tests

```bash
cd /home/aspect/source/personal/xushi2/python
pytest tests/ -v
```

#### Partial test run without heavy ML deps

If `torch` or `gymnasium` are not installed (they are large and may be omitted
on a minimal CI node), run the subset that only needs `numpy` and the C++
extension:

```bash
cd /home/aspect/source/personal/xushi2/python
pytest tests/test_extension_present.py tests/test_bindings_obs.py \
    tests/test_obs_manifest.py tests/test_phase0_determinism.py \
    tests/test_reward.py -v
```

Current status on a clean Python 3.12 node without `torch`/`gymnasium`:
- `test_extension_present.py` — 1 passed
- `test_bindings_obs.py` — 10 passed
- `test_obs_manifest.py` — 16 passed
- `test_phase0_determinism.py` — 7 passed
- `test_reward.py` — 16 passed, 1 skipped (needs `gymnasium` for env smoke)

The full suite additionally requires `torch`, `gymnasium`, `pyyaml`, `tqdm`,
`wandb`, and `tensorboard` (see `python/pyproject.toml` for exact pins).

---

## Build troubleshooting

### `relocation R_X86_64_PC32 ... can not be used when making a shared object`

If you see this while linking `xushi2_cpp`, the static `xushi2_sim` and
`xushi2_bots` libraries were not compiled with `-fPIC`.
This was fixed in `src/sim/CMakeLists.txt` and `src/bots/CMakeLists.txt` by
adding:

```cmake
set_target_properties(xushi2_sim PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(xushi2_bots PROPERTIES POSITION_INDEPENDENT_CODE ON)
```

If you are on a commit before that fix, apply the same lines or rebuild with
`-DCMAKE_POSITION_INDEPENDENT_CODE=ON`.

### `Could NOT find X11`

Either install the X11 dev packages listed above, or disable the viewer:
`-DXUSHI2_BUILD_VIEWER=OFF`.

### `ModuleNotFoundError: No module named 'gymnasium'`

Install the full dependency set:

```bash
pip install -e ".[dev]"
```

Or, if you only need the C++ extension smoke tests, run the partial pytest
command shown above.

---

## Clean rebuild

```bash
cd /home/aspect/source/personal/xushi2
cmake --build build --clean-first -j$(nproc)
```

To fully wipe and reconfigure:

```bash
rm -rf build/
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3.12
```

