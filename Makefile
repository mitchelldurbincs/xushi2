.PHONY: build-cpp build-cpp-headless test-cpp bench-cpp run-bench bench-smoke py-install train-smoke format lint clean bench-viewer

# Override on platforms where the system Python interpreter is named `python`
# (Windows, some macOS setups). Default matches Linux convention where
# `python` may not exist (e.g. stock Ubuntu without `python-is-python3`).
PYTHON ?= python3
VENV := python/.venv
VENV_PY := $(CURDIR)/$(VENV)/bin/python
VENV_RUFF := $(CURDIR)/$(VENV)/bin/ruff

BENCH_TARGETS := bench_sim_tick bench_obs_build bench_bot_decision

build-cpp:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j

# Headless variant for servers/containers without X11 dev headers (skips the
# raylib viewer). Use this on the training executor box.
build-cpp-headless:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DXUSHI2_BUILD_VIEWER=OFF
	cmake --build build -j

test-cpp: build-cpp
	ctest --test-dir build --output-on-failure

bench-cpp:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DXUSHI2_BUILD_BENCHMARKS=ON
	cmake --build build -j --target benchmarks

# Run all Google Benchmark binaries with full repetitions. JSON output
# lands under build/benchmarks/results/ keyed by binary name.
run-bench: bench-cpp
	mkdir -p build/benchmarks/results
	@sh -c 'set -eu; \
	for name in $(BENCH_TARGETS); do \
		bin=$$(find build/benchmarks -type f \( -name "$$name" -o -name "$$name.exe" \) | head -n 1); \
		if [ -z "$$bin" ]; then echo "Missing benchmark binary: $$name"; exit 1; fi; \
		out="build/benchmarks/results/$$name.json"; \
		echo "Running $$bin -> $$out"; \
		"$$bin" --benchmark_repetitions=5 --benchmark_min_time=0.1 --benchmark_out="$$out" --benchmark_out_format=json; \
	done'

# Quick smoke pass: run every Google Benchmark binary once with a
# 20ms min_time so the suite finishes in seconds.
bench-smoke: bench-cpp
	mkdir -p build/benchmarks/results
	@sh -c 'set -eu; \
	for name in $(BENCH_TARGETS); do \
		bin=$$(find build/benchmarks -type f \( -name "$$name" -o -name "$$name.exe" \) | head -n 1); \
		if [ -z "$$bin" ]; then echo "Missing benchmark binary: $$name"; exit 1; fi; \
		out="build/benchmarks/results/$$name.smoke.json"; \
		echo "Smoke $$bin -> $$out"; \
		"$$bin" --benchmark_filter=. --benchmark_min_time=0.02 --benchmark_repetitions=1 --benchmark_out="$$out" --benchmark_out_format=json; \
	done'

py-install:
	cd python && $(PYTHON) -m venv .venv && .venv/bin/pip install -e .

train-smoke:
	cd python && $(VENV_PY) -m train.train --config ../experiments/configs/phase4/smoke/phase4_mappo_smoke.yaml

format:
	cd python && $(VENV_RUFF) format .

lint:
	cd python && $(VENV_RUFF) check .

clean:
	rm -rf build
	rm -rf python/.pytest_cache python/.mypy_cache python/.ruff_cache
	find python -type d -name '__pycache__' -prune -exec rm -rf {} +

# Run the viewer bench against the typical scene fixture and check the
# result against the committed baseline (15% tolerance).
bench-viewer: build-cpp
	mkdir -p build/benchmarks/viewer
	./build/src/viewer/xushi2_viewer_bench \
	    --replay data/benchmarks/viewer/typical_match_scene.replay \
	    --mode render \
	    --warmup 120 --frames 600 \
	    --json-out build/benchmarks/viewer/result.json
	$(VENV_PY) python/scripts/check_viewer_bench.py \
	    --result build/benchmarks/viewer/result.json \
	    --baseline data/benchmarks/viewer/baseline.json \
	    --tolerance-pct 15
