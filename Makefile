.PHONY: build-cpp test-cpp bench-cpp run-bench bench-smoke py-install train-smoke format lint clean

build-cpp:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j

test-cpp: build-cpp
	ctest --test-dir build --output-on-failure

bench-cpp:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DXUSHI2_BUILD_BENCHMARKS=ON
	cmake --build build -j --target benchmarks

run-bench: bench-cpp
	mkdir -p build/benchmarks/results
	sh -c 'set -eu; \
	bins=$$(find build -maxdepth 4 -type f \( -path "*/benchmarks/*" -o -name "*bench*" \) -perm -111 | sort); \
	if [ -z "$$bins" ]; then echo "No benchmark binaries found under build/."; exit 1; fi; \
	for bin in $$bins; do \
		name=$$(basename "$$bin"); \
		out="build/benchmarks/results/$${name}.json"; \
		echo "Running $$bin -> $$out"; \
		"$$bin" --benchmark_repetitions=5 --benchmark_min_time=0.1 --benchmark_out="$$out" --benchmark_out_format=json; \
	done'

bench-smoke: bench-cpp
	mkdir -p build/benchmarks/results
	sh -c 'set -eu; \
	first=$$(find build -maxdepth 4 -type f \( -path "*/benchmarks/*" -o -name "*bench*" \) -perm -111 | sort | head -n 1); \
	if [ -z "$$first" ]; then echo "No benchmark binaries found under build/."; exit 1; fi; \
	name=$$(basename "$$first"); \
	out="build/benchmarks/results/$${name}.smoke.json"; \
	echo "Running smoke benchmark $$first -> $$out"; \
	"$$first" --benchmark_filter='.' --benchmark_min_time=0.02 --benchmark_repetitions=1 --benchmark_out="$$out" --benchmark_out_format=json'

py-install:
	cd python && python -m venv .venv && . .venv/bin/activate && pip install -e .

train-smoke:
	cd python && python -m train.train --config ../experiments/configs/phase4/smoke/phase4_mappo_smoke.yaml

format:
	cd python && ruff format .

lint:
	cd python && ruff check .

clean:
	rm -rf build
	rm -rf python/.pytest_cache python/.mypy_cache python/.ruff_cache
	find python -type d -name '__pycache__' -prune -exec rm -rf {} +
