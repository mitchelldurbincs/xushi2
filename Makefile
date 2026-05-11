.PHONY: build-cpp test-cpp py-install train-smoke format lint clean

build-cpp:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j

test-cpp: build-cpp
	ctest --test-dir build --output-on-failure

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
