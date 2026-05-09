# Xushi2 — deterministic build & test workflow
#
# All C++ build artifacts are directed into BUILD_DIR (build/ by default).
# No ad-hoc directories are created in the source tree.

BUILD_DIR        := build
PYTHON_DIR       := python
CMAKE_BUILD_TYPE ?= Release
NPROC            := $(shell nproc 2>/dev/null || echo 4)

# --- Python targets ----------------------------------------------------------

.PHONY: reward-test
reward-test:
	@echo "=== Running reward tests (deterministic build dir: $(BUILD_DIR)) ==="
	cd $(PYTHON_DIR) && python -m pytest tests/test_reward.py -v --tb=short \
		-k "not test_distance_shaping_produces_nonzero_reward_on_real_env"

.PHONY: reward-test-full
reward-test-full:
	@echo "=== Running all reward tests (deterministic build dir: $(BUILD_DIR)) ==="
	cd $(PYTHON_DIR) && python -m pytest tests/test_reward.py -v --tb=short

.PHONY: python-test
python-test:
	@echo "=== Running all Python tests ==="
	cd $(PYTHON_DIR) && python -m pytest tests/ -v --tb=short

# --- C++ / CMake targets -----------------------------------------------------

.PHONY: build
build: $(BUILD_DIR)/CMakeCache.txt
	@echo "=== Building C++ code in $(BUILD_DIR) ==="
	cmake --build $(BUILD_DIR) --parallel $(NPROC)

$(BUILD_DIR)/CMakeCache.txt:
	@echo "=== Configuring CMake in $(BUILD_DIR) ==="
	cmake -B $(BUILD_DIR) -S . \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DXUSHI2_BUILD_PYTHON_MODULE=ON \
		-DXUSHI2_BUILD_TESTS=ON \
		-DXUSHI2_BUILD_VIEWER=OFF

.PHONY: cpp-test
cpp-test: build
	@echo "=== Running C++ tests in $(BUILD_DIR) ==="
	ctest --test-dir $(BUILD_DIR) --output-on-failure

# --- Combined targets --------------------------------------------------------

.PHONY: test
test: reward-test cpp-test

# --- Maintenance targets -----------------------------------------------------

.PHONY: clean
clean:
	@echo "=== Removing $(BUILD_DIR) ==="
	rm -rf $(BUILD_DIR)

.PHONY: format-check
format-check:
	@echo "=== C++ format check ==="
	find src tests -name '*.cpp' -o -name '*.h' | xargs clang-format --dry-run --Werror
	@echo "=== Python format check ==="
	cd $(PYTHON_DIR) && ruff check .
