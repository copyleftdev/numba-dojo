# =============================================================================
# Numba Dojo - Performance Benchmark Suite
# =============================================================================
# A battle royale between Python, Rust, and CUDA implementations
#
# Usage:
#   make all          - Build everything and run benchmarks
#   make build        - Build all benchmarks
#   make run          - Run all benchmarks
#   make visualize    - Generate charts from results
#   make clean        - Clean build artifacts
# =============================================================================

SHELL := /bin/bash
.PHONY: all build build-rust build-cuda build-python run visualize clean help venv

# Configuration
PYTHON := python3
VENV := venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
RESULTS_DIR := results
ARRAY_SIZE := 20000000

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m

# =============================================================================
# Main Targets
# =============================================================================

all: build run visualize
	@echo -e "$(GREEN)✓ All done! Check $(RESULTS_DIR)/ for results$(NC)"

help:
	@echo "Numba Dojo - Performance Benchmark Suite"
	@echo ""
	@echo "Targets:"
	@echo "  make all        - Build, run benchmarks, and generate visualizations"
	@echo "  make build      - Build all benchmark binaries"
	@echo "  make run        - Run all benchmarks (outputs to results/)"
	@echo "  make visualize  - Generate charts from benchmark results"
	@echo "  make clean      - Remove build artifacts and results"
	@echo "  make venv       - Create Python virtual environment"
	@echo ""
	@echo "Individual builds:"
	@echo "  make build-rust - Build Rust benchmark"
	@echo "  make build-cuda - Build CUDA C++ benchmark"
	@echo ""
	@echo "Configuration:"
	@echo "  ARRAY_SIZE=$(ARRAY_SIZE) (override with make ARRAY_SIZE=N)"

# =============================================================================
# Virtual Environment
# =============================================================================

venv: $(VENV)/bin/activate

$(VENV)/bin/activate:
	@echo -e "$(CYAN)Creating Python virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install numpy numba matplotlib
	@echo -e "$(GREEN)✓ Virtual environment created$(NC)"

# =============================================================================
# Build Targets
# =============================================================================

build: build-rust build-cuda venv
	@echo -e "$(GREEN)✓ All builds complete$(NC)"

build-rust:
	@echo -e "$(CYAN)Building Rust benchmark...$(NC)"
	cd benchmarks/rust && cargo build --release
	@echo -e "$(GREEN)✓ Rust build complete$(NC)"

build-cuda:
	@echo -e "$(CYAN)Building CUDA C++ benchmark...$(NC)"
	@if command -v nvcc &> /dev/null; then \
		cd benchmarks/cuda && nvcc -O3 -arch=native -use_fast_math \
			--expt-relaxed-constexpr -o benchmark benchmark.cu; \
		echo -e "$(GREEN)✓ CUDA build complete$(NC)"; \
	else \
		echo -e "$(YELLOW)⚠ nvcc not found, skipping CUDA build$(NC)"; \
	fi

# =============================================================================
# Run Benchmarks
# =============================================================================

run: venv $(RESULTS_DIR)
	@echo -e "$(CYAN)Running benchmarks...$(NC)"
	$(VENV_PYTHON) scripts/run_benchmarks.py --results-dir $(RESULTS_DIR) --size $(ARRAY_SIZE)
	@echo -e "$(GREEN)✓ Benchmarks complete$(NC)"

$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

# =============================================================================
# Visualization
# =============================================================================

visualize: venv
	@echo -e "$(CYAN)Generating visualizations...$(NC)"
	$(VENV_PYTHON) scripts/visualize.py --results-dir $(RESULTS_DIR) --output-dir $(RESULTS_DIR)
	@echo -e "$(GREEN)✓ Visualizations complete$(NC)"
	@echo -e "$(CYAN)Results available in $(RESULTS_DIR)/:$(NC)"
	@ls -la $(RESULTS_DIR)/*.png $(RESULTS_DIR)/*.md 2>/dev/null || true

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo -e "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf $(RESULTS_DIR)
	rm -rf benchmarks/rust/target
	rm -f benchmarks/cuda/benchmark
	rm -rf __pycache__ benchmarks/python/__pycache__
	rm -rf .pytest_cache
	@echo -e "$(GREEN)✓ Clean complete$(NC)"

clean-all: clean
	rm -rf $(VENV)
	@echo -e "$(GREEN)✓ Full clean complete$(NC)"

# =============================================================================
# Development Helpers
# =============================================================================

check-deps:
	@echo "Checking dependencies..."
	@echo -n "  Python: " && $(PYTHON) --version
	@echo -n "  Rust: " && (cargo --version || echo "not found")
	@echo -n "  CUDA: " && (nvcc --version | head -1 || echo "not found")
	@echo -n "  NumPy: " && ($(VENV_PYTHON) -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "not installed")
	@echo -n "  Numba: " && ($(VENV_PYTHON) -c "import numba; print(numba.__version__)" 2>/dev/null || echo "not installed")

lint-rust:
	cd benchmarks/rust && cargo clippy -- -W clippy::all

fmt-rust:
	cd benchmarks/rust && cargo fmt
