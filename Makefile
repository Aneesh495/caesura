# Caesura Build, Test, and Benchmark Automation Makefile

.PHONY: all build test verify benchmark clean help

PYTHON ?= uv run python
PYTEST ?= uv run pytest

all: build

build:
	$(PYTHON) setup.py build_ext --inplace

test: build
	$(PYTEST) -v

verify: build
	$(PYTHON) -m caesura.cli.main verify

benchmark: build
	$(PYTHON) -m caesura.benchmarks.suite

quickstart: build
	$(PYTHON) examples/quickstart.py

streaming-demo: build
	$(PYTHON) examples/streaming_demo.py

train-demo: build
	$(PYTHON) examples/train_custom.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov
	find caesura -name "*.cpython-*.so" -delete
	find caesura -name "*.c" ! -name "arena.c" ! -name "murmur3.c" ! -name "robin_hood.c" ! -name "simd_utf8.c" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +

help:
	@echo "Caesura Development Targets:"
	@echo "  make build         - Compile Cython C-extensions in-place"
	@echo "  make test          - Run full pytest test suite"
	@echo "  make verify        - Run Caesura system verification pass"
	@echo "  make benchmark     - Run throughput benchmark suite"
	@echo "  make quickstart    - Run quickstart walkthrough example"
	@echo "  make streaming-demo- Run streaming detokenizer demo"
	@echo "  make train-demo    - Run custom BPE vocabulary trainer demo"
	@echo "  make clean         - Remove compiled C files and build caches"
