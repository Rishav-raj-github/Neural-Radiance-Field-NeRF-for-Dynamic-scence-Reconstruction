.PHONY: help install install-dev test lint format clean run docker-build docker-run

# Default target
help:
	@echo "Neural Radiance Field (NeRF) - Dynamic Scene Reconstruction"
	@echo "============================================================"
	@echo "Available targets:"
	@echo "  install       - Install production dependencies"
	@echo "  install-dev   - Install development dependencies"
	@echo "  test          - Run tests with coverage"
	@echo "  lint          - Run code linters (flake8)"
	@echo "  format        - Format code with black and isort"
	@echo "  clean         - Remove build artifacts and cache files"
	@echo "  run           - Run the main application"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run Docker container"

install:
	@echo "Installing production dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "Installing development dependencies..."
	pip install -e .[dev,full]
	pip install -r requirements.txt

test:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	@echo "Running linters..."
	flake8 src/ tests/ --max-line-length=100 --exclude=venv,env,.git,__pycache__

format:
	@echo "Formatting code..."
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile black

clean:
	@echo "Cleaning build artifacts and cache files..."
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete

run:
	@echo "Running NeRF application..."
	python -m src.main --config config/example_config.yaml

docker-build:
	@echo "Building Docker image..."
	docker build -t nerf-dynamic:latest .

docker-run:
	@echo "Running Docker container..."
	docker run --gpus all -it --rm -v $(PWD):/workspace nerf-dynamic:latest

# Edge device targets
jetson-setup:
	@echo "Setting up Jetson device..."
	sudo nvpmodel -m 0
	sudo jetson_clocks
	@echo "Jetson optimized for maximum performance"

# Development helpers
notebook:
	@echo "Starting Jupyter notebook..."
	jupyter notebook --ip=0.0.0.0 --no-browser

watch-logs:
	@echo "Watching logs..."
	tail -f logs/*.log
