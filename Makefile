.PHONY: help build dev dev-down train api clean

SHELL := C:/PROGRA~1/Git/usr/bin/bash.exe

IMAGE_NAME := churn-radar
DOCKER_RUN := docker compose run --rm app

help:
	@echo "Churn Radar - ML Platform (Docker-based)"
	@echo ""
	@echo "Setup:"
	@echo "  build            Build Docker images"
	@echo "  dev              Start all services (builds first)"
	@echo "  dev-down         Stop all services"
	@echo "  logs             Show logs"
	@echo ""
	@echo "Data (runs in Docker):"
	@echo "  data             Auto: real data if exists, else demo"
	@echo "  data-demo        Generate synthetic demo data"
	@echo "  data-real        Process real data from data/raw/"
	@echo "  data-sample      Process real data (10% sample)"
	@echo ""
	@echo "Training (runs in Docker):"
	@echo "  train            Train model (requires data)"
	@echo "  train-demo       Generate demo data + train"
	@echo "  train-real       Process real data + train"
	@echo "  hpo              Run hyperparameter optimization"
	@echo ""
	@echo "API:"
	@echo "  api              Start API service only"
	@echo "  api-logs         Show API logs"
	@echo ""
	@echo "Development (runs in Docker):"
	@echo "  test             Run tests"
	@echo "  lint             Run linters"
	@echo "  shell            Open shell in container"
	@echo "  clean            Clean artifacts"

# ==================== BUILD ====================

build:
	docker compose build

# ==================== SERVICES ====================

dev: build
	docker compose up -d mlflow redis
	@echo "Waiting for MLflow and Redis..."
	@docker compose exec redis sh -c "for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do sleep 1; done" 2>/dev/null || timeout /t 15 /nobreak >NUL 2>&1 || true
	docker compose up -d api airflow
	@echo ""
	@echo "Services started:"
	@echo "  API:     http://localhost:8000"
	@echo "  Docs:    http://localhost:8000/docs"
	@echo "  MLflow:  http://localhost:5000"
	@echo "  Airflow: http://localhost:8080 (admin/admin)"
	@echo ""
	@echo "Run 'make train-demo' to train a model"

dev-down:
	docker compose down

logs:
	docker compose logs -f

api:
	docker compose up -d mlflow redis
	@docker compose exec redis sh -c "for i in 1 2 3 4 5 6 7 8 9 10; do sleep 1; done" 2>/dev/null || timeout /t 10 /nobreak >NUL 2>&1 || true
	docker compose up api

api-logs:
	docker compose logs -f api

# ==================== DATA (Docker) ====================

data: _ensure-services
	@echo "Auto mode: checking for real data..."
	$(DOCKER_RUN) python -m src.data.make_dataset --auto

data-demo: _ensure-services
	@echo "Generating synthetic demo data..."
	$(DOCKER_RUN) python -m src.data.make_dataset --demo --n-users 2000

data-real: _ensure-services
	@echo "Processing real data from data/raw/..."
	$(DOCKER_RUN) python -m src.data.make_dataset --real

data-sample: _ensure-services
	@echo "Processing real data (10% sample)..."
	$(DOCKER_RUN) python -m src.data.make_dataset --real --sample 0.1 --max-users 10000

# ==================== TRAINING (Docker) ====================

train: _ensure-services
	$(DOCKER_RUN) python -m src.models.train --config configs/config.yaml

train-demo: _ensure-services
	@echo "=== Generating demo data ==="
	$(DOCKER_RUN) python -m src.data.make_dataset --demo --n-users 2000
	@echo "=== Training model ==="
	$(DOCKER_RUN) python -m src.models.train --config configs/config.yaml
	@echo "=== Demo training complete! ==="

train-real: _ensure-services
	@echo "=== Processing real data ==="
	$(DOCKER_RUN) python -m src.data.make_dataset --real
	@echo "=== Training model ==="
	$(DOCKER_RUN) python -m src.models.train --config configs/config.yaml
	@echo "=== Real data training complete! ==="

hpo: _ensure-services
	$(DOCKER_RUN) python -m src.models.hpo --algorithm lightgbm --n-trials 50

# ==================== DEVELOPMENT (Docker) ====================

test: _ensure-services
	$(DOCKER_RUN) pytest tests/ -v --tb=short

test-cov: _ensure-services
	$(DOCKER_RUN) pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	$(DOCKER_RUN) ruff check src/ tests/
	$(DOCKER_RUN) mypy src/ --ignore-missing-imports

format:
	$(DOCKER_RUN) ruff check --fix src/ tests/
	$(DOCKER_RUN) black src/ tests/
	$(DOCKER_RUN) isort src/ tests/

shell: _ensure-services
	docker compose run --rm app bash

# ==================== HELPERS ====================

_ensure-services:
	@echo "Ensuring MLflow and Redis are running..."
	@docker compose up -d mlflow redis 2>/dev/null || true
	@echo "Waiting for services to be ready..."
	@docker compose exec redis sh -c "for i in 1 2 3 4 5 6 7 8 9 10; do sleep 1; done" 2>/dev/null || timeout /t 10 /nobreak >NUL 2>&1 || true
	@docker compose ps mlflow redis

# ==================== CLEANUP ====================

clean:
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf data/*.csv data/predictions.csv
	rm -rf models/*.pkl models/*.json
	rm -rf mlruns/ site/
	docker run --rm -v "$(CURDIR):/work" -w /work alpine sh -c 'find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; find . -type f -name "*.pyc" -delete 2>/dev/null' || true

clean-docker:
	docker compose down -v --rmi local
	docker system prune -f
