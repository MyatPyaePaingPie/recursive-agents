.PHONY: help setup test lint format type-check clean run-example benchmark install-dev

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Set up development environment
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  Windows: venv\\Scripts\\activate"
	@echo "  Linux/Mac: source venv/bin/activate"
	@echo "Then run: make install-dev"

install-dev: ## Install all dependencies (dev + prod)
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

install: ## Install production dependencies only
	pip install --upgrade pip
	pip install -r requirements.txt

test: ## Run all tests with coverage
	pytest tests/ -v --cov=src/rlm --cov-report=html --cov-report=term

test-fast: ## Run tests without coverage
	pytest tests/ -v

test-unit: ## Run only unit tests
	pytest tests/unit/ -v

test-integration: ## Run only integration tests
	pytest tests/integration/ -v

test-security: ## Run security-focused tests
	pytest tests/security/ -v

lint: ## Run linter (ruff)
	ruff check src/ tests/ examples/

lint-fix: ## Run linter and auto-fix issues
	ruff check --fix src/ tests/ examples/

format: ## Format code with black and isort
	black src/ tests/ examples/
	isort src/ tests/ examples/

format-check: ## Check if code is formatted
	black --check src/ tests/ examples/
	isort --check src/ tests/ examples/

type-check: ## Run type checker (mypy)
	mypy src/

check-all: lint format-check type-check test ## Run all checks (lint, format, type, test)

clean: ## Remove generated files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-example: ## Run basic example
	python examples/basic_rlm_demo.py

run-book-example: ## Run book summarization example
	python examples/book_summary.py

benchmark: ## Run performance benchmarks
	python benchmarks/run_benchmarks.py

dev: ## Start development mode with auto-reload (if API exists)
	uvicorn src.rlm.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build: ## Build Docker image
	docker build -t recursive-language-models .

docker-run: ## Run in Docker
	docker run -p 8000:8000 --env-file .env recursive-language-models

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

update-deps: ## Update dependencies to latest versions
	pip install --upgrade -r requirements.txt -r requirements-dev.txt

docs-serve: ## Serve documentation locally
	mkdocs serve

docs-build: ## Build documentation
	mkdocs build

