# MNEME v2.0 Makefile
# Motor de Memoria Neural Mórfica (Safetensors + Locks Granulares)

.PHONY: help install install-dev test lint format clean build publish docs

help: ## Show this help message
	@echo "MNEME - Motor de Memoria Neural Mórfica"
	@echo "======================================"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install MNEME in production mode
	pip install -e .

install-dev: ## Install MNEME in development mode with all dependencies
	pip install -e .[dev,all]

test: ## Run tests
	pytest tests/ -v --cov=src/mneme --cov-report=html --cov-report=term

test-gpu: ## Run GPU tests
	pytest tests/ -v -m gpu

test-security: ## Run security tests
	pytest tests/ -v -m security

lint: ## Run linting
	flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	mypy src/mneme/ --ignore-missing-imports

format: ## Format code
	black src/ tests/ examples/
	isort src/ tests/ examples/

format-check: ## Check code formatting
	black --check src/ tests/ examples/
	isort --check-only src/ tests/ examples/

security: ## Run security checks
	bandit -r src/ -f json -o bandit-report.json
	safety check

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf bandit-report.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: ## Build package
	python -m build

publish: ## Publish to PyPI (requires authentication)
	twine upload dist/*

docs: ## Generate documentation
	@echo "Documentation is available in the docs/ directory"
	@echo "Main documentation files:"
	@ls -la docs/*.md

setup-pre-commit: ## Setup pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

run-examples: ## Run example scripts
	python examples/example_mneme.py
	python examples/example_advanced_features.py
	python examples/example_advanced_serialization.py
	python examples/example_advanced_encryption.py
	python examples/example_context_deduplication.py

benchmark: ## Run performance benchmarks
	python -c "from examples.example_mneme import main; main()"

check-all: lint test security ## Run all checks

ci: clean install-dev check-all ## Run CI pipeline locally

dev-setup: install-dev setup-pre-commit ## Setup development environment

# Development shortcuts
dev: dev-setup ## Alias for dev-setup
