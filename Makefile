# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Uni_Vision — Developer Makefile
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON   := python
APP_IMG  := uni-vision:latest

# ── Help ──────────────────────────────────────────────────────────
.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Development ───────────────────────────────────────────────────
.PHONY: install
install: ## Install editable package with dev + inference extras
	$(PYTHON) -m pip install -e ".[dev,inference]"

.PHONY: install-dev
install-dev: ## Install editable package with dev extras only (no CUDA)
	$(PYTHON) -m pip install -e ".[dev]"

# ── Quality ───────────────────────────────────────────────────────
.PHONY: lint
lint: ## Run ruff linter
	$(PYTHON) -m ruff check src/ tests/

.PHONY: lint-fix
lint-fix: ## Auto-fix lint issues
	$(PYTHON) -m ruff check --fix src/ tests/

.PHONY: format
format: ## Format code with ruff
	$(PYTHON) -m ruff format src/ tests/

.PHONY: typecheck
typecheck: ## Run mypy type checking
	$(PYTHON) -m mypy src/uni_vision/

# ── Testing ───────────────────────────────────────────────────────
.PHONY: test
test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short

.PHONY: test-quick
test-quick: ## Run tests without verbose output
	$(PYTHON) -m pytest tests/ -q

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ --cov=uni_vision --cov-report=term-missing --cov-report=html

# ── Docker ────────────────────────────────────────────────────────
.PHONY: build
build: ## Build the Docker image
	docker build -t $(APP_IMG) .

.PHONY: up
up: ## Start the full stack (docker compose)
	docker compose up -d

.PHONY: up-build
up-build: ## Rebuild and start the full stack
	docker compose up -d --build

.PHONY: down
down: ## Stop the stack (preserve volumes)
	docker compose down

.PHONY: down-clean
down-clean: ## Stop the stack and delete all volumes
	docker compose down -v

.PHONY: logs
logs: ## Tail application logs
	docker compose logs -f app

.PHONY: logs-all
logs-all: ## Tail all service logs
	docker compose logs -f

# ── Ollama Model Management ──────────────────────────────────────
.PHONY: ollama-init
ollama-init: ## Pull Gemma 4 E2B base model and create OCR + Adjudicator variants
	./scripts/init-ollama.sh

.PHONY: ollama-status
ollama-status: ## Show loaded Ollama models
	curl -s http://localhost:11434/api/tags | python -m json.tool

# ── API ───────────────────────────────────────────────────────────
.PHONY: serve
serve: ## Start the API server locally (no Docker)
	$(PYTHON) -m uni_vision.api

.PHONY: pipeline
pipeline: ## Start the full pipeline locally (no Docker)
	$(PYTHON) -m uni_vision.orchestrator.pipeline

# ── Database Migrations ──────────────────────────────────────────
.PHONY: db-upgrade
db-upgrade: ## Apply all pending migrations
	$(PYTHON) -m alembic upgrade head

.PHONY: db-downgrade
db-downgrade: ## Revert the last migration
	$(PYTHON) -m alembic downgrade -1

.PHONY: db-history
db-history: ## Show migration history
	$(PYTHON) -m alembic history --verbose

.PHONY: db-current
db-current: ## Show current migration revision
	$(PYTHON) -m alembic current

.PHONY: db-migrate
db-migrate: ## Generate a new migration (usage: make db-migrate MSG="add foo column")
	$(PYTHON) -m alembic revision -m "$(MSG)"

# ── Cleanup ───────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml dist/ build/ *.egg-info
