# AI Lab Backend — Development & Test Targets
# Usage: make <target>
#
# Targets:
#   test            Run unit tests only (fast, no database)
#   test-integration Run integration tests against SQLite
#   test-postgres   Run ALL tests against a real PostgreSQL container
#   test-all        Run unit + integration (SQLite) + postgres
#   lint            Run ruff linter
#   format          Run ruff formatter
#   type-check      Run mypy type checker
#   export-openapi  Export OpenAPI schema (json + yaml) to project root

SHELL := /usr/bin/bash
PYTHON := python3
PYTEST := python3 -m pytest
PROJECT_DIR := $(shell pwd)
POSTGRES_IMAGE := postgres:16-alpine
POSTGRES_CONTAINER := stable-diffusion-lab-test-db
POSTGRES_PORT := 54321
POSTGRES_USER := ailab
POSTGRES_PASSWORD := ailab_test
POSTGRES_DB := ailab_test
DATABASE_URL := postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT}/${POSTGRES_DB}

.PHONY: test test-integration test-postgres test-all lint format type-check stop-postgres export-openapi clean

## Default target
all: test-all

## Run unit tests only (fast, no database needed)
test:
	@echo "▶ Running unit tests..."
	$(PYTEST) tests/unit/ -v --tb=short
	@echo "✓ Unit tests passed"

## Run integration tests against SQLite
test-integration:
	@echo "▶ Running integration tests (SQLite)..."
	$(PYTEST) tests/integration/ -v --tb=short --cov=app --cov-report=term-missing
	@echo "✓ Integration tests passed"

## Start a temporary PostgreSQL container for testing
start-postgres:
	@echo "▶ Starting PostgreSQL container..."
	@if docker ps -a --format '{{.Names}}' | grep -q '^${POSTGRES_CONTAINER}$$'; then \
		echo "Container ${POSTGRES_CONTAINER} already exists, reusing..."; \
	else \
		docker run -d \
			--name ${POSTGRES_CONTAINER} \
			-e POSTGRES_USER=${POSTGRES_USER} \
			-e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} \
			-e POSTGRES_DB=${POSTGRES_DB} \
			-p ${POSTGRES_PORT}:5432 \
			${POSTGRES_IMAGE}; \
	fi
	@echo "Waiting for PostgreSQL to be ready..."
	@for i in $$(seq 1 30); do \
		if docker exec ${POSTGRES_CONTAINER} pg_isready -U ${POSTGRES_USER} > /dev/null 2>&1; then \
			echo "PostgreSQL is ready!"; \
			break; \
		fi; \
		if [ $$i -eq 30 ]; then \
			echo "ERROR: PostgreSQL failed to start within 30 seconds"; \
			docker stop ${POSTGRES_CONTAINER} > /dev/null 2>&1; \
			exit 1; \
		fi; \
		sleep 1; \
	done

## Stop and remove the temporary PostgreSQL container
stop-postgres:
	@echo "▶ Stopping PostgreSQL container..."
	-docker stop ${POSTGRES_CONTAINER} > /dev/null 2>&1
	-docker rm ${POSTGRES_CONTAINER} > /dev/null 2>&1
	@echo "PostgreSQL container stopped and removed"

## Run ALL tests against a real PostgreSQL container
test-postgres: start-postgres
	@echo "▶ Running all tests against PostgreSQL..."
	@DATABASE_URL=${DATABASE_URL} $(PYTEST) tests/ -v --tb=short --cov=app --cov-report=term-missing \
		--override-ini="asyncio_mode=auto" \
		|| { echo "Tests failed!"; docker stop ${POSTGRES_CONTAINER} > /dev/null 2>&1; docker rm ${POSTGRES_CONTAINER} > /dev/null 2>&1; exit 1; }
	@echo "✓ PostgreSQL tests passed"
	@$(MAKE) stop-postgres

## Run everything: unit + integration + postgres
test-all: test test-postgres
	@echo "✓ All tests passed (unit + integration + PostgreSQL)"

## Run ruff linter
lint:
	@echo "▶ Running ruff linter..."
	ruff check app/ tests/

## Run ruff formatter
format:
	@echo "▶ Running ruff formatter..."
	ruff format app/ tests/

## Export OpenAPI schema (json + yaml) to project root
export-openapi:
	@echo "▶ Exporting OpenAPI schema..."
	$(PYTHON) scripts/export_openapi.py
	@echo "✓ openapi.json and openapi.yaml written to project root"

## Clean test artifacts
clean: stop-postgres
	@echo "▶ Cleaning test artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage*" -delete 2>/dev/null || true
	find . -type f -name "coverage.xml" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	@echo "✓ Cleaned"