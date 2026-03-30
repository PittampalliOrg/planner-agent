# Makefile for planner-agent
# Usage: make <target>

PYTHON ?= python3
PIP    ?= pip

.PHONY: help install test lint format run

help:  ## Show this help message
	@echo "Available targets:"
	@echo "  install   Install all dependencies"
	@echo "  test      Run tests with pytest"
	@echo "  lint      Lint source files with ruff"
	@echo "  format    Format source files with ruff"
	@echo "  run       Start the FastAPI workflow service"

install:  ## Install dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install ruff pytest

test:  ## Run tests
	$(PYTHON) -m pytest tests/ -v

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check .

format:  ## Format with ruff
	$(PYTHON) -m ruff format .

run:  ## Run the FastAPI workflow service
	uvicorn workflow_service:app --host 0.0.0.0 --port 8080 --reload
