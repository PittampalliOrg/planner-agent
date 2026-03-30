# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-25

### Added

- Planner Agent core implementation (`planner_agent.py`) replicating Claude Code's plan mode using the Claude Agent SDK
- `PlanManager` for creating, persisting, and approving implementation plans (`plan_manager.py`)
- `TaskManager` for task tracking with dependency support and status lifecycle (`task_manager.py`)
- Dapr Workflow integration with execution-only mode (`workflow_service.py`)
- Server-Sent Events streaming support for real-time workflow updates (`streaming.py`)
- FastAPI HTTP entry point (`main.py`) with SDK error handling
- `Dockerfile` and `.dockerignore` for containerised deployments
- Kubernetes manifests in `k8s/` (namespace, secret, configmap, PVC, deployment, job, kustomization)
- `devspace.yaml` for DevSpace-based development workflows
- `requirements.txt` listing all Python dependencies
- `.env.example` documenting required environment variables
- `README.md` with full usage, API reference, and Kubernetes deployment guide
