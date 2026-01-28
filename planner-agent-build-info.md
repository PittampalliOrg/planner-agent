# Planner Agent - Docker Build Info

## 2 Docker Images, 3 K8s Deployments

| Image | Dockerfile | Used By |
|-------|-----------|---------|
| `gitea.cnoe.localtest.me:8443/giteaadmin/planner-orchestrator:latest` | `planner-orchestrator/Dockerfile` | `Deployment-planner-agent` (orchestrator) |
| `gitea.cnoe.localtest.me:8443/giteaadmin/planner-sdk-agent:latest` | `planner-sdk-agent/Dockerfile` | `Deployment-planner-agent-plan` AND `Deployment-planner-agent-exec` |

The SDK agent image is shared by both plan and exec deployments -- same server code, different endpoints (`/plan` vs `/execute`).

## Build & Push Commands

```bash
# From 100-improve-planning-agent/

# 1. Orchestrator (Python/FastAPI)
docker build -t gitea.cnoe.localtest.me:8443/giteaadmin/planner-orchestrator:latest planner-orchestrator/
docker push gitea.cnoe.localtest.me:8443/giteaadmin/planner-orchestrator:latest

# 2. SDK Agent (Node.js/Express + Claude Agent SDK)
docker build -t gitea.cnoe.localtest.me:8443/giteaadmin/planner-sdk-agent:latest planner-sdk-agent/
docker push gitea.cnoe.localtest.me:8443/giteaadmin/planner-sdk-agent:latest
```

## Restart Deployments After Push

```bash
kubectl rollout restart deployment/planner-orchestrator deployment/planner-agent-plan deployment/planner-agent-exec -n planner-agent
```

## Source Repo

Branch: `100-improve-planning-agent`
Repo: https://github.com/PittampalliOrg/planner-agent
