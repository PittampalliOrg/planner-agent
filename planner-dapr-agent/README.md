# Planner Dapr Agent

An interactive Chainlit-based chat interface using Dapr Agents `DurableAgent` with Claude (Anthropic) for planning software engineering tasks.

## Features

- **Interactive Chat UI**: Web-based chat interface using Chainlit
- **Durable Execution**: Each chat message creates a workflow with state persistence
- **Persistent Memory**: Agent remembers conversation history across messages using Dapr state store
- **Planning Tools**: File reading, code search, directory listing, and task creation
- **Claude Integration**: Uses Anthropic's Claude models via custom `AnthropicChatClient`

## Architecture

```
User Chat → Chainlit UI → DurableAgent → Tool Calls → Persistent State → Response
                              ↓
                    AnthropicChatClient
                              ↓
                         Claude API
```

## Prerequisites

- Python 3.12+
- Dapr CLI and runtime
- Redis (for state store and pub/sub)
- Anthropic API key

## Local Development

### 1. Start Redis

```bash
docker run -d --name redis -p 6379:6379 redis:alpine
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run with Dapr

```bash
dapr run -f dapr.yaml
```

Or run directly:

```bash
dapr run --app-id planner-dapr-agent --app-port 8000 --resources-path ./resources -- chainlit run app.py -w
```

### 5. Access the UI

Open http://localhost:8000 in your browser.

## Docker Build

Build from the parent directory (required for `anthropic_llm.py`):

```bash
# From 100-improve-planning-agent/
docker build -t gitea.cnoe.localtest.me:8443/giteaadmin/planner-dapr-agent:latest -f planner-dapr-agent/Dockerfile .
docker push gitea.cnoe.localtest.me:8443/giteaadmin/planner-dapr-agent:latest
```

## Available Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read contents of a file in the workspace |
| `list_directory` | List files and directories with glob pattern support |
| `search_code` | Search for patterns in code files using regex |
| `create_task` | Create a planning task with subject, description, and dependencies |
| `list_tasks` | List all created planning tasks |
| `get_workspace_info` | Get information about the current workspace |

## Example Usage

1. **Explore the codebase**:
   > "What files are in this project?"

2. **Plan a feature**:
   > "Plan adding user authentication with JWT tokens"

3. **Review the plan**:
   > "Show me the current tasks"

## Dapr Components

- `statestore`: Redis state store for workflow execution state
- `memory-state`: Redis state store for conversation memory
- `message-pubsub`: Redis pub/sub for agent messaging
- `registry-state`: Redis state store for agent registry
