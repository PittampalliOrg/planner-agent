# planner-sdk-agent

Minimal Claude Agent SDK wrapper with native Claude Code functionality and task dependency management.

## Setup

```bash
npm install
```

## Usage

```bash
# Default mode
npm start "Your prompt here"

# Plan mode
PERMISSION_MODE=plan npm start "Plan the implementation"

# Resume session
SESSION_ID=<id> npm start "Continue"

# Custom working directory
CWD=/path/to/project npm start "Analyze this project"
```

## Environment Variables

| Variable | Values | Default |
|----------|--------|---------|
| `ANTHROPIC_API_KEY` | API key | Required |
| `PERMISSION_MODE` | `default`, `plan`, `bypassPermissions`, `acceptEdits` | `default` |
| `SESSION_ID` | Session ID to resume | - |
| `CWD` | Working directory | Current dir |

## Features

All native Claude Code tools (21 total), including:
- **Task management**: TaskCreate, TaskUpdate, TaskList, TaskGet
- **File ops**: Read, Write, Edit, Glob, Grep
- **Execution**: Bash, Task (subagents)
- **Planning**: EnterPlanMode, ExitPlanMode, AskUserQuestion
