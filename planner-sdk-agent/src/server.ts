import express from "express";
import { query } from "@anthropic-ai/claude-agent-sdk";
import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";

process.env.CLAUDE_CODE_ENABLE_TASKS = "true";

// ---------------------------------------------------------------------------
// Logger utility
// ---------------------------------------------------------------------------
const LOG_LEVEL = (process.env.LOG_LEVEL || "debug").toLowerCase();
const LEVELS: Record<string, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

function shouldLog(level: string): boolean {
  return (LEVELS[level] ?? 1) >= (LEVELS[LOG_LEVEL] ?? 0);
}

function log(level: string, message: string, data?: Record<string, unknown>) {
  if (!shouldLog(level)) return;
  const ts = new Date().toISOString();
  const tag = level.toUpperCase();
  const prefix = `[${ts}] [${tag}]`;
  if (data) {
    console.log(`${prefix} ${message}`, JSON.stringify(data, null, 2));
  } else {
    console.log(`${prefix} ${message}`);
  }
}

// ---------------------------------------------------------------------------
// Dapr pub/sub event publishing
// ---------------------------------------------------------------------------
const DAPR_HTTP_PORT = process.env.DAPR_HTTP_PORT || "3500";
const PUBSUB_NAME = process.env.PUBSUB_NAME || "pubsub";
const PUBSUB_TOPIC = process.env.PUBSUB_TOPIC || "workflow.stream";

/**
 * Publish a workflow stream event to Dapr pub/sub.
 * Fire-and-forget: failures are logged but don't block the agent.
 */
async function publishEvent(
  workflowId: string,
  eventType: string,
  data: Record<string, unknown>,
  agentId: string = "claude-planner",
): Promise<void> {
  const event = {
    id: `agent-${workflowId}-${crypto.randomBytes(4).toString("hex")}`,
    type: eventType,
    workflowId,
    agentId,
    data,
    timestamp: new Date().toISOString(),
  };

  try {
    const url = `http://localhost:${DAPR_HTTP_PORT}/v1.0/publish/${PUBSUB_NAME}/${PUBSUB_TOPIC}`;
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(event),
    });
    if (!resp.ok) {
      log("warn", `Failed to publish ${eventType} event: ${resp.status}`);
    }
  } catch (err: any) {
    log("warn", `Failed to publish ${eventType} event: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Express app
// ---------------------------------------------------------------------------
const app = express();
app.use(express.json({ limit: "10mb" }));

// Request logging middleware
app.use((req, _res, next) => {
  log("info", `${req.method} ${req.path}`, {
    contentLength: req.headers["content-length"],
    contentType: req.headers["content-type"],
    bodyKeys: req.body ? Object.keys(req.body) : [],
  });
  next();
});

const TASKS_DIR = path.join(
  process.env.HOME || "/home/planner",
  ".claude",
  "tasks"
);

log("info", "Server starting", {
  TASKS_DIR,
  HOME: process.env.HOME || "(unset)",
  CLAUDE_CODE_ENABLE_TASKS: process.env.CLAUDE_CODE_ENABLE_TASKS || "(unset)",
  LOG_LEVEL,
  NODE_ENV: process.env.NODE_ENV || "(unset)",
});

// POST /plan - Run the agent in plan mode, return created tasks
app.post("/plan", async (req, res) => {
  const { prompt, cwd, workflow_id } = req.body;
  const startTime = Date.now();

  log("info", "=== PLAN REQUEST START ===", {
    prompt: prompt?.substring(0, 200),
    cwd,
    workflow_id,
  });

  try {
    // Check tasks dir state before
    log("debug", "Tasks dir state BEFORE planning", {
      exists: fs.existsSync(TASKS_DIR),
      contents: fs.existsSync(TASKS_DIR)
        ? fs.readdirSync(TASKS_DIR, { withFileTypes: true }).map((d) => ({
            name: d.name,
            isDir: d.isDirectory(),
          }))
        : [],
    });

    const output: string[] = [];
    let messageCount = 0;
    let toolUseCount = 0;
    // Buffer text chunks to publish as TextUIPart events in batches
    let textBuffer = "";
    // Track pending tool calls to include required fields in results
    const pendingToolCalls = new Map<string, { toolName: string; input: unknown }>();

    for await (const msg of query({
      prompt,
      options: {
        systemPrompt: { type: "preset", preset: "claude_code" },
        tools: { type: "preset", preset: "claude_code" },
        settingSources: ["project"],
        permissionMode: "bypassPermissions" as any,
        cwd: cwd || process.cwd(),
      },
    })) {
      messageCount++;
      if (msg.type === "assistant") {
        for (const b of msg.message.content) {
          if (b.type === "thinking") {
            // Extended thinking blocks - publish as ReasoningUIPart
            const thinkingBlock = b as { type: "thinking"; thinking: string };
            if (workflow_id && thinkingBlock.thinking) {
              // Flush any pending text first to maintain order
              if (textBuffer.length > 0) {
                publishEvent(workflow_id, "part", {
                  type: "text",
                  text: textBuffer,
                  state: "done",
                });
                textBuffer = "";
              }
              publishEvent(workflow_id, "part", {
                type: "reasoning",
                text: thinkingBlock.thinking,
                state: "done",
              });
              log("debug", "Published reasoning part", {
                length: thinkingBlock.thinking.length,
              });
            }
          } else if (b.type === "text") {
            output.push(b.text);
            textBuffer += b.text;
            // Publish accumulated text as TextUIPart
            if (workflow_id && textBuffer.length > 0) {
              publishEvent(workflow_id, "part", {
                type: "text",
                text: textBuffer,
                state: "done",
              });
              textBuffer = "";
            }
          } else if (b.type === "tool_use") {
            // Flush any pending text before tool call to maintain order
            if (workflow_id && textBuffer.length > 0) {
              publishEvent(workflow_id, "part", {
                type: "text",
                text: textBuffer,
                state: "done",
              });
              textBuffer = "";
            }
            toolUseCount++;
            log("debug", "Tool use: " + b.name, {
              toolId: b.id,
              inputKeys: b.input
                ? Object.keys(b.input as Record<string, unknown>)
                : [],
            });
            // Track for correlation with result
            pendingToolCalls.set(b.id, { toolName: b.name, input: b.input });
            // Publish as DynamicToolUIPart with input-available state
            if (workflow_id) {
              publishEvent(workflow_id, "part", {
                type: "dynamic-tool",
                toolCallId: b.id,
                toolName: b.name,
                state: "input-available",
                input: b.input,
              });
              // Emit semantic events for plan/task tools
              if (b.name === "EnterPlanMode") {
                publishEvent(workflow_id, "plan_created", {
                  toolCallId: b.id,
                });
              }
            }
          }
        }
      } else if (msg.type === "user" && msg.message?.content) {
        // Handle tool results - they come back in user messages
        const content = msg.message.content;
        if (Array.isArray(content)) {
          for (const block of content) {
            if (block.type === "tool_result") {
              const toolResult = block as { type: "tool_result"; tool_use_id: string; content?: unknown; is_error?: boolean };
              log("debug", "Tool result received", {
                toolUseId: toolResult.tool_use_id,
                isError: toolResult.is_error,
              });
              // Get the pending tool call info for required fields
              const pending = pendingToolCalls.get(toolResult.tool_use_id);
              // Publish as DynamicToolUIPart with output state
              if (workflow_id) {
                const resultContent = typeof toolResult.content === "string"
                  ? toolResult.content
                  : JSON.stringify(toolResult.content);
                if (toolResult.is_error) {
                  publishEvent(workflow_id, "part", {
                    type: "dynamic-tool",
                    toolCallId: toolResult.tool_use_id,
                    toolName: pending?.toolName || "unknown",
                    state: "output-error",
                    input: pending?.input,
                    errorText: resultContent,
                  });
                } else {
                  publishEvent(workflow_id, "part", {
                    type: "dynamic-tool",
                    toolCallId: toolResult.tool_use_id,
                    toolName: pending?.toolName || "unknown",
                    state: "output-available",
                    input: pending?.input,
                    output: resultContent,
                  });
                  // Emit semantic events for task/plan tool results
                  if (pending?.toolName === "TaskCreate") {
                    // Parse the task data from the result or input
                    try {
                      const taskInput = pending.input as Record<string, unknown>;
                      const taskData = {
                        id: (JSON.parse(resultContent) as any)?.id || String(Date.now()),
                        subject: taskInput?.subject as string || "",
                        description: taskInput?.description as string || "",
                        activeForm: taskInput?.activeForm as string,
                        status: "pending" as const,
                        blocks: taskInput?.blocks as string[] | undefined,
                        blockedBy: taskInput?.blockedBy as string[] | undefined,
                      };
                      publishEvent(workflow_id, "task_created", {
                        task: taskData,
                        toolCallId: toolResult.tool_use_id,
                      });
                    } catch {
                      // Fallback: emit with minimal data
                      publishEvent(workflow_id, "task_created", {
                        task: pending.input,
                        toolCallId: toolResult.tool_use_id,
                      });
                    }
                  } else if (pending?.toolName === "TaskUpdate") {
                    const updateInput = pending.input as Record<string, unknown>;
                    publishEvent(workflow_id, "task_updated", {
                      taskId: updateInput?.taskId as string,
                      taskStatus: updateInput?.status as string,
                      toolCallId: toolResult.tool_use_id,
                    });
                  } else if (pending?.toolName === "ExitPlanMode") {
                    publishEvent(workflow_id, "plan_complete", {
                      toolCallId: toolResult.tool_use_id,
                    });
                  }
                }
                if (pending) {
                  pendingToolCalls.delete(toolResult.tool_use_id);
                }
              }
            }
          }
        }
      } else if (msg.type === "result") {
        log("debug", "SDK query result received", {
          messageType: msg.type,
        });
      }
    }

    // Flush any remaining text
    if (workflow_id && textBuffer.length > 0) {
      publishEvent(workflow_id, "part", {
        type: "text",
        text: textBuffer,
        state: "done",
      });
    }

    log("info", "SDK query completed", {
      messageCount,
      toolUseCount,
      outputLength: output.join("\n").length,
      elapsedMs: Date.now() - startTime,
    });

    // Check tasks dir state after
    log("debug", "Tasks dir state AFTER planning", {
      exists: fs.existsSync(TASKS_DIR),
      contents: fs.existsSync(TASKS_DIR)
        ? fs.readdirSync(TASKS_DIR, { withFileTypes: true }).map((d) => ({
            name: d.name,
            isDir: d.isDirectory(),
          }))
        : [],
    });

    const tasks = readTasks();
    log("info", "Read " + tasks.length + " tasks from filesystem", {
      taskSummaries: tasks.map((t: any) => ({
        id: t.id,
        subject: t.subject,
        status: t.status,
        blocks: t.blocks,
        blockedBy: t.blockedBy,
      })),
    });

    clearTasks();

    const elapsed = Date.now() - startTime;
    log("info", "=== PLAN REQUEST COMPLETE === (" + elapsed + "ms)", {
      success: true,
      taskCount: tasks.length,
      outputLength: output.join("\n").length,
    });

    res.json({ success: true, tasks, output: output.join("\n") });
  } catch (error: any) {
    const elapsed = Date.now() - startTime;
    log("error", "=== PLAN REQUEST FAILED === (" + elapsed + "ms)", {
      error: error.message,
      stack: error.stack,
    });
    res.status(500).json({ success: false, error: error.message });
  }
});

// POST /execute - Restore tasks, run the agent in execution mode
app.post("/execute", async (req, res) => {
  const { prompt, cwd, tasks, workflow_id } = req.body;
  const startTime = Date.now();

  log("info", "=== EXECUTE REQUEST START ===", {
    prompt: prompt?.substring(0, 200),
    cwd,
    taskCount: tasks?.length ?? 0,
    workflow_id,
  });

  try {
    if (tasks && tasks.length > 0) {
      restoreTasks(tasks);
      log("info", "Restored " + tasks.length + " tasks to filesystem");
    }

    const output: string[] = [];
    let messageCount = 0;
    let toolUseCount = 0;
    let textBuffer = "";
    // Track pending tool calls to include required fields in results
    const pendingToolCalls = new Map<string, { toolName: string; input: unknown }>();

    for await (const msg of query({
      prompt,
      options: {
        systemPrompt: { type: "preset", preset: "claude_code" },
        tools: { type: "preset", preset: "claude_code" },
        settingSources: ["project"],
        permissionMode: "bypassPermissions" as any,
        cwd: cwd || process.cwd(),
      },
    })) {
      messageCount++;
      if (msg.type === "assistant") {
        for (const b of msg.message.content) {
          if (b.type === "thinking") {
            // Extended thinking blocks - publish as ReasoningUIPart
            const thinkingBlock = b as { type: "thinking"; thinking: string };
            if (workflow_id && thinkingBlock.thinking) {
              // Flush any pending text first to maintain order
              if (textBuffer.length > 0) {
                publishEvent(workflow_id, "part", {
                  type: "text",
                  text: textBuffer,
                  state: "done",
                }, "claude-code-agent");
                textBuffer = "";
              }
              publishEvent(workflow_id, "part", {
                type: "reasoning",
                text: thinkingBlock.thinking,
                state: "done",
              }, "claude-code-agent");
              log("debug", "Published reasoning part", {
                length: thinkingBlock.thinking.length,
              });
            }
          } else if (b.type === "text") {
            output.push(b.text);
            textBuffer += b.text;
            // Publish accumulated text as TextUIPart
            if (workflow_id && textBuffer.length > 0) {
              publishEvent(workflow_id, "part", {
                type: "text",
                text: textBuffer,
                state: "done",
              }, "claude-code-agent");
              textBuffer = "";
            }
          } else if (b.type === "tool_use") {
            // Flush any pending text before tool call to maintain order
            if (workflow_id && textBuffer.length > 0) {
              publishEvent(workflow_id, "part", {
                type: "text",
                text: textBuffer,
                state: "done",
              }, "claude-code-agent");
              textBuffer = "";
            }
            toolUseCount++;
            log("debug", "Tool use: " + b.name, {
              toolId: b.id,
              inputKeys: b.input
                ? Object.keys(b.input as Record<string, unknown>)
                : [],
            });
            // Track for correlation with result
            pendingToolCalls.set(b.id, { toolName: b.name, input: b.input });
            // Publish as DynamicToolUIPart with input-available state
            if (workflow_id) {
              publishEvent(workflow_id, "part", {
                type: "dynamic-tool",
                toolCallId: b.id,
                toolName: b.name,
                state: "input-available",
                input: b.input,
              }, "claude-code-agent");
              // Emit semantic events for plan/task tools
              if (b.name === "EnterPlanMode") {
                publishEvent(workflow_id, "plan_created", {
                  toolCallId: b.id,
                }, "claude-code-agent");
              }
            }
          }
        }
      } else if (msg.type === "user" && msg.message?.content) {
        // Handle tool results - they come back in user messages
        const content = msg.message.content;
        if (Array.isArray(content)) {
          for (const block of content) {
            if (block.type === "tool_result") {
              const toolResult = block as { type: "tool_result"; tool_use_id: string; content?: unknown; is_error?: boolean };
              log("debug", "Tool result received", {
                toolUseId: toolResult.tool_use_id,
                isError: toolResult.is_error,
              });
              // Get the pending tool call info for required fields
              const pending = pendingToolCalls.get(toolResult.tool_use_id);
              // Publish as DynamicToolUIPart with output state
              if (workflow_id) {
                const resultContent = typeof toolResult.content === "string"
                  ? toolResult.content
                  : JSON.stringify(toolResult.content);
                if (toolResult.is_error) {
                  publishEvent(workflow_id, "part", {
                    type: "dynamic-tool",
                    toolCallId: toolResult.tool_use_id,
                    toolName: pending?.toolName || "unknown",
                    state: "output-error",
                    input: pending?.input,
                    errorText: resultContent,
                  }, "claude-code-agent");
                } else {
                  publishEvent(workflow_id, "part", {
                    type: "dynamic-tool",
                    toolCallId: toolResult.tool_use_id,
                    toolName: pending?.toolName || "unknown",
                    state: "output-available",
                    input: pending?.input,
                    output: resultContent,
                  }, "claude-code-agent");
                  // Emit semantic events for task/plan tool results
                  if (pending?.toolName === "TaskCreate") {
                    try {
                      const taskInput = pending.input as Record<string, unknown>;
                      const taskData = {
                        id: (JSON.parse(resultContent) as any)?.id || String(Date.now()),
                        subject: taskInput?.subject as string || "",
                        description: taskInput?.description as string || "",
                        activeForm: taskInput?.activeForm as string,
                        status: "pending" as const,
                        blocks: taskInput?.blocks as string[] | undefined,
                        blockedBy: taskInput?.blockedBy as string[] | undefined,
                      };
                      publishEvent(workflow_id, "task_created", {
                        task: taskData,
                        toolCallId: toolResult.tool_use_id,
                      }, "claude-code-agent");
                    } catch {
                      publishEvent(workflow_id, "task_created", {
                        task: pending.input,
                        toolCallId: toolResult.tool_use_id,
                      }, "claude-code-agent");
                    }
                  } else if (pending?.toolName === "TaskUpdate") {
                    const updateInput = pending.input as Record<string, unknown>;
                    publishEvent(workflow_id, "task_updated", {
                      taskId: updateInput?.taskId as string,
                      taskStatus: updateInput?.status as string,
                      toolCallId: toolResult.tool_use_id,
                    }, "claude-code-agent");
                  } else if (pending?.toolName === "ExitPlanMode") {
                    publishEvent(workflow_id, "plan_complete", {
                      toolCallId: toolResult.tool_use_id,
                    }, "claude-code-agent");
                  }
                }
                if (pending) {
                  pendingToolCalls.delete(toolResult.tool_use_id);
                }
              }
            }
          }
        }
      }
    }

    // Flush any remaining text
    if (workflow_id && textBuffer.length > 0) {
      publishEvent(workflow_id, "part", {
        type: "text",
        text: textBuffer,
        state: "done",
      }, "claude-code-agent");
    }

    log("info", "SDK query completed", {
      messageCount,
      toolUseCount,
      outputLength: output.join("\n").length,
      elapsedMs: Date.now() - startTime,
    });

    clearTasks();

    const elapsed = Date.now() - startTime;
    log("info", "=== EXECUTE REQUEST COMPLETE === (" + elapsed + "ms)", {
      success: true,
      outputLength: output.join("\n").length,
    });

    res.json({ success: true, output: output.join("\n") });
  } catch (error: any) {
    const elapsed = Date.now() - startTime;
    log("error", "=== EXECUTE REQUEST FAILED === (" + elapsed + "ms)", {
      error: error.message,
      stack: error.stack,
    });
    res.status(500).json({ success: false, error: error.message });
  }
});

app.get("/health", (_req, res) => {
  res.json({ status: "healthy" });
});

// --- Task filesystem helpers ---

function readTasks(): any[] {
  if (!fs.existsSync(TASKS_DIR)) {
    log("debug", "readTasks: TASKS_DIR does not exist", { TASKS_DIR });
    return [];
  }

  const dirs = fs
    .readdirSync(TASKS_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory());

  if (dirs.length === 0) {
    log("debug", "readTasks: No subdirectories found in TASKS_DIR", {
      TASKS_DIR,
    });
    return [];
  }

  log("debug", "readTasks: Found " + dirs.length + " subdirectory(ies)", {
    dirNames: dirs.map((d) => d.name),
  });

  const taskDir = path.join(TASKS_DIR, dirs[0].name);
  const files = fs
    .readdirSync(taskDir)
    .filter((f) => f.endsWith(".json"))
    .sort((a, b) => {
      const numA = parseInt(path.basename(a, ".json"));
      const numB = parseInt(path.basename(b, ".json"));
      return numA - numB;
    });

  log("debug", "readTasks: Found " + files.length + " JSON file(s) in " + taskDir, {
    files,
  });

  return files.map((f) => {
    const filePath = path.join(taskDir, f);
    const content = fs.readFileSync(filePath, "utf-8");
    log("debug", "readTasks: Read task file " + f, {
      size: content.length,
      preview: content.substring(0, 200),
    });
    return JSON.parse(content);
  });
}

function restoreTasks(tasks: any[]): void {
  const taskListId = "workflow-tasks";
  const tasksPath = path.join(TASKS_DIR, taskListId);
  fs.mkdirSync(tasksPath, { recursive: true });

  log("debug", "restoreTasks: Writing " + tasks.length + " tasks to " + tasksPath);

  tasks.forEach((task, i) => {
    const filePath = path.join(tasksPath, (i + 1) + ".json");
    fs.writeFileSync(filePath, JSON.stringify(task, null, 2));
    log("debug", "restoreTasks: Wrote " + filePath, {
      taskId: task.id,
      subject: task.subject,
    });
  });

  process.env.CLAUDE_CODE_TASK_LIST_ID = taskListId;
  log("debug", "restoreTasks: Set CLAUDE_CODE_TASK_LIST_ID=" + taskListId);
}

function clearTasks(): void {
  if (!fs.existsSync(TASKS_DIR)) {
    log("debug", "clearTasks: TASKS_DIR does not exist, nothing to clear");
    return;
  }

  const dirs = fs
    .readdirSync(TASKS_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory());

  log("debug", "clearTasks: Clearing " + dirs.length + " subdirectory(ies)", {
    dirNames: dirs.map((d) => d.name),
  });

  for (const dir of dirs) {
    fs.rmSync(path.join(TASKS_DIR, dir.name), { recursive: true, force: true });
  }
}

const PORT = parseInt(process.env.PORT || "3000");
app.listen(PORT, () => {
  log("info", "Agent server listening on port " + PORT);
});
