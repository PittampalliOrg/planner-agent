import express from "express";
import { query } from "@anthropic-ai/claude-agent-sdk";
import * as fs from "node:fs";
import * as path from "node:path";

process.env.CLAUDE_CODE_ENABLE_TASKS = "true";

const app = express();
app.use(express.json({ limit: "10mb" }));

const TASKS_DIR = path.join(
  process.env.HOME || "/home/planner",
  ".claude",
  "tasks"
);

// POST /plan - Run the agent in plan mode, return created tasks
app.post("/plan", async (req, res) => {
  const { prompt, cwd } = req.body;

  try {
    const output: string[] = [];
    for await (const msg of query({
      prompt,
      options: {
        systemPrompt: { type: "preset", preset: "claude_code" },
        tools: { type: "preset", preset: "claude_code" },
        settingSources: ["project"],
        permissionMode: "plan" as any,
        cwd: cwd || process.cwd(),
      },
    })) {
      if (msg.type === "assistant") {
        for (const b of msg.message.content) {
          if (b.type === "text") output.push(b.text);
        }
      }
    }

    const tasks = readTasks();
    clearTasks();

    res.json({ success: true, tasks, output: output.join("\n") });
  } catch (error: any) {
    console.error("Planning failed:", error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// POST /execute - Restore tasks, run the agent in execution mode
app.post("/execute", async (req, res) => {
  const { prompt, cwd, tasks } = req.body;

  try {
    if (tasks && tasks.length > 0) {
      restoreTasks(tasks);
    }

    const output: string[] = [];
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
      if (msg.type === "assistant") {
        for (const b of msg.message.content) {
          if (b.type === "text") output.push(b.text);
        }
      }
    }

    clearTasks();

    res.json({ success: true, output: output.join("\n") });
  } catch (error: any) {
    console.error("Execution failed:", error);
    res.status(500).json({ success: false, error: error.message });
  }
});

app.get("/health", (_req, res) => {
  res.json({ status: "healthy" });
});

// --- Task filesystem helpers ---

function readTasks(): any[] {
  if (!fs.existsSync(TASKS_DIR)) return [];

  const dirs = fs
    .readdirSync(TASKS_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory());

  if (dirs.length === 0) return [];

  const taskDir = path.join(TASKS_DIR, dirs[0].name);
  const files = fs
    .readdirSync(taskDir)
    .filter((f) => f.endsWith(".json"))
    .sort((a, b) => {
      const numA = parseInt(path.basename(a, ".json"));
      const numB = parseInt(path.basename(b, ".json"));
      return numA - numB;
    });

  return files.map((f) => {
    const content = fs.readFileSync(path.join(taskDir, f), "utf-8");
    return JSON.parse(content);
  });
}

function restoreTasks(tasks: any[]): void {
  const taskListId = "workflow-tasks";
  const tasksPath = path.join(TASKS_DIR, taskListId);
  fs.mkdirSync(tasksPath, { recursive: true });

  tasks.forEach((task, i) => {
    fs.writeFileSync(
      path.join(tasksPath, `${i + 1}.json`),
      JSON.stringify(task, null, 2)
    );
  });

  process.env.CLAUDE_CODE_TASK_LIST_ID = taskListId;
}

function clearTasks(): void {
  if (!fs.existsSync(TASKS_DIR)) return;

  const dirs = fs
    .readdirSync(TASKS_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory());

  for (const dir of dirs) {
    fs.rmSync(path.join(TASKS_DIR, dir.name), { recursive: true, force: true });
  }
}

const PORT = parseInt(process.env.PORT || "3000");
app.listen(PORT, () => {
  console.log(`Agent server listening on port ${PORT}`);
});
