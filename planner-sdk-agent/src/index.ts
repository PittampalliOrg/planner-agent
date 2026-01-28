import { query } from "@anthropic-ai/claude-agent-sdk";

process.env.CLAUDE_CODE_ENABLE_TASKS = "true";

const prompt = process.argv.slice(2).join(" ") || "What can you help me with?";

for await (const msg of query({
  prompt,
  options: {
    systemPrompt: { type: "preset", preset: "claude_code" },
    tools: { type: "preset", preset: "claude_code" },
    settingSources: ["project"],
    permissionMode: (process.env.PERMISSION_MODE as any) ?? "default",
    cwd: process.env.CWD ?? process.cwd(),
    resume: process.env.SESSION_ID,
  },
})) {
  if (msg.type === "system" && msg.subtype === "init") {
    console.log(`Session: ${msg.session_id}\nTools: ${msg.tools.length}`);
  }
  if (msg.type === "assistant") {
    for (const b of msg.message.content) {
      if (b.type === "text") console.log(b.text);
      if (b.type === "tool_use") console.log(`[${b.name}]`);
    }
  }
  if ("result" in msg) console.log(`\n${msg.result}`);
}
