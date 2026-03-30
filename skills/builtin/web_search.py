"""Built-in web search skill using the DuckDuckGo instant-answer API."""

from __future__ import annotations

import json
from typing import Any

import httpx
from claude_agent_sdk import tool

from skills.base import BaseSkill, SkillDefinition


@tool(
    "skill_web_search",
    "Search the web for information",
    {"query": str, "max_results": str},
)
async def skill_web_search(args: dict[str, Any]) -> dict[str, Any]:
    """Call the DuckDuckGo instant-answer API and return results as JSON."""
    query = args["query"]
    max_results = int(args.get("max_results", 5))

    url = f"https://api.duckduckgo.com/?q={query}&format=json"

    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        data = response.json()

    results: list[dict[str, Any]] = []

    if data.get("AbstractText"):
        results.append(
            {
                "type": "abstract",
                "title": data.get("Heading", ""),
                "text": data["AbstractText"],
                "url": data.get("AbstractURL", ""),
            }
        )

    for topic in data.get("RelatedTopics", [])[:max_results]:
        if "Text" in topic and "FirstURL" in topic:
            results.append(
                {
                    "type": "related",
                    "text": topic["Text"],
                    "url": topic["FirstURL"],
                }
            )

    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps({"query": query, "results": results}, indent=2),
            }
        ]
    }


class WebSearchSkill(BaseSkill):
    """Skill that provides web search capability via DuckDuckGo."""

    def get_definition(self) -> SkillDefinition:
        """Return the web search skill definition."""
        return SkillDefinition(
            name="web_search",
            version="1.0.0",
            description="Enables web search capability",
            tools=[skill_web_search],
            system_prompt_snippet=(
                "You can use the skill_web_search tool to search the web"
                " when you need current information."
            ),
            allowed_tool_names=["mcp__planner__skill_web_search"],
        )


def get_skill() -> BaseSkill:
    """Return an instance of the WebSearchSkill."""
    return WebSearchSkill()
