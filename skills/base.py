"""Base classes and data models for the skills system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SkillDefinition:
    """Describes a skill's identity, tools, and prompt contribution."""

    name: str
    version: str
    description: str
    tools: list[Callable] = field(default_factory=list)
    system_prompt_snippet: str = ""
    allowed_tool_names: list[str] = field(default_factory=list)


class BaseSkill(ABC):
    """Abstract base class that all skills must implement."""

    @abstractmethod
    def get_definition(self) -> SkillDefinition:
        """Return the skill's definition."""
