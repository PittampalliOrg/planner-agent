"""Registry for dynamic skill management."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Callable

from skills.base import BaseSkill, SkillDefinition

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Manages registration and discovery of skills."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    def register(self, skill: BaseSkill | SkillDefinition) -> None:
        """Register a skill, raising ValueError if the name is already taken."""
        if isinstance(skill, BaseSkill):
            definition = skill.get_definition()
        else:
            definition = skill

        if definition.name in self._skills:
            raise ValueError(f"Skill '{definition.name}' is already registered")

        self._skills[definition.name] = definition
        logger.debug("Registered skill '%s' v%s", definition.name, definition.version)

    def unregister(self, name: str) -> bool:
        """Remove a skill by name; returns True if it existed."""
        if name in self._skills:
            del self._skills[name]
            logger.debug("Unregistered skill '%s'", name)
            return True
        return False

    def get(self, name: str) -> SkillDefinition | None:
        """Return the SkillDefinition for *name*, or None if not found."""
        return self._skills.get(name)

    def list_skills(self) -> list[SkillDefinition]:
        """Return all registered skill definitions."""
        return list(self._skills.values())

    def load_from_directory(self, path: Path | str) -> list[str]:
        """Import every .py module in *path*, auto-registering skills via get_skill()."""
        directory = Path(path)
        if not directory.is_dir():
            raise NotADirectoryError(f"'{directory}' is not a directory")

        loaded: list[str] = []
        for module_path in sorted(directory.glob("*.py")):
            if module_path.name.startswith("_"):
                continue

            module_name = f"_skill_dynamic_{module_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                logger.warning("Could not load spec for '%s', skipping", module_path)
                continue

            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)  # type: ignore[union-attr]
            except Exception:
                logger.exception("Error executing module '%s', skipping", module_path)
                continue

            get_skill_fn = getattr(module, "get_skill", None)
            if not callable(get_skill_fn):
                logger.debug("No get_skill() in '%s', skipping", module_path.name)
                continue

            try:
                skill = get_skill_fn()
                self.register(skill)
                name = (
                    skill.get_definition().name
                    if isinstance(skill, BaseSkill)
                    else skill.name
                )
                loaded.append(name)
            except Exception:
                logger.exception("Error registering skill from '%s', skipping", module_path)

        return loaded

    def get_all_tools(self) -> list[Callable]:
        """Return all tool callables from every registered skill."""
        tools: list[Callable] = []
        for definition in self._skills.values():
            tools.extend(definition.tools)
        return tools

    def get_all_allowed_tool_names(self) -> list[str]:
        """Return all MCP tool name strings from every registered skill."""
        names: list[str] = []
        for definition in self._skills.values():
            names.extend(definition.allowed_tool_names)
        return names

    def get_combined_system_prompt_snippet(self) -> str:
        """Return all non-empty prompt snippets joined with newlines."""
        snippets = [
            d.system_prompt_snippet
            for d in self._skills.values()
            if d.system_prompt_snippet
        ]
        return "\n".join(snippets)


skill_registry = SkillRegistry()
