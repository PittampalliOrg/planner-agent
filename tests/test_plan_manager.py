"""Tests for PlanManager core logic."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from plan_manager import PlanManager, Plan, PlanStep
from task_manager import TaskManager, TaskStatus


def make_plan_manager(tmp_path):
    """Return a PlanManager backed by a temp directory."""
    return PlanManager(storage_dir=tmp_path / "plans")


def make_task_manager():
    """Return a TaskManager with no disk side-effects."""
    return TaskManager(storage_path="/tmp/tm_test_unused.json")


SAMPLE_STEPS = [
    {
        "title": "Step one",
        "description": "Do the first thing",
        "files_affected": ["a.py"],
        "estimated_complexity": "low",
    },
    {
        "title": "Step two",
        "description": "Do the second thing",
        "files_affected": ["b.py"],
        "complexity": "high",
    },
]


class TestCreatePlan:
    def test_basic_fields(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="My Plan", summary="A summary")
        assert plan.title == "My Plan"
        assert plan.summary == "A summary"
        assert plan.status == "draft"
        assert plan.id.startswith("plan_")

    def test_estimated_complexity_key(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="T", summary="S", steps=[SAMPLE_STEPS[0]])
        assert plan.steps[0].estimated_complexity == "low"

    def test_complexity_key_alias(self, tmp_path):
        """'complexity' key must be accepted as a fallback for 'estimated_complexity'."""
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="T", summary="S", steps=[SAMPLE_STEPS[1]])
        assert plan.steps[0].estimated_complexity == "high"

    def test_both_keys_present_prefers_estimated_complexity(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        step = {
            "title": "Mixed",
            "description": "desc",
            "estimated_complexity": "medium",
            "complexity": "high",
        }
        plan = pm.create_plan(title="T", summary="S", steps=[step])
        assert plan.steps[0].estimated_complexity == "medium"

    def test_default_complexity_when_neither_key(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(
            title="T", summary="S",
            steps=[{"title": "X", "description": "d"}],
        )
        assert plan.steps[0].estimated_complexity == "medium"

    def test_step_count(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="T", summary="S", steps=SAMPLE_STEPS)
        assert len(plan.steps) == 2

    def test_current_plan_set(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="T", summary="S")
        assert pm.current_plan is plan

    def test_critical_files_and_considerations(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(
            title="T", summary="S",
            critical_files=["main.py"],
            considerations=["Keep it simple"],
        )
        assert plan.critical_files == ["main.py"]
        assert plan.considerations == ["Keep it simple"]


class TestApprovePlan:
    def test_sets_status_approved(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="T", summary="S")
        pm.approve_plan(plan)
        assert plan.status == "approved"

    def test_sets_approved_at(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="T", summary="S")
        assert plan.approved_at is None
        pm.approve_plan(plan)
        assert plan.approved_at is not None

    def test_uses_current_plan_when_none_passed(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="T", summary="S")
        result = pm.approve_plan()
        assert result is plan
        assert plan.status == "approved"

    def test_returns_none_when_no_plan(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        assert pm.approve_plan() is None


class TestConvertPlanToTasks:
    def test_creates_correct_number_of_tasks(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        tm = make_task_manager()
        plan = pm.create_plan(title="T", summary="S", steps=SAMPLE_STEPS)
        tasks = pm.convert_plan_to_tasks(tm, plan)
        assert len(tasks) == len(SAMPLE_STEPS)

    def test_dependency_chain(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        tm = make_task_manager()
        plan = pm.create_plan(title="T", summary="S", steps=SAMPLE_STEPS)
        tasks = pm.convert_plan_to_tasks(tm, plan)

        # First task has no blockers
        assert tasks[0].blocked_by == []
        # Second task is blocked by the first
        assert tasks[0].id in tasks[1].blocked_by

    def test_three_step_chain(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        tm = make_task_manager()
        steps = [
            {"title": f"Step {i}", "description": f"desc {i}"} for i in range(1, 4)
        ]
        plan = pm.create_plan(title="T", summary="S", steps=steps)
        tasks = pm.convert_plan_to_tasks(tm, plan)

        assert tasks[0].blocked_by == []
        assert tasks[0].id in tasks[1].blocked_by
        assert tasks[1].id in tasks[2].blocked_by

    def test_plan_status_set_to_implementing(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        tm = make_task_manager()
        plan = pm.create_plan(title="T", summary="S", steps=SAMPLE_STEPS)
        pm.convert_plan_to_tasks(tm, plan)
        assert plan.status == "implementing"

    def test_task_subjects_match_step_titles(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        tm = make_task_manager()
        plan = pm.create_plan(title="T", summary="S", steps=SAMPLE_STEPS)
        tasks = pm.convert_plan_to_tasks(tm, plan)

        for task, step in zip(tasks, plan.steps):
            assert task.subject == step.title

    def test_metadata_contains_plan_id_and_step_number(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        tm = make_task_manager()
        plan = pm.create_plan(title="T", summary="S", steps=SAMPLE_STEPS)
        tasks = pm.convert_plan_to_tasks(tm, plan)

        for i, task in enumerate(tasks, 1):
            assert task.metadata["plan_id"] == plan.id
            assert task.metadata["step_number"] == i

    def test_returns_empty_list_when_no_plan(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        tm = make_task_manager()
        result = pm.convert_plan_to_tasks(tm)
        assert result == []


class TestSavePlan:
    def test_json_file_created(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="Save Test", summary="S")
        json_path = pm.save_plan(plan)
        assert json_path.exists()
        assert json_path.suffix == ".json"

    def test_markdown_file_created(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="Save Test", summary="S")
        json_path = pm.save_plan(plan)
        md_path = json_path.with_suffix(".md")
        assert md_path.exists()

    def test_json_content_is_valid(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="T", summary="S", steps=SAMPLE_STEPS)
        json_path = pm.save_plan(plan)
        with open(json_path) as f:
            data = json.load(f)
        assert data["title"] == "T"
        assert len(data["steps"]) == 2

    def test_markdown_contains_title(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="My Feature", summary="S")
        json_path = pm.save_plan(plan)
        md_path = json_path.with_suffix(".md")
        content = md_path.read_text()
        assert "My Feature" in content

    def test_custom_filename(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="T", summary="S")
        json_path = pm.save_plan(plan, filename="custom_name")
        assert json_path.name == "custom_name.json"

    def test_storage_dir_created_if_missing(self, tmp_path):
        storage = tmp_path / "deep" / "nested" / "plans"
        pm = PlanManager(storage_dir=storage)
        plan = pm.create_plan(title="T", summary="S")
        pm.save_plan(plan)
        assert storage.exists()


class TestLoadPlan:
    def test_load_existing_plan(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="Load Me", summary="S", steps=SAMPLE_STEPS)
        pm.save_plan(plan)

        pm2 = PlanManager(storage_dir=pm.storage_dir)
        loaded = pm2.load_plan(plan.id)

        assert loaded is not None
        assert loaded.title == "Load Me"
        assert len(loaded.steps) == 2

    def test_load_sets_current_plan(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="T", summary="S")
        pm.save_plan(plan)

        pm2 = PlanManager(storage_dir=pm.storage_dir)
        loaded = pm2.load_plan(plan.id)
        assert pm2.current_plan is loaded

    def test_load_missing_file_returns_none(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        result = pm.load_plan("nonexistent_plan")
        assert result is None

    def test_load_with_json_extension(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="T", summary="S")
        pm.save_plan(plan)

        pm2 = PlanManager(storage_dir=pm.storage_dir)
        loaded = pm2.load_plan(f"{plan.id}.json")
        assert loaded is not None

    def test_load_preserves_step_complexity(self, tmp_path):
        pm = make_plan_manager(tmp_path)
        plan = pm.create_plan(title="T", summary="S", steps=SAMPLE_STEPS)
        pm.save_plan(plan)

        pm2 = PlanManager(storage_dir=pm.storage_dir)
        loaded = pm2.load_plan(plan.id)
        assert loaded.steps[0].estimated_complexity == "low"
