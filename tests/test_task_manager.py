"""Tests for TaskManager core logic."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_manager import TaskManager, TaskStatus


def make_manager():
    """Return a fresh TaskManager with no storage path side-effects."""
    return TaskManager(storage_path="/tmp/tasks_test_unused.json")


class TestCreateTask:
    def test_defaults(self):
        tm = make_manager()
        task = tm.create_task(subject="Do something", description="Details here")

        assert task.id == "1"
        assert task.subject == "Do something"
        assert task.description == "Details here"
        assert task.status == TaskStatus.PENDING
        assert task.blocks == []
        assert task.blocked_by == []
        assert task.owner is None

    def test_active_form_default(self):
        tm = make_manager()
        task = tm.create_task(subject="Fix bug", description="desc")
        assert task.active_form == "Fix bug..."

    def test_active_form_custom(self):
        tm = make_manager()
        task = tm.create_task(subject="Fix bug", description="desc", active_form="Fixing bug")
        assert task.active_form == "Fixing bug"

    def test_metadata_stored(self):
        tm = make_manager()
        task = tm.create_task(subject="X", description="Y", metadata={"key": "val"})
        assert task.metadata["key"] == "val"

    def test_ids_increment(self):
        tm = make_manager()
        t1 = tm.create_task(subject="A", description="a")
        t2 = tm.create_task(subject="B", description="b")
        assert t1.id == "1"
        assert t2.id == "2"

    def test_task_stored_in_manager(self):
        tm = make_manager()
        task = tm.create_task(subject="A", description="a")
        assert tm.get_task(task.id) is task


class TestUpdateTask:
    def test_update_status_to_in_progress(self):
        tm = make_manager()
        task = tm.create_task(subject="A", description="a")
        updated = tm.update_task(task.id, status=TaskStatus.IN_PROGRESS)
        assert updated.status == TaskStatus.IN_PROGRESS

    def test_update_status_by_string(self):
        tm = make_manager()
        task = tm.create_task(subject="A", description="a")
        updated = tm.update_task(task.id, status="completed")
        assert updated.status == TaskStatus.COMPLETED

    def test_update_subject_and_description(self):
        tm = make_manager()
        task = tm.create_task(subject="Old", description="old desc")
        tm.update_task(task.id, subject="New", description="new desc")
        assert task.subject == "New"
        assert task.description == "new desc"

    def test_update_nonexistent_task_returns_none(self):
        tm = make_manager()
        result = tm.update_task("999", status=TaskStatus.COMPLETED)
        assert result is None

    def test_status_transitions(self):
        tm = make_manager()
        task = tm.create_task(subject="A", description="a")
        tm.update_task(task.id, status=TaskStatus.IN_PROGRESS)
        assert task.status == TaskStatus.IN_PROGRESS
        tm.update_task(task.id, status=TaskStatus.COMPLETED)
        assert task.status == TaskStatus.COMPLETED


class TestResolveDependencies:
    def test_completing_task_removes_it_from_blocked_by(self):
        tm = make_manager()
        t1 = tm.create_task(subject="First", description="first")
        t2 = tm.create_task(subject="Second", description="second")
        tm.update_task(t2.id, add_blocked_by=[t1.id])

        assert t1.id in t2.blocked_by

        tm.update_task(t1.id, status=TaskStatus.COMPLETED)

        assert t1.id not in t2.blocked_by

    def test_completing_task_does_not_affect_unrelated_tasks(self):
        tm = make_manager()
        t1 = tm.create_task(subject="First", description="first")
        t2 = tm.create_task(subject="Second", description="second")
        t3 = tm.create_task(subject="Third", description="third")
        tm.update_task(t3.id, add_blocked_by=[t2.id])

        tm.update_task(t1.id, status=TaskStatus.COMPLETED)

        assert t2.id in t3.blocked_by

    def test_add_blocked_by_updates_blocker_blocks_list(self):
        tm = make_manager()
        t1 = tm.create_task(subject="First", description="first")
        t2 = tm.create_task(subject="Second", description="second")
        tm.update_task(t2.id, add_blocked_by=[t1.id])

        assert t2.id in t1.blocks


class TestGetAvailableTasks:
    def test_unblocked_pending_task_is_available(self):
        tm = make_manager()
        task = tm.create_task(subject="Free task", description="no deps")
        assert task in tm.get_available_tasks()

    def test_blocked_task_is_not_available(self):
        tm = make_manager()
        t1 = tm.create_task(subject="Blocker", description="first")
        t2 = tm.create_task(subject="Blocked", description="second")
        tm.update_task(t2.id, add_blocked_by=[t1.id])

        available = tm.get_available_tasks()
        assert t2 not in available
        assert t1 in available

    def test_task_becomes_available_after_blocker_completes(self):
        tm = make_manager()
        t1 = tm.create_task(subject="Blocker", description="first")
        t2 = tm.create_task(subject="Blocked", description="second")
        tm.update_task(t2.id, add_blocked_by=[t1.id])

        tm.update_task(t1.id, status=TaskStatus.COMPLETED)

        available_ids = [t.id for t in tm.get_available_tasks()]
        assert t2.id in available_ids

    def test_in_progress_task_not_available(self):
        tm = make_manager()
        task = tm.create_task(subject="Active", description="desc")
        tm.update_task(task.id, status=TaskStatus.IN_PROGRESS)
        assert task not in tm.get_available_tasks()

    def test_completed_task_not_available(self):
        tm = make_manager()
        task = tm.create_task(subject="Done", description="desc")
        tm.update_task(task.id, status=TaskStatus.COMPLETED)
        assert task not in tm.get_available_tasks()

    def test_claimed_task_not_available(self):
        tm = make_manager()
        task = tm.create_task(subject="Claimed", description="desc")
        tm.update_task(task.id, owner="agent-1")
        assert task not in tm.get_available_tasks()


class TestSaveLoadRoundTrip:
    def test_round_trip(self, tmp_path):
        storage = tmp_path / "tasks.json"
        tm = TaskManager(storage_path=storage)
        t1 = tm.create_task(subject="Alpha", description="first task")
        t2 = tm.create_task(subject="Beta", description="second task")
        tm.update_task(t2.id, add_blocked_by=[t1.id])
        tm.update_task(t1.id, status=TaskStatus.IN_PROGRESS)

        tm.save()

        tm2 = TaskManager(storage_path=storage)
        tm2.load()

        assert len(tm2.tasks) == 2
        loaded_t1 = tm2.get_task(t1.id)
        loaded_t2 = tm2.get_task(t2.id)

        assert loaded_t1.subject == "Alpha"
        assert loaded_t1.status == TaskStatus.IN_PROGRESS
        assert t1.id in loaded_t2.blocked_by
        assert t2.id in loaded_t1.blocks

    def test_next_id_preserved(self, tmp_path):
        storage = tmp_path / "tasks.json"
        tm = TaskManager(storage_path=storage)
        tm.create_task(subject="A", description="a")
        tm.create_task(subject="B", description="b")
        tm.save()

        tm2 = TaskManager(storage_path=storage)
        tm2.load()
        t3 = tm2.create_task(subject="C", description="c")
        assert t3.id == "3"

    def test_load_missing_file_does_nothing(self, tmp_path):
        storage = tmp_path / "nonexistent.json"
        tm = TaskManager(storage_path=storage)
        tm.load()
        assert tm.tasks == {}
