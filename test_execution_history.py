"""Tests for execution history and metrics tracking."""

import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

from task_manager import (
    EventType,
    Task,
    TaskEvent,
    TaskManager,
    TaskMetrics,
    TaskStatus,
)
from plan_manager import PlanManager


def test_task_event_creation():
    ev = TaskEvent(event_type=EventType.STARTED)
    assert ev.event_type == EventType.STARTED
    assert ev.timestamp is not None
    assert ev.duration_ms is None
    assert ev.error is None
    assert ev.metadata == {}


def test_task_event_serialization():
    ev = TaskEvent(
        event_type=EventType.FAILED,
        timestamp="2024-01-01T00:00:00",
        duration_ms=123.4,
        error="timeout",
        metadata={"retry": True},
    )
    d = ev.to_dict()
    assert d["event_type"] == "failed"
    assert d["duration_ms"] == 123.4
    assert d["error"] == "timeout"
    assert d["metadata"] == {"retry": True}

    restored = TaskEvent.from_dict(d)
    assert restored.event_type == EventType.FAILED
    assert restored.duration_ms == 123.4
    assert restored.error == "timeout"


def test_task_events_field():
    task = Task(id="1", subject="Test", description="desc")
    assert task.events == []


def test_task_to_dict_includes_events():
    ev = TaskEvent(event_type=EventType.STARTED, timestamp="2024-01-01T00:00:00")
    task = Task(id="1", subject="Test", description="desc", events=[ev])
    d = task.to_dict()
    assert "events" in d
    assert len(d["events"]) == 1
    assert d["events"][0]["event_type"] == "started"


def test_task_from_dict_restores_events():
    data = {
        "id": "1",
        "subject": "Test",
        "description": "desc",
        "events": [
            {"event_type": "started", "timestamp": "2024-01-01T00:00:00"},
            {"event_type": "completed", "timestamp": "2024-01-01T00:01:00", "duration_ms": 60000.0},
        ],
    }
    task = Task.from_dict(data)
    assert len(task.events) == 2
    assert task.events[0].event_type == EventType.STARTED
    assert task.events[1].event_type == EventType.COMPLETED
    assert task.events[1].duration_ms == 60000.0


def test_update_task_records_started_event():
    tm = TaskManager()
    task = tm.create_task(subject="Do thing", description="desc")
    assert len(task.events) == 0

    tm.update_task(task.id, status=TaskStatus.IN_PROGRESS)
    assert len(task.events) == 1
    assert task.events[0].event_type == EventType.STARTED


def test_update_task_records_completed_event():
    tm = TaskManager()
    task = tm.create_task(subject="Do thing", description="desc")
    tm.update_task(task.id, status=TaskStatus.IN_PROGRESS)
    tm.update_task(task.id, status=TaskStatus.COMPLETED)
    assert len(task.events) == 2
    assert task.events[1].event_type == EventType.COMPLETED
    assert task.events[1].duration_ms is not None


def test_update_task_records_failed_event():
    tm = TaskManager()
    task = tm.create_task(subject="Do thing", description="desc")
    tm.update_task(task.id, status=TaskStatus.IN_PROGRESS)
    tm.update_task(task.id, status=TaskStatus.PENDING)
    assert len(task.events) == 2
    assert task.events[1].event_type == EventType.FAILED


def test_update_task_records_retried_event():
    tm = TaskManager()
    task = tm.create_task(subject="Do thing", description="desc")
    tm.update_task(task.id, status=TaskStatus.IN_PROGRESS)
    tm.update_task(task.id, status=TaskStatus.COMPLETED)
    tm.update_task(task.id, status=TaskStatus.IN_PROGRESS)
    assert len(task.events) == 3
    assert task.events[2].event_type == EventType.RETRIED


def test_claim_task_records_started_event():
    tm = TaskManager()
    task = tm.create_task(subject="Do thing", description="desc")
    tm.claim_task(task.id, "agent-1")
    assert len(task.events) == 1
    assert task.events[0].event_type == EventType.STARTED


def test_record_task_event_manual():
    tm = TaskManager()
    task = tm.create_task(subject="Do thing", description="desc")
    ev = tm.record_task_event(
        task.id,
        EventType.BLOCKED,
        error="waiting on dependency",
        metadata={"blocker": "task-2"},
    )
    assert ev is not None
    assert ev.event_type == EventType.BLOCKED
    assert ev.error == "waiting on dependency"
    assert len(task.events) == 1


def test_record_task_event_returns_none_for_missing_task():
    tm = TaskManager()
    assert tm.record_task_event("nonexistent", EventType.STARTED) is None


def test_get_task_metrics_basic():
    tm = TaskManager()
    task = tm.create_task(subject="Do thing", description="desc")
    tm.update_task(task.id, status=TaskStatus.IN_PROGRESS)
    tm.update_task(task.id, status=TaskStatus.COMPLETED)

    metrics = tm.get_task_metrics(task.id)
    assert metrics is not None
    assert metrics.attempt_count == 1
    assert metrics.first_started_at is not None
    assert metrics.last_completed_at is not None
    assert metrics.error_count == 0
    assert metrics.total_duration_ms >= 0


def test_get_task_metrics_with_errors():
    tm = TaskManager()
    task = tm.create_task(subject="Flaky", description="desc")
    tm.update_task(task.id, status=TaskStatus.IN_PROGRESS)
    tm.update_task(task.id, status=TaskStatus.PENDING)
    tm.update_task(task.id, status=TaskStatus.IN_PROGRESS)
    tm.update_task(task.id, status=TaskStatus.COMPLETED)

    metrics = tm.get_task_metrics(task.id)
    assert metrics is not None
    assert metrics.attempt_count == 2
    assert metrics.error_count == 1


def test_get_task_metrics_blocked_duration():
    tm = TaskManager()
    task = tm.create_task(subject="Blocked", description="desc")

    t1 = datetime.now()
    t2 = t1 + timedelta(seconds=5)
    tm.record_task_event(task.id, EventType.BLOCKED, metadata={})
    task.events[-1].timestamp = t1.isoformat()
    tm.record_task_event(task.id, EventType.UNBLOCKED, metadata={})
    task.events[-1].timestamp = t2.isoformat()

    metrics = tm.get_task_metrics(task.id)
    assert metrics is not None
    assert metrics.blocked_duration_ms >= 4900


def test_get_task_metrics_returns_none_for_missing():
    tm = TaskManager()
    assert tm.get_task_metrics("nonexistent") is None


def test_get_plan_metrics():
    tm = TaskManager()
    t1 = tm.create_task(subject="Task 1", description="desc")
    t2 = tm.create_task(subject="Task 2", description="desc")

    tm.update_task(t1.id, status=TaskStatus.IN_PROGRESS)
    tm.update_task(t1.id, status=TaskStatus.COMPLETED)
    tm.update_task(t2.id, status=TaskStatus.IN_PROGRESS)
    tm.update_task(t2.id, status=TaskStatus.COMPLETED)

    plan_metrics = tm.get_plan_metrics()
    assert plan_metrics["total_tasks"] == 2
    assert plan_metrics["completed"] == 2
    assert plan_metrics["failed"] == 0
    assert plan_metrics["avg_duration_ms"] >= 0
    assert plan_metrics["total_wall_clock_ms"] >= 0


def test_get_plan_metrics_empty():
    tm = TaskManager()
    plan_metrics = tm.get_plan_metrics()
    assert plan_metrics["total_tasks"] == 0
    assert plan_metrics["completed"] == 0
    assert plan_metrics["avg_duration_ms"] == 0.0


def test_get_task_timeline():
    tm = TaskManager()
    task = tm.create_task(subject="Timeline task", description="desc")
    tm.update_task(task.id, status=TaskStatus.IN_PROGRESS)
    tm.update_task(task.id, status=TaskStatus.COMPLETED)

    timeline = tm.get_task_timeline(task.id)
    assert "Timeline task" in timeline
    assert "started" in timeline
    assert "completed" in timeline


def test_get_task_timeline_no_events():
    tm = TaskManager()
    task = tm.create_task(subject="Empty", description="desc")
    timeline = tm.get_task_timeline(task.id)
    assert "no events recorded" in timeline


def test_get_task_timeline_missing_task():
    tm = TaskManager()
    timeline = tm.get_task_timeline("nonexistent")
    assert "not found" in timeline


def test_save_load_preserves_events():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "tasks.json"

        tm = TaskManager(storage_path=path)
        task = tm.create_task(subject="Persist me", description="desc")
        tm.update_task(task.id, status=TaskStatus.IN_PROGRESS)
        tm.record_task_event(task.id, EventType.BLOCKED, error="dep missing")
        tm.save()

        tm2 = TaskManager(storage_path=path)
        tm2.load()
        loaded = tm2.get_task(task.id)
        assert loaded is not None
        assert len(loaded.events) == 2
        assert loaded.events[0].event_type == EventType.STARTED
        assert loaded.events[1].event_type == EventType.BLOCKED
        assert loaded.events[1].error == "dep missing"


def test_get_execution_summary():
    tm = TaskManager()
    pm = PlanManager()

    plan = pm.create_plan(
        title="Test Plan",
        summary="A test plan",
        steps=[
            {"title": "Step 1", "description": "First step"},
            {"title": "Step 2", "description": "Second step"},
        ],
    )
    tasks = pm.convert_plan_to_tasks(tm, plan)
    assert len(tasks) == 2

    tm.update_task(tasks[0].id, status=TaskStatus.IN_PROGRESS)
    tm.update_task(tasks[0].id, status=TaskStatus.COMPLETED)
    tm.update_task(tasks[1].id, status=TaskStatus.IN_PROGRESS)
    tm.update_task(tasks[1].id, status=TaskStatus.COMPLETED)

    summary = pm.get_execution_summary(tm, plan)
    assert "Test Plan" in summary
    assert "Step 1" in summary
    assert "Step 2" in summary
    assert "Completed" in summary or "completed" in summary.lower()


def test_get_execution_summary_with_errors():
    tm = TaskManager()
    pm = PlanManager()

    plan = pm.create_plan(
        title="Error Plan",
        summary="A plan with errors",
        steps=[{"title": "Failing step", "description": "Will fail"}],
    )
    tasks = pm.convert_plan_to_tasks(tm, plan)

    tm.update_task(tasks[0].id, status=TaskStatus.IN_PROGRESS)
    tm.record_task_event(tasks[0].id, EventType.FAILED, error="connection timeout")
    tm.update_task(tasks[0].id, status=TaskStatus.PENDING)

    summary = pm.get_execution_summary(tm, plan)
    assert "Errors" in summary
    assert "connection timeout" in summary


def test_get_execution_summary_no_plan():
    tm = TaskManager()
    pm = PlanManager()
    assert pm.get_execution_summary(tm) == "No plan available."


def test_event_type_enum_values():
    assert EventType.STARTED.value == "started"
    assert EventType.COMPLETED.value == "completed"
    assert EventType.FAILED.value == "failed"
    assert EventType.RETRIED.value == "retried"
    assert EventType.BLOCKED.value == "blocked"
    assert EventType.UNBLOCKED.value == "unblocked"


def test_task_metrics_dataclass():
    m = TaskMetrics(
        total_duration_ms=5000.0,
        attempt_count=2,
        first_started_at="2024-01-01T00:00:00",
        last_completed_at="2024-01-01T00:05:00",
        error_count=1,
        blocked_duration_ms=1000.0,
    )
    assert m.total_duration_ms == 5000.0
    assert m.attempt_count == 2
    assert m.error_count == 1


if __name__ == "__main__":
    import sys
    import traceback

    test_funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for fn in test_funcs:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    sys.exit(1 if failed else 0)
