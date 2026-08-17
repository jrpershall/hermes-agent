from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from cron import executions, scheduler
from tools.environments import local


@pytest.fixture
def execution_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cron" / "executions.db"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", path)
    return path


def test_schema_migrates_managed_execution_columns(execution_ledger: Path) -> None:
    executions.create_execution("ordinary", source="test")
    with sqlite3.connect(execution_ledger) as database:
        columns = {
            str(row[1]) for row in database.execute("PRAGMA table_info(executions)")
        }
    assert {
        "worker_started_at",
        "lane_id",
        "session_sha256",
        "capability_sha256",
    } <= columns


def test_managed_capability_is_hashed_and_worker_start_is_durable(
    execution_ledger: Path,
) -> None:
    record, capability = executions.create_managed_execution(
        "review-job", source="test", lane_id="review-2"
    )
    assert capability not in json.dumps(record, sort_keys=True)
    assert record["lane_id"] == "review-2"
    assert record["capability_sha256"] == hashlib.sha256(
        capability.encode("utf-8")
    ).hexdigest()
    assert len(record["session_sha256"]) == 64
    assert record["worker_started_at"] is None

    assert executions.mark_execution_running(record["id"])["status"] == "running"
    started = executions.mark_managed_worker_started(record["id"])
    assert started is not None
    assert started["worker_started_at"]
    assert executions.mark_managed_worker_started(record["id"]) is None

    persisted = executions.latest_execution("review-job")
    assert persisted is not None
    assert capability not in json.dumps(persisted, sort_keys=True)


def test_ordinary_execution_remains_unmanaged(execution_ledger: Path) -> None:
    record = executions.create_execution("ordinary", source="test")
    assert record["lane_id"] is None
    assert record["session_sha256"] is None
    assert record["capability_sha256"] is None
    assert executions.mark_execution_running(record["id"])
    assert executions.mark_managed_worker_started(record["id"]) is None


def test_scheduler_requires_exact_managed_context_shape() -> None:
    assert scheduler._managed_execution_lane({}) is None
    context = {
        "lane_id": "review-3",
        "execution_id_env": "HERMES_MANAGED_EXECUTION_ID",
        "execution_capability_env": "HERMES_MANAGED_EXECUTION_CAPABILITY",
    }
    assert scheduler._managed_execution_lane(
        {"managed_execution_context": context}
    ) == "review-3"
    with pytest.raises(ValueError, match="malformed"):
        scheduler._managed_execution_lane(
            {"managed_execution_context": {**context, "lane_id": ""}}
        )
    with pytest.raises(ValueError, match="malformed"):
        scheduler._managed_execution_lane(
            {
                "managed_execution_context": {
                    **context,
                    "execution_capability_env": "ATTACKER_CHOSEN_NAME",
                }
            }
        )


def test_capability_is_context_local_and_absent_after_reset() -> None:
    stale = {
        "HERMES_MANAGED_EXECUTION_ID": "foreign",
        "HERMES_MANAGED_EXECUTION_CAPABILITY": "foreign-secret",
    }
    unbound = local.build_subprocess_env(stale)
    assert "HERMES_MANAGED_EXECUTION_ID" not in unbound
    assert "HERMES_MANAGED_EXECUTION_CAPABILITY" not in unbound

    token = local.bind_managed_execution_env("execution-1", "private-capability")
    try:
        bound = local.build_subprocess_env({})
        assert bound["HERMES_MANAGED_EXECUTION_ID"] == "execution-1"
        assert (
            bound["HERMES_MANAGED_EXECUTION_CAPABILITY"] == "private-capability"
        )
    finally:
        local.reset_managed_execution_env(token)
    reset = local.build_subprocess_env(stale)
    assert "HERMES_MANAGED_EXECUTION_ID" not in reset
    assert "HERMES_MANAGED_EXECUTION_CAPABILITY" not in reset


def test_pre_run_script_receives_only_bound_execution_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    probe = scripts / "probe.py"
    probe.write_text(
        "import json, os\n"
        "print(json.dumps({k: os.environ.get(k) for k in "
        "('HERMES_MANAGED_EXECUTION_ID', "
        "'HERMES_MANAGED_EXECUTION_CAPABILITY')}))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)

    token = local.bind_managed_execution_env("exec-2", "secret-2")
    try:
        ok, output = scheduler._run_job_script("probe.py")
    finally:
        local.reset_managed_execution_env(token)
    assert ok is True
    assert json.loads(output) == {
        "HERMES_MANAGED_EXECUTION_ID": "exec-2",
        "HERMES_MANAGED_EXECUTION_CAPABILITY": "secret-2",
    }

    ok, output = scheduler._run_job_script("probe.py")
    assert ok is True
    assert json.loads(output) == {
        "HERMES_MANAGED_EXECUTION_ID": None,
        "HERMES_MANAGED_EXECUTION_CAPABILITY": None,
    }


def test_run_one_job_binds_and_closes_managed_execution(
    execution_ledger: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = {
        "lane_id": "review-1",
        "execution_id_env": "HERMES_MANAGED_EXECUTION_ID",
        "execution_capability_env": "HERMES_MANAGED_EXECUTION_CAPABILITY",
    }
    observed: dict[str, str] = {}

    def run_job(job: dict, **_kwargs: object) -> tuple[bool, str, str, None]:
        env = local.build_subprocess_env({})
        observed["execution_id"] = env["HERMES_MANAGED_EXECUTION_ID"]
        observed["capability"] = env["HERMES_MANAGED_EXECUTION_CAPABILITY"]
        return True, "output", "[SILENT]", None

    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(scheduler, "run_job", run_job)
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: tmp_path / "out")
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: None)

    assert scheduler.run_one_job(
        {"id": "review-job", "managed_execution_context": context}
    )

    record = executions.latest_execution("review-job")
    assert record is not None
    assert record["id"] == observed["execution_id"]
    assert record["status"] == "completed"
    assert record["worker_started_at"]
    assert record["lane_id"] == "review-1"
    assert record["capability_sha256"] == hashlib.sha256(
        observed["capability"].encode("utf-8")
    ).hexdigest()
    assert observed["capability"] not in json.dumps(record, sort_keys=True)
    assert "HERMES_MANAGED_EXECUTION_ID" not in local.build_subprocess_env({})
    assert "HERMES_MANAGED_EXECUTION_CAPABILITY" not in local.build_subprocess_env({})


def test_tick_rejects_malformed_managed_context_without_registration_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = {
        "id": "invalid-managed-job",
        "managed_execution_context": {
            "lane_id": "review-1",
            "execution_id_env": "ATTACKER_SELECTED",
            "execution_capability_env": "HERMES_MANAGED_EXECUTION_CAPABILITY",
        },
    }
    monkeypatch.setattr(scheduler, "get_due_jobs", lambda: [job])
    monkeypatch.setattr(scheduler, "advance_next_runs", lambda _ids: 1)
    monkeypatch.setattr(
        scheduler,
        "create_managed_execution",
        lambda *_args, **_kwargs: pytest.fail("invalid job must not create an execution"),
    )

    assert scheduler.tick(verbose=False, sync=True) == 0
    assert "invalid-managed-job" not in scheduler.get_running_job_ids()
